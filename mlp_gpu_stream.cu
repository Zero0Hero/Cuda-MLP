

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>

const int INPUT_SIZE = 64*64;
const int HIDDEN_SIZE = 512;
const int OUTPUT_SIZE = 10;

// CPU 实现
void matrixMultiply_CPU(float *a, float *b, float *c, int m, int n, int k) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			float value = 0.0f;
			for (int l = 0; l < n; ++l) {
				value += a[i * n + l] * b[l * k + j];
			}
			c[i * k + j] = value;
		}
	}
}
void addBias_CPU(float *output, float *bias, int size) {
	for (int i = 0; i < size; ++i) {
		output[i] += bias[i];
		if (output[i]<0)output[i]=0;
	}
}


// GPU 实现
__global__ void matrixMultiply(float *a, float *b, float *c, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < k) {
		float value = 0.0f;
		for (int i = 0; i < n; i++) {
			value += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = value;
	}
}

__global__ void addBias(float *output, float *bias, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		output[idx] += bias[idx];
		if (output[idx] < 0)output[idx] = 0;
	}
}

// GPU
class MLP_GPU_Stream {
public:
	MLP_GPU_Stream() {
		// 分配 GPU 端内存
		cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float));
		cudaMalloc((void**)&d_hiddenWeights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
		cudaMalloc((void**)&d_hiddenBias, HIDDEN_SIZE * sizeof(float));
		cudaMalloc((void**)&d_hiddenOutput0, HIDDEN_SIZE * sizeof(float)); cudaMalloc((void**)&d_hiddenOutput1, HIDDEN_SIZE * sizeof(float));
		cudaMalloc((void**)&d_outputWeights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
		cudaMalloc((void**)&d_outputBias, OUTPUT_SIZE * sizeof(float));
		cudaMalloc((void**)&d_output, OUTPUT_SIZE * 2 * sizeof(float));
		cudaStreamCreate(&stream0);
		cudaStreamCreate(&stream1);

	}

	~MLP_GPU_Stream() {
		// 释放 GPU 端内存
		cudaFree(d_input);
		cudaFree(d_hiddenWeights);
		cudaFree(d_hiddenBias);
		cudaFree(d_hiddenOutput0); cudaFree(d_hiddenOutput1);
		cudaFree(d_outputWeights);
		cudaFree(d_outputBias);
		cudaFree(d_output);
		cudaStreamDestroy(stream0);
		cudaStreamDestroy(stream1);
	}

	// 
	void Init(float *hiddenWeights, float *hiddenBias, float *outputWeights, float *outputBias)
	{
		cudaMemcpyAsync(d_hiddenWeights, hiddenWeights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_hiddenBias, hiddenBias, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_outputWeights, outputWeights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_outputBias, outputBias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaDeviceSynchronize();
	}

	void setInput(float *input)
	{
		cudaMemcpyAsync(d_input, input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_input + INPUT_SIZE, input + INPUT_SIZE, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
	}

	void forward() {
		matrixMultiply << <(HIDDEN_SIZE + 255) / 256, 256, 0, stream0 >> > (d_input, d_hiddenWeights, d_hiddenOutput0, 1, INPUT_SIZE, HIDDEN_SIZE);
		addBias << <(HIDDEN_SIZE + 255) / 256, 256, 0, stream0 >> > (d_hiddenOutput0, d_hiddenBias, HIDDEN_SIZE);
		matrixMultiply << <(OUTPUT_SIZE + 255) / 256, 256, 0, stream0 >> > (d_hiddenOutput0, d_outputWeights, d_output, 1, HIDDEN_SIZE, OUTPUT_SIZE);
		addBias << <(OUTPUT_SIZE + 255) / 256, 256, 0, stream0 >> > (d_output, d_outputBias, OUTPUT_SIZE);

		matrixMultiply << <(HIDDEN_SIZE + 255) / 256, 256, 0, stream1 >> > (d_input + INPUT_SIZE, d_hiddenWeights, d_hiddenOutput1, 1, INPUT_SIZE, HIDDEN_SIZE);
		addBias << <(HIDDEN_SIZE + 255) / 256, 256, 0, stream1 >> > (d_hiddenOutput1, d_hiddenBias, HIDDEN_SIZE);
		matrixMultiply << <(OUTPUT_SIZE + 255) / 256, 256, 0, stream1 >> > (d_hiddenOutput1, d_outputWeights, d_output + OUTPUT_SIZE, 1, HIDDEN_SIZE, OUTPUT_SIZE);
		addBias << <(OUTPUT_SIZE + 255) / 256, 256, 0, stream1 >> > (d_output + OUTPUT_SIZE, d_outputBias, OUTPUT_SIZE);
	}

	void getOutput(float *output) {
		cudaMemcpyAsync(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(output + OUTPUT_SIZE, d_output + OUTPUT_SIZE, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream1);
		cudaDeviceSynchronize();
	}

private:
	float *d_input;
	float *d_hiddenWeights;
	float *d_hiddenBias;
	float *d_hiddenOutput0;
	float * d_hiddenOutput1;

	float *d_outputWeights;
	float *d_outputBias;
	float *d_output;

	cudaStream_t stream0, stream1;
};

//GPU
class MLP_GPU {
public:
	MLP_GPU() {
		// 分配 GPU 端内存
		cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float));
		cudaMalloc((void**)&d_hiddenWeights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
		cudaMalloc((void**)&d_hiddenBias, HIDDEN_SIZE * sizeof(float));
		cudaMalloc((void**)&d_hiddenOutput0, HIDDEN_SIZE * sizeof(float)); cudaMalloc((void**)&d_hiddenOutput1, HIDDEN_SIZE * sizeof(float));
		cudaMalloc((void**)&d_outputWeights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
		cudaMalloc((void**)&d_outputBias, OUTPUT_SIZE * sizeof(float));
		cudaMalloc((void**)&d_output, OUTPUT_SIZE * 2 * sizeof(float));
	}

	~MLP_GPU() {
		// 释放 GPU 端内存
		cudaFree(d_input);
		cudaFree(d_hiddenWeights);
		cudaFree(d_hiddenBias);
		cudaFree(d_hiddenOutput0); cudaFree(d_hiddenOutput1);
		cudaFree(d_outputWeights);
		cudaFree(d_outputBias);
		cudaFree(d_output);
	}

	// 
	void Init(float *hiddenWeights, float *hiddenBias, float *outputWeights, float *outputBias)
	{
		cudaMemcpy(d_hiddenWeights, hiddenWeights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_hiddenBias, hiddenBias, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_outputWeights, outputWeights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_outputBias, outputBias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	}

	// 输入数据
	void setInput(float *input)
	{
		cudaMemcpyAsync(d_input, input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_input + INPUT_SIZE, input + INPUT_SIZE, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	}

	// 前向传播
	void forward() {
		matrixMultiply << <(HIDDEN_SIZE + 255) / 256, 256, 0, 0 >> > (d_input, d_hiddenWeights, d_hiddenOutput0, 1, INPUT_SIZE, HIDDEN_SIZE);
		addBias << <(HIDDEN_SIZE + 255) / 256, 256, 0, 0 >> > (d_hiddenOutput0, d_hiddenBias, HIDDEN_SIZE);
		matrixMultiply << <(OUTPUT_SIZE + 255) / 256, 256, 0, 0 >> > (d_hiddenOutput0, d_outputWeights, d_output, 1, HIDDEN_SIZE, OUTPUT_SIZE);
		addBias << <(OUTPUT_SIZE + 255) / 256, 256, 0, 0 >> > (d_output, d_outputBias, OUTPUT_SIZE);

		matrixMultiply << <(HIDDEN_SIZE + 255) / 256, 256, 0, 0 >> > (d_input + INPUT_SIZE, d_hiddenWeights, d_hiddenOutput1, 1, INPUT_SIZE, HIDDEN_SIZE);
		addBias << <(HIDDEN_SIZE + 255) / 256, 256, 0, 0 >> > (d_hiddenOutput1, d_hiddenBias, HIDDEN_SIZE);
		matrixMultiply << <(OUTPUT_SIZE + 255) / 256, 256, 0, 0 >> > (d_hiddenOutput1, d_outputWeights, d_output + OUTPUT_SIZE, 1, HIDDEN_SIZE, OUTPUT_SIZE);
		addBias << <(OUTPUT_SIZE + 255) / 256, 256, 0, 0 >> > (d_output + OUTPUT_SIZE, d_outputBias, OUTPUT_SIZE);
	}

	// 获取输出数据
	void getOutput(float *output) {
		cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(output + OUTPUT_SIZE, d_output + OUTPUT_SIZE, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	}

private:
	float *d_input;
	float *d_hiddenWeights;
	float *d_hiddenBias;
	float *d_hiddenOutput0;
	float * d_hiddenOutput1;

	float *d_outputWeights;
	float *d_outputBias;
	float *d_output;
};


////
class MLP_CPU {
public:
	MLP_CPU() {
	}

	~MLP_CPU() {
	}
	// 执行前向传播
	void forward(float *input_data, float *hiddenWeights, float *hiddenBias, float *outputWeights, float *outputBias, float *output) {

		// 计算隐藏层输出
		matrixMultiply_CPU(input_data, hiddenWeights, hiddenOutput, 1, INPUT_SIZE, HIDDEN_SIZE);
		addBias_CPU(hiddenOutput, hiddenBias, HIDDEN_SIZE);

		// 计算输出层输出
		matrixMultiply_CPU(hiddenOutput, outputWeights, output, 1, HIDDEN_SIZE, OUTPUT_SIZE);
		addBias_CPU(output, outputBias, OUTPUT_SIZE);

		// 在这里可以添加激活函数，如sigmoid或ReLU
	}

private:
	float hiddenOutput[HIDDEN_SIZE] = { 0 };
};

int main() {

	float hiddenWeights[INPUT_SIZE * HIDDEN_SIZE];
	float hiddenBias[HIDDEN_SIZE];
	float outputWeights[HIDDEN_SIZE * OUTPUT_SIZE];
	float outputBias[OUTPUT_SIZE];

	// 随机初始化
	for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
		hiddenWeights[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	for (int i = 0; i < HIDDEN_SIZE; i++) {
		hiddenBias[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
		outputWeights[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		outputBias[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	clock_t sT_C, eT_C, sT_G, eT_G, sT_GS, eT_GS;
	// GPU 模型
	MLP_GPU_Stream mlp_gpu_steam;
	MLP_GPU mlp_gpu;
	MLP_CPU mlp_cpu;
	// 准备输入数据（为了简化，这里使用随机数据）
	float *input = new float[INPUT_SIZE * 2];
	for (int i = 0; i < INPUT_SIZE * 2; i++) {
		input[i] = static_cast<float>(rand()) / RAND_MAX;
	}
#define N 100
	////
	float *output_gs = new float[OUTPUT_SIZE * 2];
	sT_GS = clock();
	mlp_gpu_steam.Init(hiddenWeights, hiddenBias, outputWeights, outputBias);
	for (int ii = 0; ii < N; ii++)
	{
		mlp_gpu_steam.setInput(input);
		mlp_gpu_steam.forward();
	}
	mlp_gpu_steam.getOutput(output_gs);
	eT_GS = clock();
	std::cout << "GPU-Stream=  ";
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		std::cout << output_gs[i] << ' ';
	}
	std::cout << std::endl;
	std::cout << eT_GS - sT_GS << "ms" << std::endl;
	////

	////
	float *output = new float[OUTPUT_SIZE * 2];
	sT_G = clock();
	mlp_gpu.Init(hiddenWeights, hiddenBias, outputWeights, outputBias);
	for (int ii = 0; ii < N; ii++)
	{
		mlp_gpu.setInput(input);
		mlp_gpu.forward();
	}
	mlp_gpu.getOutput(output);
	eT_G = clock();
	std::cout << "GPU=  ";
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		std::cout << output[i] << ' ';
	}
	std::cout << std::endl;
	std::cout << eT_G - sT_G << "ms" << std::endl;
	////

	////
	float *output_cpu = new float[OUTPUT_SIZE * 2];
	sT_C = clock();
	for (int ii = 0; ii < N * 2; ii++)
		mlp_cpu.forward(input, hiddenWeights, hiddenBias, outputWeights, outputBias, output_cpu);

	eT_C = clock();
	std::cout << "CPU=  ";
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		std::cout << output_cpu[i] << ' ';
	}
	std::cout << std::endl;
	std::cout << eT_C - sT_C << "ms" << std::endl;
	
	std::cout << "GPU加速比" <<  (eT_C - sT_C) *1.0f/(eT_G - sT_G) << std::endl;
	std::cout << "GPU-Stream加速比" << (eT_C - sT_C)*1.0f / (eT_GS - sT_GS) << std::endl;
	// 释放内存
	delete[] input;
	delete[] output; delete[] output_gs;
	delete[] output_cpu;
	return 0;
}
