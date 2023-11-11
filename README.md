# Cuda-MLP
homework of GPU course of USTC

## 优化
我们可以考虑每个批次两张图片，由于GPU中的流并行类似于CPU上的任务并行，即每个流都可以看作是一个独立的任务，每个流中的代码操作顺序执行，即可以设计两个流，如图
![image](https://github.com/Zero0Hero/Cuda-MLP/assets/64831522/25103971-4025-492e-ac39-68f51aab288f) 

这样就出现了流并行，即1-input执行完成后，2-input（第二个流从内存传入显存时）时，计算引擎已经具备执行流1的可能，这样2-input与1-matrix multi add可并行执行以掩盖2-input的数据传输时间；同理当1-matrix multi add执行完成后，可以执行1-output数据传输任务，同时2-matrix multi add可并行执行，对比如图
![image](https://github.com/Zero0Hero/Cuda-MLP/assets/64831522/d7012fc0-99c3-401c-b17a-8b7bbef993d5)


##可能的优化？
以上，我们设计了两个流的优化，但是流内仍有优化空间，我们将单个任务中的两次矩阵相乘拆开看，如图5，即当进行1-matrix multi add1后1-matrix multi add2执行时，完全可以使用1-matrix multi add1的硬件执行下一次1-matrix multi add1（3-matrix multi add1），即我们拥有的硬件只要空闲，就应有输入数据投喂进来。在Xilinx HLS（High-Level Synthesis）中，可以构造如下图所示的流水线（函数级Pipelining），但在本次作业内还为找到类似那样显式的流水线构造，后续将会继续探索
![image](https://github.com/Zero0Hero/Cuda-MLP/assets/64831522/06bf54de-ad13-4274-8908-0a4062b498ae)
