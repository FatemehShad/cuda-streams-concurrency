# CUDA Streams and Concurrency
According to NVIDIA, stream in cuda is a sequence of operations that execute in issue-order on the GPU. <br>
Those operations involve:
1. Copy data from host memory (CPU) to device memory (GPU) 
2. Kernel invocation
3. Transfer (copy) data back from device to the host.
To conduct concurrent execution of those three operations, we need use multiple streams. 

In the [cuda-performance-comparison](https://github.com/luckyp71/cuda-performance-comparison) repo, we've seen that the best result in conducting vector addition (with 1M elements of each) belongs to the multiple blocks and multiple threads approach (it took only **36.896us** to finish the execution). The three vector addition approaches in [cuda-performance-comparison](https://github.com/luckyp71/cuda-performance-comparison) used stream as well, but that stream is a default stream. This repo will show us how to use multiple streams (non-default stream) to achieve concurrent execution of CUDA operations mentioned above.  

## Prerequisites:
1. C++11
2. [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
3. NVIDIA GPU

## Summary
After tested vector addition using multiple streams with the same grid size (number of blocks) and block size (number of threads) as multiple blocks and multiple threads approach of the [cuda-performance-comparison](https://github.com/luckyp71/cuda-performance-comparison), it took only **5.4us** to finish the execution or it is about 6.8 times faster than using a default stream.

Note:
Your result might be different, it really depends on your device (GPU) specification, mine is NVIDIA GeForce RTX 2070 MAX-Q.