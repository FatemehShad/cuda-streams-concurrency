#include <stdio.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

namespace vectorAdd1{
__global__ void vectorAdd1(float *a, float *b, float *c, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  c[i] = a[i] + b[i];
}

float maxError(float *a, int n) 
{
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

float* launchKernel()
{
  const int blockSize = 256;
  const int nStreams = 8;
  const int n = 1 << 20; // 1M elements
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);

  float *a, *b, *c, *d_a, *d_b, *d_c;

  //ALlocate pinned host memory
  checkCuda( cudaMallocHost((void**)&a, bytes) );      
  checkCuda( cudaMallocHost((void**)&b, bytes) );      
  checkCuda( cudaMallocHost((void**)&c, bytes) );      
  
  //Allocate device memory
  checkCuda( cudaMalloc((void**)&d_a, bytes) ); 
  checkCuda( cudaMalloc((void**)&d_b, bytes) ); 
  checkCuda( cudaMalloc((void**)&d_c, bytes) );

  // float ms; // elapsed time in milliseconds
  
  //Setup CUDA Stream
  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; ++i){
    checkCuda( cudaStreamCreate(&stream[i]) );
  }

  // Initialize host arrays
  for(int i=0; i < n; i++){
    a[i] =10*i;
    b[i]=40*i;
  }

  /*
    Async: loop over copy from host to device, kernel invocation, and transfer data back from device to the host
  */
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );

    checkCuda( cudaMemcpyAsync(&d_b[offset], &a[offset], 
                              streamBytes, cudaMemcpyHostToDevice, 
                              stream[i]) );
  }

  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    vectorAdd1<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, d_b, d_c, offset);
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&c[offset], &d_c[offset], 
      streamBytes, cudaMemcpyDeviceToHost,
      stream[i]) );
  }

  printf("Max error: %e\n", maxError(c, n));

  //Free up memory
  for (int i = 0; i < nStreams; ++i){
    checkCuda( cudaStreamDestroy(stream[i]));
  }
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(a);
  cudaFreeHost(b);

  return c;
  }
}