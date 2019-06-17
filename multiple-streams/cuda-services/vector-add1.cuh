#ifndef VECTOR_ADD1_H
#define VECTOR_ADD1_H

#include <cuda_runtime.h>

namespace vectorAdd1{
    __global__ void vectorAdd1(float *a, float *b, float *c, int offset);
    float* launchKernel();
}
#endif