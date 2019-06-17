#include <iostream>
#include "./cuda-services/vector-add1.cuh"

int main(){ 
    auto result = vectorAdd1::launchKernel();

      const int n = 1 << 20; // 1M elements

    for(int i=0; i < n; i++){
        std::cout<<result[i]<<std::endl;
    }


    return 0;
}