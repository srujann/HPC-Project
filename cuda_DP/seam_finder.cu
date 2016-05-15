#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "gputimer.h"

#define N 4096
#define MAX_THREADS 1024
using namespace std;
GpuTimer timer;

__global__ void computeMinEnergy(float *energy, float *min_energy, int height, int width, int tempSize) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int pos = bid * blockDim.x + tid;
    
    extern __shared__ float shared_row_energy[];

    if(pos < width) {
        for(int i=tid; i<width; i += blockDim.x) {
            shared_row_energy[i] = energy[i];
            min_energy[i] = energy[i];
        }
    } else {
        return;
    }
    __syncthreads();
    
    float *temp = (float*) malloc(tempSize * sizeof(float));
    for(int i=1; i<height; i++) {
        int k = 0;
        for(int j=tid; j<width; j += blockDim.x) {
            float l = (j==0) ? 999999999 :  shared_row_energy[j-1];
            float m = shared_row_energy[j];
            float r = (j==width-1) ? 999999999: shared_row_energy[j+1];
            
            float minimum = energy[i*width + j] + min(l, min(m, r));
            temp[k++] = minimum;
        }
        __syncthreads();
        k = 0;
        for(int j=tid; j<width; j += blockDim.x) {
            shared_row_energy[j] = temp[k];
            min_energy[i * width + j] = temp[k++];
        }
    }
}

int main(int argc, char** argv)
{
    int height = N;
    int width = N;
    float *h_energy = (float*) malloc(N * N * sizeof(float));
    float *h_energy1 = (float*) malloc(N * N * sizeof(float));

    cout<<"Original Matrix"<<endl;
    for(int i=0; i<5; i++) {
        for(int j=0; j<width; j++) {
            h_energy[i * N + j] = (i * N + j) % 17;
            cout << h_energy[i * N + j] << "  ";
        }
        cout<<endl;
    }   
    
    int noOfBlocks = 1; 
    int noOfThreads = min(MAX_THREADS, width);
    int sharedSize = ((N-1) / noOfThreads + 1) * noOfThreads * sizeof(float);
    int tempSize = ((N-1) / noOfThreads + 1);
    cout<<"Blocks: "<<noOfBlocks<<"  Threads: "<<noOfThreads<<" SharedSize: "<<sharedSize<<endl;
    dim3 grid(noOfBlocks), block(noOfThreads);
    float *d_energy, *d_min_energy;
    cudaError_t rc;
    
    rc = cudaMalloc((void**) &d_energy, N * N * sizeof(float));
    if(rc != cudaSuccess) {
        cout<<"Malloc Failed for d_energy"<<endl;
    }
    rc = cudaMalloc((void**) &d_min_energy, N * N * sizeof(float));
    if(rc != cudaSuccess) {
        cout<<"Malloc Failed for d_min_energy"<<endl;
    }
    rc = cudaMemcpy(d_energy, h_energy, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if(rc != cudaSuccess) {
        cout<<"Memcpy failed from host to device"<<endl;
    }
    
    timer.Start();
    computeMinEnergy<<<grid, block, sharedSize>>>(d_energy, d_min_energy, height, width, tempSize);
    timer.Stop();
    
    rc = cudaMemcpy(h_energy1, d_min_energy, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if(rc != cudaSuccess) {
        cout<<"Memcpy failed from device to host"<<endl;
    }
    
    cout<<"Output Matrix"<<endl;
    for(int i=0; i<5; i++) {
        for(int j=0; j<width; j++) {
            cout << h_energy1[i * N + j] << "  ";
        }
        cout<<endl;
    }
    cudaFree(d_energy);
    cudaFree(d_min_energy);
    cout<<"Time: "<<timer.Elapsed()<<endl;
        
    return 0;
}