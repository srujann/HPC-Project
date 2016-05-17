#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "gputimer.h"

#define N 4096
#define MAX_THREADS 1024
using namespace std;
GpuTimer timer;

__global__ void computeMinEnergyMatrix(float *energy, float *min_energy, int height, int width) {
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
    
    float temp[4];
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
    float *h_min_energy = (float*) malloc(N * N * sizeof(float));

    cout<<"Original Matrix"<<endl;
    for(int i=0; i<5; i++) {
        for(int j=0; j<width; j++) {
            h_energy[i * N + j] = (i * N + j) > width/2 ? ((i * N + j) * 23)%17 : ((i * N + j) * 23)%17 + 2;
            cout << h_energy[i * N + j] << "  ";
        }
        cout<<endl;
    }   
    
    cout << "-------------------- Minimum Energy Matrix Calculation Starts --------------------" << endl;

    int noOfBlocks = 1; 
    int noOfThreads = min(MAX_THREADS, width);
    int sharedSize = ((width-1) / noOfThreads + 1) * noOfThreads * sizeof(float);
    
    cout<<"Blocks: "<<noOfBlocks<<"  Threads: "<<noOfThreads<<" SharedSize: "<<sharedSize<<endl;
    
    // Number to blocks will always stay at 1    
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
    computeMinEnergyMatrix<<<grid, block, sharedSize>>>(d_energy, d_min_energy, height, width);
    timer.Stop();
    
    rc = cudaMemcpy(h_min_energy, d_min_energy, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if(rc != cudaSuccess) {
        cout<<"Memcpy failed from device to host"<<endl;
    }
    
    cout<<"Output Matrix"<<endl;
    for(int i=0; i<5; i++) {
        for(int j=0; j<width; j++) {
            cout << h_min_energy[i * N + j] << "  ";
        }
        cout<<endl;
    }
    cudaFree(d_energy);
    cudaFree(d_min_energy);
    cout<<"Time: "<<timer.Elapsed()<<endl;
    fflush(stdout);
    
    cout << "---------------------- Minimum Value Calculation Starts ----------------------" << endl;
    
    timer.Start();
    float mn = h_min_energy[0];
    for(int i=1; i<width; i++) {
        if(h_min_energy[i] < mn) {
            mn = h_min_energy[i];
        }
    }
    timer.Stop();
    cout<<"Timer Elapsed: "<<timer.Elapsed()<<endl;
    cout<<"mn: "<<mn<<endl;
    
    
    return 0;
}