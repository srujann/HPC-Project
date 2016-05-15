#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "gputimer.h"

#define N 4096
#define MAX_THREADS 1024
using namespace std;
GpuTimer timer;

__global__ void computeMinEnergyValue(float *last_row, int *min_idx, int width) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int pos = 2 * bid * blockDim.x + tid;

    __shared__ float shared_last_row[MAX_THREADS];
    __shared__ int shared_row_idx[MAX_THREADS];
    
    if((pos + blockDim.x) < width) {
        if(last_row[pos] < last_row[pos + blockDim.x]) {
            shared_last_row[tid] = last_row[pos];
            shared_row_idx[tid] = pos;
        } else {
            shared_last_row[tid] = last_row[pos + blockDim.x];
            shared_row_idx[tid] = pos + blockDim.x;
        }
    } else {
        return;
    }
    
    __syncthreads();
    
    for(int i = blockDim.x/2; i > 0; i = i>>1) {
        if(tid < i) {
            if(shared_last_row[tid] > shared_last_row[tid + i]) {
                shared_last_row[tid] = shared_last_row[tid + i];
                shared_row_idx[tid] = shared_row_idx[tid+i];
            }
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        min_idx[bid] = shared_last_row[0];
        //printf("Min: %f\n", shared_last_row[0]);
    }
    
}

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

unsigned int nextPow2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
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
    
    int noOfBlocks = 1; 
    int noOfThreads = min(MAX_THREADS, width);
    int sharedSize = ((N-1) / noOfThreads + 1) * noOfThreads * sizeof(float);
    
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
    
    noOfThreads = (width < MAX_THREADS*2) ? nextPow2((width + 1)/ 2) : MAX_THREADS;
    noOfBlocks = (width + (noOfThreads * 2 - 1)) / (noOfThreads * 2);    
    sharedSize = noOfBlocks * sizeof(float);
    
    cout<<"Blocks: "<<noOfBlocks<<"  Threads: "<<noOfThreads<<" SharedSize: "<<sharedSize<<endl;

    dim3 grid_find_min(noOfBlocks), block_find_min(noOfThreads);
    float *d_last_row;
    int *d_min_idx;
    int *h_min_idx = (int*) malloc(noOfBlocks * sizeof(int));
    
    rc = cudaMalloc((void**) &d_last_row, width * sizeof(float));
    if(rc != cudaSuccess) {
        cout<<"Malloc failed for last_row"<<endl;
    }
    
    rc = cudaMalloc((void**) &d_min_idx, noOfBlocks * sizeof(int));
    if(rc != cudaSuccess) {
        cout<<"Malloc failed for min_idx"<<endl;
    }
    
    timer.Start();
    // TODO: Modify for last row --> ((height-1) * width)
    rc = cudaMemcpy(d_last_row, h_min_energy , width * sizeof(float), cudaMemcpyHostToDevice);
    if(rc != cudaSuccess) {
        cout<<"Memcpy failed from host to device for min_energy -> last_row"<<endl;
    }
    
    
    computeMinEnergyValue<<<grid_find_min, block_find_min>>>(d_last_row, d_min_idx, width);
    
    
    rc = cudaMemcpy(h_min_idx, d_min_idx , noOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    if(rc != cudaSuccess) {
        cout<<"Memcpy failed from device to host for min_idx -> min_idx "<<endl;
    }
    timer.Stop();
    cout<<"Timer Elapsed: "<<timer.Elapsed()<<endl;
    cout<<h_min_idx[0]<<"  "<<h_min_idx[1] <<endl;
    
    
    return 0;
}