#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "gputimer.h"

#define N 6
#define imgHeight 6
#define MAX_THREADS 1024
#define fMax 999999
using namespace std;
GpuTimer timer;

struct minPixel{
	float energy;
	int column;
};

__global__ void computeMinEnergyMatrix(float *energy, float *min_energy,int height,int width) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int pos = bid * blockDim.x + tid;
    
    extern __shared__ minPixel shared_minSet[];
    __shared__ int shared_minX;
    __shared__ int shared_minY;
    __shared__ int shared_minEnergy;
    
    float columnEnergy[imgHeight];
    float minPathEnergy[imgHeight];
    bool activeNodes[imgHeight];
    int minCol = 0;
    int minEnergy = fMax;
    
    for(int i = 0; i < height; i++) {
    	columnEnergy[i] = energy[width*i + pos];
    	minPathEnergy[i] = fMax;
    	activeNodes[i] = false;
    }
    
    minPathEnergy[0] = columnEnergy[0];
    activeNodes[0] = true;
    shared_minSet[pos].energy = columnEnergy[0];
    shared_minSet[pos].column = 0;
    __syncthreads();
    
    for(int i = 0; i < (height*width)-1; i++) {
    	if(tid == 0) {
    		int tMinX = 0;
    		int tMinEnergy = shared_minSet[tMinX].energy;
    		for(int j = 1; j < width; j++) {
    			if(shared_minSet[j].energy < tMinEnergy) {
    				tMinEnergy = shared_minSet[j].energy;
    				tMinX = j;
    			}
    		}
    		shared_minX = tMinX;
    		shared_minY = shared_minSet[tMinX].column; 	
    		shared_minEnergy = tMinEnergy;
    		//printf("minX:%d, minY:%d \n", shared_minX, shared_minY);
    	}
    	 __syncthreads();
    	 
    	 if(shared_minY == height-1) {
    	    break;
    	 }
    	 
    	 if(shared_minX == pos) {
    	    activeNodes[minCol] = false;
    	 }
    	 
    	 if(pos == shared_minX-1 || pos == shared_minX || pos == shared_minX+1) {
    	 	if(minPathEnergy[shared_minY+1] > shared_minEnergy+columnEnergy[shared_minY+1]) {
    	 		minPathEnergy[shared_minY+1] = shared_minEnergy+columnEnergy[shared_minY+1];
    	 		activeNodes[shared_minY+1] = true;
    	 	}
    	 }
    	 
    	 minEnergy = fMax;
    	 for(int j = 0; j < height; j++) {
    	    if(activeNodes[j] && minPathEnergy[j] < minEnergy) {
    	 		minCol = j;
    	 		minEnergy = minPathEnergy[j];
    	 	}
    	 }
    	 shared_minSet[pos].energy = minEnergy;
    	 shared_minSet[pos].column = minCol;
    	 __syncthreads();
    }
    
    for(int i = 0; i < height; i++) {
    	min_energy[width*i+pos] = minPathEnergy[i];
    }
    __syncthreads();
}

int main(int argc, char** argv)
{
    int height = N;
    int width = N;
    float *h_energy = (float*) malloc(N * N * sizeof(float));
    float *h_min_energy = (float*) malloc(N * N * sizeof(float));

    cout<<"Original Matrix"<<endl;
    for(int i= 0; i<height; i++) {
        for(int j=0; j<width; j++) {
            h_energy[i * N + j] = (i * N + j) > width/2 ? ((i * N + j) * 23)%17 : ((i * N + j) * 23)%17 + 2;
            //cout << h_energy[i * N + j] << "  ";
        }
        //cout<<endl;
    }   
    
    cout << "-------------------- Minimum Energy Matrix Calculation Starts --------------------" << endl;

    int noOfBlocks = 1; 
    int noOfThreads = min(MAX_THREADS, width);
    int sharedSize = width * sizeof(minPixel);
    
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
    for(int i=height-1; i< height; i++) {
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