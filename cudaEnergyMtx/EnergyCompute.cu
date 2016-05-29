#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "gputimer.h"

#define MAX_THREADS 1024
#define MAX_WIDTH 2048
using namespace std;
GpuTimer timer;

void energyCalculation(int* energy, unsigned char* data, int size, int height, int width) {
	/*Energy Calculation*/
	int sumX = 0, sumY = 0;
	for (int i = 0; i < size / 3; i++) {
		sumX = 0;
		sumY = 0;
		for (int j = 0; j < 3; j++) {
			if (i % width == 0) {
				sumX = sumX + ((data[(i + 1) * 3 + j] - data[(i + width - 1) * 3 + j]) * (data[(i + 1) * 3 + j] - data[(i + width - 1) * 3 + j]));
			}
			if (i % width == width - 1) {
				sumX = sumX + ((data[(i - width + 1) * 3 + j] - data[(i - 1) * 3 + j]) * (data[(i - width + 1) * 3 + j] - data[(i - 1) * 3 + j]));
			}
			if (i % width != 0 && i % width != width - 1) {
				sumX = sumX + ((data[(i + 1) * 3 + j] - data[(i - 1) * 3 + j]) * (data[(i + 1) * 3 + j] - data[(i - 1) * 3 + j]));
			}
			if (i >= 0 && i < width) {
				sumY = sumY + ((data[(i + (height - 1) * width) * 3 + j] - data[(i + width) * 3 + j]) * (data[(i + (height - 1) * width) * 3 + j] - data[(i + width) * 3 + j]));
			}
			if (i >= (height - 1) * width && i < (height * width)) {
				sumY = sumY + ((data[(i - (height - 1) * width) * 3 + j] - data[(i - width) * 3 + j]) * (data[(i - (height - 1) * width) * 3 + j] - data[(i - width) * 3 + j]));
			}
			if (i >= width && i < (height - 1) * width) {
				sumY = sumY + ((data[(i + width) * 3 + j] - data[(i - width) * 3 + j]) * (data[(i + width) * 3 + j] - data[(i - width) * 3 + j]));
			}
		}
		energy[i] = sumX + sumY;
	}
}

void testMatrixGenerate() {
	int size = 0, height = 0, width = 0;
    FILE* documentRead = fopen("TestImage3", "rb");
    if (!documentRead) {
		cout << "The input document is not available" << endl;
	}
	
	fread(&height, sizeof(int), 1, documentRead);
	fread(&width, sizeof(int), 1, documentRead);
	size = 3 * height * width;
	
	//cout << "Size:" << size << endl; 
	unsigned char* data = new unsigned char[size]; 
	fread(data, sizeof(unsigned char), size, documentRead);
	fclose(documentRead);
	
	int* energy = new int[size/3];
	timer.Start();
	energyCalculation(energy, data, size, height, width);
	timer.Stop();
	
	cout << "Time taken for serial energy computation to execute:" << timer.Elapsed() << endl;
	
	/*for(int i = 0; i < 9; i++) {
		cout << data[i] << " ";
	}
	cout <<  endl;*/
	
	FILE* docWrite = fopen("testEnergyMatrix", "wb");
	fwrite(&height, sizeof(int), 1, docWrite);
	fwrite(&width, sizeof(int), 1, docWrite);
	fwrite(energy, sizeof(int), size/3, docWrite);
	fclose(docWrite);
	
	/*for(int i = 0; i < 10; i++) {
		cout << energy[i] << " ";
	}
	cout <<  endl;*/
}

__global__ void energyMtxCompute(const unsigned char* imgData, int* energyMtx,
		const int height, const int width) {

	__shared__
	unsigned char shared_imgData[3][MAX_WIDTH * 3];
	//__shared__ int shared_energyMtx[MAX_WIDTH];

	int x = threadIdx.x;
	int currIndex = 0, row = 0;

	//load first row into shared memory
	for(int i = x; i < 3*width; i = i+1024) {
		shared_imgData[0][i] = imgData[i];
	}

	//load the last row shared memory
	row = (height-1)*width*3;
	for(int i = x; i < 3*width; i = i +1024) {
		currIndex = row + i;
		shared_imgData[2][i] = imgData[currIndex];
	}

	int hCurrent = 0, hMove = 0, hShared = 0;
	int dx = 0, dy = 0, energyVal = 0, left = 0, right = 0, up = 0, down = 0;

	for (int y = 0; y < height; y++) {

		hCurrent = y % 3;
		hMove = (y + 1) % height;
		hShared = (y + 1) % 3;

		//move next row into shared memory
		row = hMove * width * 3;
		for(int i = x; i < 3*width; i = i+1024) {
			currIndex = row + i; 
			shared_imgData[hShared][i] = imgData[currIndex];
		}
		__syncthreads();

		for (int i = x; i < width; i = i + 1024) {
			energyVal = 0;
			if (i == 0) {
				left = width - 1;
			} else {
				left = i - 1;
			}

			if (i == width - 1) {
				right = 0;
			} else {
				right = i + 1;
			}

			if (y == 0) {
				up = 2;
			} else {
				up = (y - 1) % 3;
			}

			down = (y + 1) % 3;

			for (int j = 0; j < 3; j++) {
				dx = shared_imgData[hCurrent][right * 3 + j] - shared_imgData[hCurrent][left * 3 + j];
				dy = shared_imgData[down][i * 3 + j] - shared_imgData[up][i * 3 + j];
				energyVal = energyVal + dx * dx + dy * dy;
			}
			energyMtx[y * width + i] = energyVal;
		}
	}
}

int main(int argc, char** argv)
{
	testMatrixGenerate();
	int height = 0, width = 0, energyMtxSize = 0;
	int *testEnergy;
	FILE* docRead = fopen("testEnergyMatrix", "rb");
	if (!docRead) {
		cout << "testEnergyMatrix file is not available" << endl;
	}
	
	fread(&height, sizeof(int), 1, docRead);
	fread(&width, sizeof(int), 1, docRead);
	energyMtxSize = height * width;
	
	//cout << "Height:" << height << endl;
	//cout << "Width:" << width << endl;
	//cout << "Size:" << energyMtxSize << endl;
	
	testEnergy = (int*)malloc(energyMtxSize * sizeof(int));
	fread(testEnergy, sizeof(int), energyMtxSize, docRead);
	fclose(docRead);
	
	FILE* documentRead = fopen("TestImage3", "rb");
    if (!documentRead) {
		cout << "The input image data document is not available" << endl;
	}
	
	fread(&height, sizeof(int), 1, documentRead);
	fread(&width, sizeof(int), 1, documentRead);
	int imgDataSize = 3 * height * width;
	unsigned char* h_imgData = (unsigned char *)malloc(imgDataSize); 
	fread(h_imgData, sizeof(unsigned char), imgDataSize, documentRead);
	fclose(documentRead);
	
	cudaError_t rc;
    int *d_energyMtx, *h_energyMtx;
    unsigned char* d_imgData;
    
    h_energyMtx = (int *)malloc(energyMtxSize * sizeof(int));
    
    rc = cudaMalloc((void**) &d_energyMtx, energyMtxSize * sizeof(int));
    if(rc != cudaSuccess) {
        cout<<"Malloc Failed for d_energyMtx"<<endl;
    }
    
    rc = cudaMalloc((void**) &d_imgData,  imgDataSize);
    if(rc != cudaSuccess) {
        cout<<"Malloc Failed for d_imgData"<<endl;
    }
    
    rc = cudaMemcpy(d_imgData, h_imgData, imgDataSize, cudaMemcpyHostToDevice);
    if(rc != cudaSuccess) {
        cout<<"Memcpy failed from host to device"<<endl;
    }
	
	dim3 gridDim(1, 1, 1);
	dim3 blockDim(1024, 1, 1);
	
    timer.Start();
    energyMtxCompute<<<gridDim, blockDim>>>(d_imgData, d_energyMtx, height, width);
	timer.Stop();
    
    rc = cudaMemcpy(h_energyMtx, d_energyMtx, energyMtxSize * sizeof(int), cudaMemcpyDeviceToHost);
    if(rc != cudaSuccess) {
        cout<<"Memcpy failed from device to host with rc:"<< rc << endl;
    }   
	
	cout << "Time taken for parallel energy computation to execute:" << timer.Elapsed() << endl;
	int  i = 0;
	for(i = 0; i < height*width; i++) {
		if(testEnergy[i] != h_energyMtx[i]) {
			cout << "error at index i:" << i << "|expected:" << testEnergy[i] << "|actual:" << h_energyMtx[i] << endl;  
		}
	}
	
	if(i == height*width) {
		cout << "Success" << endl;
	} else {
		cout << "Failure" << endl;
	}
	
	cudaFree(d_energyMtx);
	cudaFree(d_imgData);
	free(h_energyMtx);
	free(h_imgData);
	free(testEnergy);

    return 0;
}