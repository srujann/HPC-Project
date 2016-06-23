#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "gputimer.h"

#define MAX_THREADS 1024
#define MAX_WIDTH 2048
using namespace std;
GpuTimer timer;

void energyCalculation(float* energy, unsigned char* data, int size, int height, int width) {
	/*Energy Calculation*/
	float sumX = 0, sumY = 0;
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

void sequentialEnergyCompute() {
	int size = 0, height = 0, width = 0;
	FILE* documentRead = fopen("TestImage3", "rb");
	if (!documentRead) {
		cout << "The input document is not available" << endl;
	}

	fread(&height, sizeof(int), 1, documentRead);
	fread(&width, sizeof(int), 1, documentRead);
	size = 3 * height * width;

	unsigned char* data = new unsigned char[size];
	fread(data, sizeof(unsigned char), size, documentRead);
	fclose(documentRead);

	float* energy = new float[size / 3];
	timer.Start();
	energyCalculation(energy, data, size, height, width);
	timer.Stop();

	cout << "Time taken for serial energy computation to execute:" << timer.Elapsed() << endl;

	FILE* docWrite = fopen("testEnergyMatrix", "wb");
	fwrite(&height, sizeof(int), 1, docWrite);
	fwrite(&width, sizeof(int), 1, docWrite);
	fwrite(energy, sizeof(float), size / 3, docWrite);
	fclose(docWrite);
}

__global__ void energyMtxCompute(const unsigned char* imgData, float* energyMtx, const int height, const int width) {

	extern __shared__ unsigned char curr_imgData[];

	int tid = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tid;
	int y = blockIdx.y;
	int currIndex = 0;
	
	int blockWidth = (width < (blockIdx.x + 1) * blockDim.x) ? (width - (blockIdx.x * blockDim.x)) : blockDim.x;

	if(tid >= blockWidth)
    	return;

	int boundX = (y * width) * 3 + blockDim.x * blockIdx.x * 3;
	for (int j = tid; j < 3 * blockWidth; j = j + blockWidth) {
		curr_imgData[j] = imgData[boundX + j];
	}
	__syncthreads();

	//compute the left, right, up and down indices for current pixel
	int left = 0, right = 0, up = 0, down = 0;

	if (tid == 0) {
		//fetch left pixel for block's left boundary
		if (x == 0) {
			currIndex = (y * width + width - 1) * 3;
		} else {
			currIndex = (y * width + x - 1) * 3;
		}

		for (int i = 0; i < 3; i++) {
			curr_imgData[3 * blockWidth + i] = imgData[currIndex + i];
		}
		left = 3 * blockWidth;
		right = (tid + 1) * 3;

	} else if (tid == blockWidth - 1) {
		//fetch right pixel for block's right boundary
		if (x == width - 1) {
			currIndex = (y * width) * 3;
		} else {
			currIndex = (y * width + x + 1) * 3;
		}

		for (int i = 0; i < 3; i++) {
			curr_imgData[3 * (blockWidth + 1) + i] = imgData[currIndex + i];
		}

		left = (tid - 1) * 3;
		right = (blockWidth + 1) * 3;

	} else {
		right = (tid + 1) * 3;
		left = (tid - 1) * 3;
	}

	up = (y == 0) ? (((height - 1) * width + x) * 3) : (((y - 1) * width + x) * 3);
	down = (((y + 1) % height) * width + x) * 3;

	float energyVal = 0, dx = 0, dy = 0;
	//compute the energy value for the pixel
	for (int i = 0; i < 3; i++) {
		dx = curr_imgData[right + i] - curr_imgData[left + i];
		dy = imgData[up + i] - imgData[down + i];
		energyVal += dx * dx + dy * dy;
	}

	//store the computed energy value to global energy matrix
	energyMtx[y * width + x] = energyVal;
}

int main(int argc, char** argv) {
	sequentialEnergyCompute();
	int height = 0, width = 0, energyMtxSize = 0;
	float *testEnergy;
	string inImageFileName = "TestImage3";
	FILE* docRead = fopen("testEnergyMatrix", "rb");
	if (!docRead) {
		cout << "testEnergyMatrix file is not available" << endl;
		return 0;
	}

	fread(&height, sizeof(int), 1, docRead);
	fread(&width, sizeof(int), 1, docRead);
	energyMtxSize = height * width;

	testEnergy = (float*) malloc(energyMtxSize * sizeof(int));
	fread(testEnergy, sizeof(float), energyMtxSize, docRead);
	fclose(docRead);

	FILE* documentRead = fopen(inImageFileName.c_str(), "rb");
	if (!documentRead) {
		cout << "The input image file is not available" << endl;
		return 0;
	}

	cout << "Input Image read from :" << inImageFileName << endl;
	cout << "Height of the image   :" << height << endl;
	cout << "Width of the image    :" << width << endl;

	fread(&height, sizeof(int), 1, documentRead);
	fread(&width, sizeof(int), 1, documentRead);
	int imgDataSize = 3 * height * width;
	unsigned char* h_imgData = (unsigned char *) malloc(imgDataSize);
	fread(h_imgData, sizeof(unsigned char), imgDataSize, documentRead);
	fclose(documentRead);

	cudaError_t rc;
	float *d_energyMtx, *h_energyMtx;
	unsigned char* d_imgData;

	h_energyMtx = (float *) malloc(energyMtxSize * sizeof(float));

	rc = cudaMalloc((void**) &d_energyMtx, energyMtxSize * sizeof(float));
	if (rc != cudaSuccess) {
		cout << "Malloc Failed for d_energyMtx" << endl;
	}

	rc = cudaMalloc((void**) &d_imgData, imgDataSize);
	if (rc != cudaSuccess) {
		cout << "Malloc Failed for d_imgData" << endl;
	}

	rc = cudaMemcpy(d_imgData, h_imgData, imgDataSize, cudaMemcpyHostToDevice);
	if (rc != cudaSuccess) {
		cout << "Memcpy failed from host to device" << endl;
	}

	int numThreads = min(MAX_THREADS, width);
	int gridCols = height;
	int sharedSize = (numThreads + 2) * 3;

	int gridRows = (width % numThreads == 0) ? (width / numThreads) : (width / numThreads + 1);

	dim3 grid_Dim(gridRows, gridCols, 1);
	dim3 block_Dim(numThreads, 1, 1);

	timer.Start();
	energyMtxCompute<<<grid_Dim, block_Dim, sharedSize>>>(d_imgData, d_energyMtx, height, width);
	timer.Stop();

	rc = cudaMemcpy(h_energyMtx, d_energyMtx, energyMtxSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (rc != cudaSuccess) {
		cout << "Memcpy failed from device to host with rc:" << rc << endl;
	}

	cout << "Time taken for parallel energy computation to execute:" << timer.Elapsed() << endl;
	int i = 0;
	for (i = 0; i < height * width; i++) {
		if (testEnergy[i]-h_energyMtx[i] > 0.00001) {
			cout << "error at index i:" << i << "|expected:" << testEnergy[i] << "|actual:" << h_energyMtx[i] << endl;
			break;
		}
	}

	if (i == height * width) {
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