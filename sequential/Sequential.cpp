#include <iostream>
#include <stdio.h>
#include "FileConverterWrapper.h"
#include <string.h>

using namespace std;

int size = 0, height = 0, width = 0;
unsigned char* data;
float* energy;
float* leastEnergySum;
int* indexLeastEnergySumInRow;

int fileOperations(string, int, codeType);
void energyCalculation();
void energyPathDP();
int leastCumSumLastRow();
void leastEnergyPathBacktrack(int indexLeastEnergySumPath);
void removeRGBValues();

int main() {
	return fileOperations("TestImage3", 800, sequential);
}

int fileOperations(string fileName, int seamsToGenerate, codeType typeSeamCarving) {

	int debugPrint = 50;
	BMPtoRGB(fileName, typeSeamCarving);
	FILE* documentRead = fopen(fileName.c_str(), "rb");
	if (!documentRead) {
		cout << "The input document is not available" << endl;
		return 0;
	}
	string writeFileName;
	switch (typeSeamCarving) {
	case sequential:
		writeFileName = fileName + "_seq";
		break;
	case cuda_DP:
		writeFileName = fileName + "_cuDP";
		break;
	case cuda_Greedy:
		writeFileName = fileName + "_cuGR";
		break;
	}

	FILE* documentWrite = fopen(writeFileName.c_str(), "wb");
	fread(&height, sizeof(int), 1, documentRead);
	fread(&width, sizeof(int), 1, documentRead);
	cout << endl << "The input image file is loaded for seam carving." << endl;
	cout << "Initial width: " << width << endl << "Initial Height: " << height << endl;

	//Transferring the RGB values to a 1D matrix
	size = 3 * width * height;
	data = new unsigned char[size];
	fread(data, sizeof(unsigned char), size, documentRead);
	fclose(documentRead);

	energy = new float[size / 3];
	leastEnergySum = new float[size / 3];
	indexLeastEnergySumInRow = new int[height];

	//To generate multiple seams using energyCalculation, energyPathDP, leastCumSumLastRow, leastEnergyPathBacktrack, removeRGBValues.
	for (int i = 0; i < seamsToGenerate; i++) {
		energyCalculation();
		if (i % debugPrint == 0)
			cout << "Energy calculation is completed till iteration: " << i << endl;
		energyPathDP();
		if (i % debugPrint == 0)
			cout << "Energy path using DP is completed till iteration: " << i << endl;
		int indexLeastEnergySumPath = leastCumSumLastRow();
		if (i % debugPrint == 0)
			cout << "Least energy at last row calculated till interation : " << i << endl;
		leastEnergyPathBacktrack(indexLeastEnergySumPath);
		if (i % debugPrint == 0)
			cout << "Computed the indexes of lowest energy path till interation: " << i << endl;
		removeRGBValues();
		if (i % debugPrint == 0)
			cout << "Removed the corresponding RGB values of the seam till iteration: " << i << endl;
	}

	cout << endl << "New Width:" << width << endl << "New Height:" << height;
	/*	cout << "Final data values" << endl;
	 for (int p = 0; p < size; p++) {
	 cout << (int) data[p] << endl;
	 }*/

	fwrite(&height, sizeof(int), 1, documentWrite);
	fwrite(&width, sizeof(int), 1, documentWrite);
	fwrite(data, sizeof(char), size, documentWrite);
	RGBtoBMP(writeFileName, typeSeamCarving);
	free(energy);
	free(leastEnergySum);
	free(indexLeastEnergySumInRow);
	return 0;
}

void energyCalculation() {
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
	/*cout<<"Energy Values"<<endl;
	 for (int i=0;i<size/3;i++){
	 cout<<energy[i]<<endl;
	 }*/

}

void energyPathDP() {
	// Least energy path using Dynamic Programming
	for (int i = 0; i < size / 3; i++) {
		if (i >= 0 && i < width) {
			leastEnergySum[i] = energy[i];
		} else {
			if (i % width == 0) {
				leastEnergySum[i] = energy[i] + min(leastEnergySum[i - width], leastEnergySum[i - width + 1]);
			} else if (i % width == width - 1) {
				leastEnergySum[i] = energy[i] + min(leastEnergySum[i - width - 1], leastEnergySum[i - width]);
			} else {
				leastEnergySum[i] = energy[i] + min(min(leastEnergySum[i - width - 1], leastEnergySum[i - width]), leastEnergySum[i - width + 1]);
			}
		}
	}
	/*cout<<endl<<"Least energy sum path"<<endl;
	 for(int i=0;i<width*height;i++){
	 cout<<leastEnergySum[i]<<endl;
	 }*/

}

int leastCumSumLastRow() {
	//Least cumulative sum in the last row.
	int indexLeastEnergySumPath;
	float leastEnergySumPathValue = INT_MAX;
	for (int i = width * (height - 1); i < height * width; i++) {
		if (leastEnergySumPathValue >= leastEnergySum[i]) {
			leastEnergySumPathValue = leastEnergySum[i];
			indexLeastEnergySumPath = i;
		}
	}
	//cout << endl << "Least Sum Path Value & Index: " << leastEnergySumPathValue << endl << indexLeastEnergySumPath;
	return indexLeastEnergySumPath;

}

void leastEnergyPathBacktrack(int indexLeastEnergySumPath) {
	//Calculating the indexes of the Least energy path by backtracking
	int tempCountHeight = height - 1;
	int tempIndexLeastEnergySumPath = indexLeastEnergySumPath;
	indexLeastEnergySumInRow[tempCountHeight] = indexLeastEnergySumPath;
	while (tempCountHeight > 0) {
		int left = tempIndexLeastEnergySumPath - width - 1;
		int middle = tempIndexLeastEnergySumPath - width;
		int right = tempIndexLeastEnergySumPath - width + 1;
		int leftLimit = (tempCountHeight - 1) * width;
		int rightLimit = tempCountHeight * width;
		if (leftLimit <= left && right < rightLimit) {
			if (leastEnergySum[left] <= leastEnergySum[middle] && leastEnergySum[left] <= leastEnergySum[right])
				tempIndexLeastEnergySumPath = left;
			else if (leastEnergySum[middle] <= leastEnergySum[left] && leastEnergySum[middle] <= leastEnergySum[right])
				tempIndexLeastEnergySumPath = middle;
			else if (leastEnergySum[right] <= leastEnergySum[middle] && leastEnergySum[right] <= leastEnergySum[left])
				tempIndexLeastEnergySumPath = right;
		} else if (leftLimit > left) {
			tempIndexLeastEnergySumPath = leastEnergySum[middle] <= leastEnergySum[right] ? middle : right;
		} else {
			tempIndexLeastEnergySumPath = leastEnergySum[left] <= leastEnergySum[middle] ? left : middle;
		}
		tempCountHeight--;
		indexLeastEnergySumInRow[tempCountHeight] = tempIndexLeastEnergySumPath;
	}
	/*cout<<endl<<"Indexes of the least sum in each row:"<<endl;
	 for(int i=0;i<height;i++){
	 cout<<indexLeastEnergySumInRow[i]<<endl;
	 }*/

}

void removeRGBValues() {
	//Removing the RGB values of the indexes calculated in the previous step from the main matrix
	//		for (int i = 0; i < height; i++) {
	//			data[indexLeastEnergySumInRow[i] * 3] = 255;
	//			data[indexLeastEnergySumInRow[i] * 3 + 1] = 0;
	//			data[indexLeastEnergySumInRow[i] * 3 + 2] = 0;
	//		}
	int countRGBDeleted = 0;
	for (int i = 0; i < height - 1; i++) {
		for (int j = indexLeastEnergySumInRow[i]; j < indexLeastEnergySumInRow[i + 1]; j++) {
			data[(j * 3) - (countRGBDeleted * 3)] = data[(j * 3) + 3];
			data[(j * 3) - (countRGBDeleted * 3) + 1] = data[(j * 3) + 3 + 1];
			data[(j * 3) - (countRGBDeleted * 3) + 2] = data[(j * 3) + 3 + 2];
		}
		countRGBDeleted++;
	}
	for (int j = indexLeastEnergySumInRow[height - 1]; j < height * width; j++) {
		data[(j * 3) - (countRGBDeleted * 3)] = data[(j * 3) + 3];
		data[(j * 3) - (countRGBDeleted * 3) + 1] = data[(j * 3) + 3 + 1];
		data[(j * 3) - (countRGBDeleted * 3) + 2] = data[(j * 3) + 3 + 2];
	}
	width = width - 1;
	size = 3 * width * height;
}
