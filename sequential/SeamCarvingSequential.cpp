#include <iostream>
#include <stdio.h>
using namespace std;

int main() {
	FILE* image = fopen("splashw_mine.bmp", "rb");
	FILE* imageWrite = fopen("seamedImageNew5.bmp", "wb");
	if (image) {
		unsigned char header[54];
		fread(header, sizeof(unsigned char), 54, image);
		int width = *(int*) &header[18];
		int height = *(int*) &header[22];
		cout << "Width: " << width << endl << "Height: " << height << endl;
		//Transferring the RGB values to a 1D matrix
		int size = 3 * width * height;
		unsigned char* data = new unsigned char[size];
		fread(data, sizeof(unsigned char), size, image);
		fclose(image);
		for (int i = 0; i < size; i += 3) {
			unsigned char tmp = data[i];
			data[i] = data[i + 2];
			data[i + 2] = tmp;
		}
		/*cout<<"RGB Values"<<endl;
		 for (int i=0;i<size;i++){
		 cout<<(int) data[i]<<endl;
		 }*/
		/*Energy Calculation*/
		float* energy = new float[size / 3];
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

		// Least energy path using Dynamic Programming
		float* leastEnergySum = new float[size / 3];
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
		//Least cumulative sum in the last row.
		int indexLeastEnergySumPath;
		float leastEnergySumPathValue = INT_MAX;
		for (int i = width * (height - 1); i < height * width; i++) {
			if (leastEnergySumPathValue >= leastEnergySum[i]) {
				leastEnergySumPathValue = leastEnergySum[i];
				indexLeastEnergySumPath = i;
			}
		}
		cout << endl << "Least Sum Path Value & Index: " << leastEnergySumPathValue << endl << indexLeastEnergySumPath;

	} else {
		cout << "Image not available" << endl;
	}
	return 0;
}
