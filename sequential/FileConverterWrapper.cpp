//============================================================================
// Name        : HPCSeamCarving.cpp
// Author      : Naren
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "FileConverterWrapper.h"

void RGBtoBMP(string fileName, codeType ct) {
	int width = 0, height = 0;
	long int size = 0;

	FILE *fin = fopen(fileName.c_str(), "rb");

	fread(&height, sizeof(int), 1, fin);
	fread(&width, sizeof(int), 1, fin);
	size = 3 * width * height;

//	cout << "Converting RGB Matrix to BMP Image" << endl;
//	cout << "File Read: " << fileName << endl;
//	cout << "Height: " << height << "  Width: " << width << endl;
//	cout << "Size of Buffer: " << size << endl;

	BYTE *rgbBuffer = new BYTE[size];
	fread(rgbBuffer, sizeof(char), size, fin);

	BYTE *newBitmap = ConvertRGBToBMPBuffer(rgbBuffer, width, height, &size);
	switch (ct) {
	case sequential:
		SaveBMP(newBitmap, width, height, size, (fileName + "_seq.bmp").c_str());
		break;
	case cuda_DP:
		SaveBMP(newBitmap, width, height, size, (fileName + "_cuDP.bmp").c_str());
		break;
	case cuda_Greedy:
		SaveBMP(newBitmap, width, height, size, (fileName + "_cuGR.bmp").c_str());
		break;
	}

	fclose(fin);
//	cout << "****Done****" << endl;
}

void BMPtoRGB(string fileName, codeType ct) {
	int width = 0, height = 0;
	long int size = 0;

	FILE *fout = fopen(fileName.c_str(), "wb");
	BYTE *bitmap = LoadBMP(&width, &height, &size, (fileName + ".bmp").c_str());
	BYTE *rgbBuffer = ConvertBMPToRGBBuffer(bitmap, width, height);

//	cout << "Converting BMP Image to RGB Matrix" << endl;
//	cout << "File Read: " << fileName << ".bmp" << endl;
//	cout << "Height: " << height << "  Width: " << width << endl;
//	cout << "Size of Buffer: " << size << endl;

	fwrite(&height, sizeof(int), 1, fout);
	fwrite(&width, sizeof(int), 1, fout);
	fwrite(rgbBuffer, sizeof(char), size, fout);

	fclose(fout);
//	cout << "****Done****" << endl;
}
