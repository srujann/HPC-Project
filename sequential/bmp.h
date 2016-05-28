/*
 * bmp.h
 *
 *  Created on: 15-May-2016
 *      Author: Karan Mehta
 */

#ifndef BMP_H_
#define BMP_H_

#include <windows.h>
#include <stdio.h>       // for memset
#include <iostream>

BYTE* ConvertRGBToBMPBuffer(BYTE* Buffer, int width, int height, long* newsize);
BYTE* ConvertBMPToRGBBuffer(BYTE* Buffer, int width, int height);
bool SaveBMP(BYTE* Buffer, int width, int height, long paddedsize, LPCTSTR bmpfile);
BYTE* LoadBMP(int* width, int* height, long* size, LPCTSTR bmpfile);

#endif /* BMP_H_ */
