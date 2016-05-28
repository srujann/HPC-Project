/*
 * FileConverterWrapper.h
 *
 *  Created on: 22-May-2016
 *      Author: Karan Mehta
 */

#ifndef FILECONVERTERWRAPPER_H_
#define FILECONVERTERWRAPPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <cwchar>
#include <string>
#include <iostream>
#include "bmp.h"

using namespace std;

enum codeType {sequential, cuda_DP, cuda_Greedy};

void RGBtoBMP(string s, codeType ct);
void BMPtoRGB(string s, codeType ct);

#endif /* FILECONVERTERWRAPPER_H_ */
