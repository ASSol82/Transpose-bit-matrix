#ifndef __TRANSPOSE_ALGORITHM1__
#define __TRANSPOSE_ALGORITHM1__

#include <stdint.h>
#include <immintrin.h>

void Transpose_256x256_Alg1(__m256i data[]);

void Transpose_to_256x128_Alg1(const uint8_t *src, __m256i data[]); //128x128x2
void Transpose_from_256x128_Alg1(const __m256i *src, uint8_t *dst); //128x128x2

void Transpose_to_256x64_Alg1(const uint8_t* src, __m256i data[]); //64x64x4
void Transpose_from_256x64_Alg1(const __m256i* src, uint8_t *dst); //64x64x4

float Transpose_256x256_Alg1_benchmark();
float Transpose_256x128_Alg1_benchmark();
float Transpose_256x64_Alg1_benchmark();

#endif
