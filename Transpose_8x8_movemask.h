

#ifndef __TRANSPOSE_8x8_MOVEMASK_HEADER__
#define __TRANSPOSE_8x8_MOVEMASK_HEADER__


#include <stdint.h>


void ConvertBitslice256x128(const uint8_t* src, uint8_t* dst);
void ConvertBitslice128x256(const uint8_t* src, uint8_t* dst);
void ConvertBitslice256x64(const uint8_t* src, uint8_t* dst);
void ConvertBitslice64x256(const uint8_t* src, uint8_t* dst);


void ConvertBitslice256x128_movemask(const uint8_t* src, uint8_t* dst);
void ConvertBitslice128x256_movemask(const uint8_t* src, uint8_t* dst);
void ConvertBitslice256x64_movemask(const uint8_t* src, uint8_t* dst);
void ConvertBitslice64x256_movemask(const uint8_t* src, uint8_t* dst);


void Transpose_256x256_8x8_square(uint8_t* data);
void Transpose_256x256_movemask_square(uint8_t* data);


float Transpose_256x256_8x8_benchmark();
float Transpose_256x128_8x8_benchmark();
float Transpose_256x64_8x8_benchmark();

float Transpose_256x256_Movemask_benchmark();
float Transpose_256x128_Movemask_benchmark();
float Transpose_256x64_Movemask_benchmark();


#endif
