// Transpose binary matrix, examples
// Author: Anatoly Solovyev, soloviov-anatoly@mail.ru


#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <malloc.h>
#include <time.h>
#include "TransposeBitMatrix_512x512.h"
#include "MemoryAlign.h"


#define __GATHER__ 0 // 1 - use _mm256_i32gather_epi32, 0 - use _mm256_set_epi64x


#define ITER_ (1<<20)
#define ITER_HALF (ITER_/2)


#define _OR(a,b) _mm256_or_si256((a), (b))
#define _AND(a,b) _mm256_and_si256((a), (b))
#define _SHIFTR64(a,count) _mm256_srli_epi64(a, count) // сдвиг в сторону младших разрядов в рамках 64-х битовых отрезков
#define _SHIFTL64(a,count) _mm256_slli_epi64(a, count) // сдвиг в сторону старших разрядов в рамках 64-х битовых отрезков


#define DECL_CONST_C \
const __m256i c1 = _mm256_set1_epi64x(0xAA55AA55AA55AA55LL); \
const __m256i c2 = _mm256_set1_epi64x(0x00AA00AA00AA00AALL); \
const __m256i c3 = _mm256_set1_epi64x(0xCCCC3333CCCC3333LL); \
const __m256i c4 = _mm256_set1_epi64x(0x0000CCCC0000CCCCLL); \
const __m256i c5 = _mm256_set1_epi64x(0xF0F0F0F00F0F0F0FLL); \
const __m256i c6 = _mm256_set1_epi64x(0x00000000F0F0F0F0LL);


#define DECL_PERM \
_ALIGN(32) const __m256i perm = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0); \
_ALIGN(32) const __m256i perm8x32 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);


#if __GATHER__==0
#define DECL_INDEX1(of1)
#else
#define DECL_INDEX1(of1) _ALIGN(32) const __m256i index1 = _mm256_set_epi32((of1)*7,(of1)*6,(of1)*5,(of1)*4,(of1)*3,(of1)*2,(of1),0);
#endif

/* выше универсальное решение index1 */
/*_ALIGN(32) const __m256i index1 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);*/


#define transpose8x8_4_macros(x) { \
x = _OR(_OR(_AND(x, c1), _SHIFTL64(_AND(x, c2), 7)), _AND(_SHIFTR64(x, 7), c2)); \
x = _OR(_OR(_AND(x, c3), _SHIFTL64(_AND(x, c4), 14)), _AND(_SHIFTR64(x, 14), c4)); \
x = _OR(_OR(_AND(x, c5), _SHIFTL64(_AND(x, c6), 28)), _AND(_SHIFTR64(x, 28), c6)); }

#if __GATHER__==0
#define _mm256_set_8x32(p32,start,offset) _mm256_set_epi64x( \
	(uint64_t)p32[start+6*offset]	|	((uint64_t)p32[start+7*offset])<<32, \
	(uint64_t)p32[start+4*offset]	|	((uint64_t)p32[start+5*offset])<<32, \
	(uint64_t)p32[start+2*offset]	|	((uint64_t)p32[start+3*offset])<<32, \
	(uint64_t)p32[start+0*offset]	|	((uint64_t)p32[start+1*offset])<<32)
#define Read_32x32_macros(w256, src32, offset) \
	w256[0] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_set_8x32(src32,0,offset), perm), perm8x32); \
	w256[1] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_set_8x32(src32,8*offset, offset), perm), perm8x32); \
	w256[2] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_set_8x32(src32,16*offset,offset), perm), perm8x32); \
	w256[3] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_set_8x32(src32,24*offset,offset), perm), perm8x32);
#else
#define Read_32x32_macros(w256, src32, offset) \
	w256[0] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_i32gather_epi32((const int *)(src32), 			 index1, 1), perm), perm8x32); \
	w256[1] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_i32gather_epi32((const int *)(src32+offset* 8), index1, 1), perm), perm8x32); \
	w256[2] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_i32gather_epi32((const int *)(src32+offset*16), index1, 1), perm), perm8x32); \
	w256[3] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_i32gather_epi32((const int *)(src32+offset*24), index1, 1), perm), perm8x32);
#endif


#define Extract_epi32_macros(dst,src,s,offset) \
	dst[(s) * offset] = _mm256_extract_epi32(src, 0); \
	dst[((s)+1) * offset] = _mm256_extract_epi32(src, 1); \
	dst[((s)+2) * offset] = _mm256_extract_epi32(src, 2); \
	dst[((s)+3) * offset] = _mm256_extract_epi32(src, 3); \
	dst[((s)+16) * offset] = _mm256_extract_epi32(src, 4); \
	dst[((s)+17) * offset] = _mm256_extract_epi32(src, 5); \
	dst[((s)+18) * offset] = _mm256_extract_epi32(src, 6); \
	dst[((s)+19) * offset] = _mm256_extract_epi32(src, 7);


#define ConvertBitslice_32x32_out_macros(w256, dst32, offset) \
{ \
	transpose8x8_4_macros(w256[0]) \
	transpose8x8_4_macros(w256[1]) \
	transpose8x8_4_macros(w256[2]) \
	transpose8x8_4_macros(w256[3]) \
\
	__m256i t1, t2, t3; \
	t1 = _mm256_unpacklo_epi8(w256[0], w256[1]); \
	t2 = _mm256_unpacklo_epi8(w256[2], w256[3]); \
	t3 = _mm256_unpacklo_epi16(t1, t2); \
	Extract_epi32_macros(dst32,t3,0,offset) \
	t3 = _mm256_unpackhi_epi16(t1, t2); \
	Extract_epi32_macros(dst32,t3,4,offset) \
	t1 = _mm256_unpackhi_epi8(w256[0], w256[1]); \
	t2 = _mm256_unpackhi_epi8(w256[2], w256[3]); \
	t3 = _mm256_unpacklo_epi16(t1, t2); \
	Extract_epi32_macros(dst32,t3,8,offset) \
	t3 = _mm256_unpackhi_epi16(t1, t2); \
	Extract_epi32_macros(dst32,t3,12,offset) \
}


#define ConvertBitslice_32x32_macros(src32, dst32, offsetRead, offsetWrite) \
{ \
	__m256i w256[4]; \
	Read_32x32_macros(w256, src32, offsetRead) \
	ConvertBitslice_32x32_out_macros(w256, dst32, offsetWrite) \
}


// fn - name function, row - count row in bits, col - count column in bits, row and col divisible 32
#define CreateConvertBitslice(fn, row, col) \
void fn(const uint8_t* src, uint8_t* dst) \
{ \
	DECL_CONST_C DECL_PERM DECL_INDEX1((col)/8) \
	for (uint32_t i = 0; i < ((row) / 32); ++i) \
	{ \
		uint32_t* p_src32 = (uint32_t*)src + i * (((col) / 32) * 32); \
		uint32_t* p_dst32 = (uint32_t*)dst + i; \
		for (uint32_t k = 0; k < ((col) / 32); ++k) \
		{ \
			ConvertBitslice_32x32_macros(p_src32, p_dst32, ((col) / 32), ((row) / 32)); \
			p_src32 += 1; \
			p_dst32 += (((row) / 32) * 32); \
		} \
	} \
}


// transpose only square matrix nxn, n divided by 32
#define CreateConvertBitslice_square(fn, n) \
void fn(uint8_t* data) \
{ \
	DECL_CONST_C DECL_PERM DECL_INDEX1((n)/8) \
	__m256i p1[8], * p2 = p1 + 4; \
	for (uint32_t i = 0; i < (n>>5); ++i) \
	{ \
		uint32_t* p_src32 = (uint32_t*)data + i * n + i, * p_dst32 = p_src32; \
		ConvertBitslice_32x32_macros(p_src32, p_src32, (n>>5), (n>>5)); \
		for (uint32_t j = i + 1; j < (n>>5); ++j) \
		{ \
			p_src32 += 1; p_dst32 += n; \
			Read_32x32_macros(p1, p_src32, (n>>5)); \
			Read_32x32_macros(p2, p_dst32, (n>>5)); \
			ConvertBitslice_32x32_out_macros(p2, p_src32, (n>>5)); \
			ConvertBitslice_32x32_out_macros(p1, p_dst32, (n>>5)); \
		} \
	} \
}


#define movemask_8(p32,s,of,tmp) \
	p32[s+7*(of)] = _mm256_movemask_epi8(tmp); \
	p32[s+6*(of)] = _mm256_movemask_epi8(_SHIFTL64(tmp,1)); \
	p32[s+5*(of)] = _mm256_movemask_epi8(_SHIFTL64(tmp,2)); \
	p32[s+4*(of)] = _mm256_movemask_epi8(_SHIFTL64(tmp,3)); \
	p32[s+3*(of)] = _mm256_movemask_epi8(_SHIFTL64(tmp,4)); \
	p32[s+2*(of)] = _mm256_movemask_epi8(_SHIFTL64(tmp,5)); \
	p32[s+1*(of)] = _mm256_movemask_epi8(_SHIFTL64(tmp,6)); \
	p32[s+0*(of)] = _mm256_movemask_epi8(_SHIFTL64(tmp,7)); \


#define set_extract(w256, i) _mm256_set_epi64x(_mm256_extract_epi64(w256[3], (i)), _mm256_extract_epi64(w256[2], (i)), _mm256_extract_epi64(w256[1], (i)), _mm256_extract_epi64(w256[0], (i)))


#define ConvertBitslice_32x32_movemask_out_macros(t, w256, dst32, offset) \
{ \
	t = set_extract(w256, 0); movemask_8(dst32, 0, offset, t) \
	t = set_extract(w256, 1); movemask_8(dst32, 8 * offset, offset, t) \
	t = set_extract(w256, 2); movemask_8(dst32, 16 * offset, offset, t) \
	t = set_extract(w256, 3); movemask_8(dst32, 24 * offset, offset, t) \
}


#define ConvertBitslice_32x32_movemask_macros(t, w256, src32, p32, offsetRead, offsetWrite) \
{ \
	Read_32x32_macros(w256, src32, offsetRead) \
	ConvertBitslice_32x32_movemask_out_macros(t, w256, p32, offsetWrite) \
}


#define CreateConvertBitsliceMovemask(fn, row, col) \
void fn(const uint8_t* src, uint8_t* dst) \
{ \
	DECL_PERM DECL_INDEX1((col)/8) \
	_ALIGN(32) __m256i w256[4], t; \
	for (uint32_t i = 0; i < ((row) / 32); ++i) \
	{ \
		uint32_t* p_src32 = (uint32_t*)src + i * (((col) / 32) * 32); \
		uint32_t* p_dst32 = (uint32_t*)dst + i; \
		for (uint32_t k = 0; k < ((col) / 32); ++k) \
		{ \
			ConvertBitslice_32x32_movemask_macros(t, w256, p_src32, p_dst32, ((col) / 32), ((row) / 32)); \
			p_src32 += 1; \
			p_dst32 += (((row) / 32) * 32); \
		} \
	} \
}


// transpose only square matrix nxn, n divided by 32
#define CreateConvertBitsliceMovemask_square(fn, n) \
void fn(uint8_t* data) \
{ \
	DECL_PERM DECL_INDEX1((n)/8) \
	_ALIGN(32) __m256i p1[4], p2[4], t; \
	for (uint32_t i = 0; i < (n>>5); ++i) \
	{ \
		uint32_t* p_src32 = (uint32_t*)data + i * n + i, * p_dst32 = p_src32; \
		ConvertBitslice_32x32_movemask_macros(t, p1, p_src32, p_src32, (n>>5), (n>>5)); \
		for (uint32_t j = i + 1; j < (n>>5); ++j) \
		{ \
			p_src32 += 1; p_dst32 += n; \
			Read_32x32_macros(p1, p_src32, (n>>5)); \
			Read_32x32_macros(p2, p_dst32, (n>>5)); \
			ConvertBitslice_32x32_movemask_out_macros(t, p2, p_src32, (n>>5)); \
			ConvertBitslice_32x32_movemask_out_macros(t, p1, p_dst32, (n>>5)); \
		} \
	} \
}


// считать битовый элемент матрицы на которую указывает указатель p, 
// strbyte - длина строки в байтах, чтобы правильно определить смещение элемента, str - номер строки, col - номер столбца, нумерация с 0.
#define GetBitItem(p, strbyte, str, col) ((((uint8_t*)(p))[(strbyte)*(str) + ((col)>>3)]>>((col)&7))&1)


void InitAr_uint8(void* ar_, const uint32_t count)
{
	uint8_t* ar = (uint8_t*)ar_;
	for (uint64_t i = 0; i < count; ++i)
	{
		ar[i] = (uint8_t)i;
	}
}


//manual create function 
//void ConvertBitslice_256x256(const uint8_t* src, uint8_t* dst)
//{
//	DECL_CONST_C DECL_PERM
//	for (uint32_t i = 0; i < 8; ++i)
//	{
//		uint32_t* p_src32 = (uint32_t*)src + i * 256;
//		uint32_t* p_dst32 = (uint32_t*)dst + i;
//		for (uint32_t k = 0; k < 8; ++k)
//		{
//			ConvertBitslice_32x32_macros(p_src32, p_dst32, 8, 8);
//			p_src32 += 1;
//			p_dst32 += 256;
//		}
//	}
//}
//or create function ConvertBitslice_256x256 possible with macros
CreateConvertBitslice(ConvertBitslice_256x256, 256, 256)
CreateConvertBitslice_square(Transpose8x8_256x256_square, 256)
CreateConvertBitslice_square(Transpose8x8_512x512_square, 512)


CreateConvertBitslice(ConvertBitslice256x128, 256, 128)
CreateConvertBitslice(ConvertBitslice128x256, 128, 256)


CreateConvertBitslice(ConvertBitslice256x64, 256, 64)
CreateConvertBitslice(ConvertBitslice64x256, 64, 256)


//manual create function, more effective then macros create
void ConvertBitslice_256x256_movemask(const uint8_t* src, uint8_t* dst)
{
	DECL_PERM DECL_INDEX1((256)/8)
	_ALIGN(32) __m256i w256[4], t;
	for (uint32_t i = 0; i < 8; ++i)
	{
		uint32_t* p_src32 = (uint32_t*)src + i * 256;
		uint32_t* p_dst32 = (uint32_t*)dst + i;
		for (uint32_t k = 0; k < 8; ++k)
		{
			ConvertBitslice_32x32_movemask_macros(t, w256, p_src32, p_dst32, 8, 8);
			p_src32 += 1;
			p_dst32 += 256;
		}
	}
}
//or create function ConvertBitslice_256x256 possible with macros
//CreateConvertBitsliceMovemask(ConvertBitslice_256x256_movemask, 256, 256)


CreateConvertBitsliceMovemask(ConvertBitslice256x128_movemask, 256, 128)
CreateConvertBitsliceMovemask(ConvertBitslice128x256_movemask, 128, 256)

CreateConvertBitsliceMovemask(ConvertBitslice256x64_movemask, 256, 64)
CreateConvertBitsliceMovemask(ConvertBitslice64x256_movemask, 64, 256)

CreateConvertBitsliceMovemask_square(TransposeMovemask_256x256_movemask_square, 256)
CreateConvertBitsliceMovemask_square(TransposeMovemask_512x512_square, 512)

//void TransposeMovemask_256x256_movemask_square(uint8_t* data)
//{
//	DECL_PERM
//	_ALIGN(32) __m256i p1[4], p2[4], t;
//	for (uint32_t i = 0; i < 8; ++i)
//	{
//		uint32_t* p_src32 = (uint32_t*)data + i * 256 + i, * p_dst32 = p_src32;
//		ConvertBitslice_32x32_movemask_macros(t, p1, p_src32, p_src32, 8, 8);
//		for (uint32_t j = i + 1; j < 8; ++j)
//		{
//			p_src32 += 1; p_dst32 += 256;
//			Read_32x32_macros(p1, p_src32, 8);
//			Read_32x32_macros(p2, p_dst32, 8);
//			ConvertBitslice_32x32_movemask_out_macros(t, p2, p_src32, 8);
//			ConvertBitslice_32x32_movemask_out_macros(t, p1, p_dst32, 8);
//		}
//	}
//}


// Проверка, что транспонирование выполняется корректно, т.е. формируется правильная результирующая матрица
int Transpose8x8_example1()
{
	{		
		uint8_t* src = malloc(8192), * dst = malloc(8192), * dst2 = malloc(8192); //uint8_t src[8192], dst[8192], dst2[8192];
		InitAr_uint8(src, 8192);
		ConvertBitslice_256x256(src, dst);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 256; ++j)
			{
				if (GetBitItem(src, 32, i, j) != GetBitItem(dst, 32, j, i))
				{
					printf("ConvertBitslice_256x256 error\n");
					return -1;
				}
			}
		}

		ConvertBitslice_256x256(dst, dst2);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 256; ++j)
			{
				if (GetBitItem(src, 32, i, j) != GetBitItem(dst2, 32, i, j))
				{
					printf("ConvertBitslice_256x256 error\n");
					return -1;
				}
			}
		}
		printf("ConvertBitslice_256x256 ok\n");
		free(src); free(dst); free(dst2);
	}

	{		
		uint8_t* src = malloc(4096), * dst = malloc(4096), * dst2 = malloc(4096); //uint8_t src[4096], dst[4096], dst2[4096];
		InitAr_uint8(src, 4096);
		ConvertBitslice256x128(src, dst);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 128; ++j)
			{
				if (GetBitItem(src, 16, i, j) != GetBitItem(dst, 32, j, i))
				{
					printf("ConvertBitslice256x128 error\n");
					return -1;
				}
			}
		}
		printf("ConvertBitslice_256x128 ok\n");

		ConvertBitslice128x256(dst, dst2);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 128; ++j)
			{
				if (GetBitItem(src, 16, i, j) != GetBitItem(dst2, 16, i, j))
				{
					printf("ConvertBitslice128x256 error\n");
					return -1;
				}
			}
		}
		printf("ConvertBitslice_128x256 ok\n");
		free(src); free(dst); free(dst2);
	}
	return 0;
}


// Проверка другого варианта транспонирования
int TransposeMovemask_example2()
{
	{
		uint8_t* src = malloc(8192), * dst = malloc(8192), * dst2 = malloc(8192);
		InitAr_uint8(src, 8192);
		ConvertBitslice_256x256_movemask(src, dst);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 256; ++j)
			{
				if (GetBitItem(src, 32, i, j) != GetBitItem(dst, 32, j, i))
				{
					printf("ConvertBitslice_256x256_movemask error\n");
					return -1;
				}
			}
		}

		ConvertBitslice_256x256_movemask(dst, dst2);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 256; ++j)
			{
				if (GetBitItem(src, 32, i, j) != GetBitItem(dst2, 32, i, j))
				{
					printf("ConvertBitslice_256x256_movemask error\n");
					return -1;
				}
			}
		}
		printf("ConvertBitslice_256x256_movemask ok\n");
		free(src); free(dst); free(dst2);
	}

	{
		uint8_t* src = malloc(4096), * dst = malloc(4096), * dst2 = malloc(4096); // память для матрицы 32x32
		InitAr_uint8(src, 4096);
		ConvertBitslice256x128_movemask(src, dst);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 128; ++j)
			{
				if (GetBitItem(src, 16, i, j) != GetBitItem(dst, 32, j, i))
				{
					printf("ConvertBitslice256x128_movemask error\n");
					return -1;
				}
			}
		}
		printf("ConvertBitslice_256x128_movemask ok\n");

		ConvertBitslice128x256_movemask(dst, dst2);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 128; ++j)
			{
				if (GetBitItem(src, 16, i, j) != GetBitItem(dst2, 16, i, j))
				{
					printf("ConvertBitslice128x256_movemask error\n");
					return -1;
				}
			}
		}
		printf("ConvertBitslice_128x256_movemask ok\n");
		free(src); free(dst); free(dst2);
	}
	return 0;
}


int Transpose8x8_256x256_benchmark()
{
	printf("Transpose8x8_256x256_benchmark\n");
	_ALIGN(32) __m256i data[256];
	for (uint32_t i=0;i<256;++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		Transpose8x8_256x256_square((uint8_t*)data); //ConvertToBitslice_1024x1024(i & 1 ? arr_dst : arr_src, i & 1 ? arr_src : arr_dst);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


int Transpose8x8_256x128_benchmark()
{
	printf("Transpose8x8_256x128_benchmark\n");
	_ALIGN(32) __m256i data[128*2];
	for (uint32_t i=0;i<128;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2,
			(uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1));
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		(i&1) ? ConvertBitslice128x256((uint8_t*)(data+128), (uint8_t*)data) : ConvertBitslice256x128((uint8_t*)(data), (uint8_t*)(data+128));
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


int Transpose8x8_256x64_benchmark()
{
	printf("Transpose8x8_256x64_benchmark\n");
	_ALIGN(32) __m256i data[64*2];
	for (uint32_t i=0;i<64;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, 
			(uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1),
			(uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), 
			(uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3));
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		(i&1) ? ConvertBitslice64x256((uint8_t*)(data+64), (uint8_t*)data) : ConvertBitslice256x64((uint8_t*)(data), (uint8_t*)(data+64));
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


int TransposeMovemask_256x256_benchmark()
{
	printf("TransposeMovemask_256x256_benchmark\n");
	_ALIGN(32) __m256i data[256];
	for (uint32_t i=0;i<256;++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		TransposeMovemask_256x256_movemask_square((uint8_t*)data); //ConvertBitslice_256x256_square((uint8_t*)data); //ConvertToBitslice_1024x1024(i & 1 ? arr_dst : arr_src, i & 1 ? arr_src : arr_dst);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


int TransposeMovemask_256x128_benchmark()
{
	printf("TransposeMovemask_256x128_benchmark\n");
	_ALIGN(32) __m256i data[128*2];
	for (uint32_t i=0;i<128;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2,
			(uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1));
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		(i&1) ? ConvertBitslice128x256_movemask((uint8_t*)(data+128), (uint8_t*)data) : ConvertBitslice256x128_movemask((uint8_t*)(data), (uint8_t*)(data+128));
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


int TransposeMovemask_256x64_benchmark()
{
	printf("TransposeMovemask_256x64_benchmark\n");
	_ALIGN(32) __m256i data[64*2];
	for (uint32_t i=0;i<64;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, 
			(uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1),
			(uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), 
			(uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3));
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_HALF; ++i)
	{
		ConvertBitslice256x64_movemask((uint8_t*)data, (uint8_t*)(data+64));
		ConvertBitslice64x256_movemask((uint8_t*)(data+64), (uint8_t*)data);
		//(i&1) ? ConvertBitslice64x256_movemask((uint8_t*)(data+64), (uint8_t*)data) : ConvertBitslice256x64_movemask((uint8_t*)(data), (uint8_t*)(data+64));
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


void real_ortho_256x256_modify(__m256i data[]) {

	__m256i mask_l[8] = {
	  _mm256_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
	  _mm256_set1_epi64x(0xccccccccccccccccUL),
	  _mm256_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
	  _mm256_set1_epi64x(0xff00ff00ff00ff00UL),
	  _mm256_set1_epi64x(0xffff0000ffff0000UL),
	  _mm256_set1_epi64x(0xffffffff00000000UL),
	  _mm256_setr_epi64x(0UL,0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF),
	  _mm256_setr_epi64x(0UL,0UL,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF),

	};

	__m256i mask_r[8] = {
	  _mm256_set1_epi64x(0x5555555555555555UL),
	  _mm256_set1_epi64x(0x3333333333333333UL),
	  _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
	  _mm256_set1_epi64x(0x00ff00ff00ff00ffUL),
	  _mm256_set1_epi64x(0x0000ffff0000ffffUL),
	  _mm256_set1_epi64x(0x00000000ffffffffUL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF,0UL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0UL,0UL),
	};

	for (int i = 0; i < 8; i++) {
		int n = (1UL << i);
		for (int j = 0; j < 256; j += (2 * n))
			for (int k = 0; k < n; k++) {
				__m256i u = _mm256_and_si256(data[j + k], mask_r[i]);
				__m256i v = _mm256_and_si256(data[j + k], mask_l[i]);
				__m256i x = _mm256_and_si256(data[j + n + k], mask_r[i]);
				__m256i y = _mm256_and_si256(data[j + n + k], mask_l[i]);
				if (i <= 5) {
					data[j + k] = _mm256_or_si256(u, _mm256_slli_epi64(x, n));
					data[j + n + k] = _mm256_or_si256(_mm256_srli_epi64(v, n), y);
				}
				else if (i == 6) {
					/* Note the "inversion" of srli and slli. */
					data[j + k] = _mm256_or_si256(u, _mm256_slli_si256(x, 8));
					data[j + n + k] = _mm256_or_si256(_mm256_srli_si256(v, 8), y);
				}
				else {
					data[j + k] = _mm256_or_si256(u, _mm256_permute2x128_si256(x, x, 1));
					data[j + n + k] = _mm256_or_si256(_mm256_permute2x128_si256(v, v, 1), y);
				}
			}
	}
}


void real_ortho_to_128x128x2_modify(const uint8_t *src, __m256i data[])
{
	__m256i mask_l[8] = {
	  _mm256_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
	  _mm256_set1_epi64x(0xccccccccccccccccUL),
	  _mm256_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
	  _mm256_set1_epi64x(0xff00ff00ff00ff00UL),
	  _mm256_set1_epi64x(0xffff0000ffff0000UL),
	  _mm256_set1_epi64x(0xffffffff00000000UL),
	  _mm256_setr_epi64x(0UL,0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF),
	  _mm256_setr_epi64x(0UL,0UL,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF),
	};

	__m256i mask_r[8] = {
	  _mm256_set1_epi64x(0x5555555555555555UL),
	  _mm256_set1_epi64x(0x3333333333333333UL),
	  _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
	  _mm256_set1_epi64x(0x00ff00ff00ff00ffUL),
	  _mm256_set1_epi64x(0x0000ffff0000ffffUL),
	  _mm256_set1_epi64x(0x00000000ffffffffUL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF,0UL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0UL,0UL),
	};

	uint64_t *src64 = (uint64_t*)src;

#if __GATHER__==0	
	for (uint32_t i=0;i<128;++i)
	{
		data[i] = _mm256_setr_epi64x(src64[i*2], src64[i*2+1], src64[(i+128)*2], src64[(i+128)*2+1]);
	}
#else
	const __m256i idx = _mm256_setr_epi64x(0,1*8,256*8,257*8); //_mm256_setr_epi64x(0,1*8,256*8,257*8);
	for (uint32_t i=0;i<128;++i)
	{
		data[i] = _mm256_i64gather_epi64((const long long int *)(src64+i*2), idx, 1);
	}
#endif

	for (int i = 0; i < 7; i++) {
		int n = (1UL << i);
		for (int j = 0; j < 128; j += (2 * n))
			for (int k = 0; k < n; k++) {
				__m256i u = _mm256_and_si256(data[j + k], mask_r[i]);
				__m256i v = _mm256_and_si256(data[j + k], mask_l[i]);
				__m256i x = _mm256_and_si256(data[j + n + k], mask_r[i]);
				__m256i y = _mm256_and_si256(data[j + n + k], mask_l[i]);
				if (i <= 5) {
					data[j + k] = _mm256_or_si256(u, _mm256_slli_epi64(x, n));
					data[j + n + k] = _mm256_or_si256(_mm256_srli_epi64(v, n), y);
				}
				else if (i == 6) {
					/* Note the "inversion" of srli and slli. */
					data[j + k] = _mm256_or_si256(u, _mm256_slli_si256(x, 8));
					data[j + n + k] = _mm256_or_si256(_mm256_srli_si256(v, 8), y);
				}
				//else {
				//	data[j + k] = _mm256_or_si256(u, _mm256_permute2x128_si256(x, x, 1));
				//	data[j + n + k] = _mm256_or_si256(_mm256_permute2x128_si256(v, v, 1), y);
				//}
			}
	}
}


void real_ortho_from_128x128x2_modify(const __m256i *src, uint8_t *dst)
{
	__m256i mask_l[8] = {
	  _mm256_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
	  _mm256_set1_epi64x(0xccccccccccccccccUL),
	  _mm256_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
	  _mm256_set1_epi64x(0xff00ff00ff00ff00UL),
	  _mm256_set1_epi64x(0xffff0000ffff0000UL),
	  _mm256_set1_epi64x(0xffffffff00000000UL),
	  _mm256_setr_epi64x(0UL,0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF),
	  _mm256_setr_epi64x(0UL,0UL,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF),
	};

	__m256i mask_r[8] = {
	  _mm256_set1_epi64x(0x5555555555555555UL),
	  _mm256_set1_epi64x(0x3333333333333333UL),
	  _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
	  _mm256_set1_epi64x(0x00ff00ff00ff00ffUL),
	  _mm256_set1_epi64x(0x0000ffff0000ffffUL),
	  _mm256_set1_epi64x(0x00000000ffffffffUL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF,0UL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0UL,0UL),
	};

//	__m256i *data = src;
	_ALIGN(32) __m256i data[128];
	for (uint32_t i=0;i<128;++i) data[i]=src[i];

	for (int i = 0; i < 7; i++) {
		int n = (1UL << i);
		for (int j = 0; j < 128; j += (2 * n))
			for (int k = 0; k < n; k++) {
				__m256i u = _mm256_and_si256(data[j + k], mask_r[i]);
				__m256i v = _mm256_and_si256(data[j + k], mask_l[i]);
				__m256i x = _mm256_and_si256(data[j + n + k], mask_r[i]);
				__m256i y = _mm256_and_si256(data[j + n + k], mask_l[i]);
				if (i <= 5) {
					data[j + k] = _mm256_or_si256(u, _mm256_slli_epi64(x, n));
					data[j + n + k] = _mm256_or_si256(_mm256_srli_epi64(v, n), y);
				}
				else if (i == 6) {
					/* Note the "inversion" of srli and slli. */
					data[j + k] = _mm256_or_si256(u, _mm256_slli_si256(x, 8));
					data[j + n + k] = _mm256_or_si256(_mm256_srli_si256(v, 8), y);
				}
				//else {
				//	data[j + k] = _mm256_or_si256(u, _mm256_permute2x128_si256(x, x, 1));
				//	data[j + n + k] = _mm256_or_si256(_mm256_permute2x128_si256(v, v, 1), y);
				//}
			}
	}

	uint64_t *dst64 = (uint64_t*)dst;
	for (uint32_t i=0;i<128;++i)
	{
//		dst64[i*2] = _mm256_extract_epi64(data[i], 0);
//		dst64[i*2+1] = _mm256_extract_epi64(data[i], 1);
//		dst64[(i+128)*2] = _mm256_extract_epi64(data[i], 2);
//		dst64[(i+128)*2+1] = _mm256_extract_epi64(data[i], 3);

		*(__m128i*)(dst64+i*2) = _mm256_extracti128_si256(data[i], 0);
		*(__m128i*)(dst64+(i+128)*2) = _mm256_extracti128_si256(data[i], 1);
	}
}


void real_ortho_to_64x64x4_modify(const uint8_t* src, __m256i data[])
{
	__m256i mask_l[8] = {
	  _mm256_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
	  _mm256_set1_epi64x(0xccccccccccccccccUL),
	  _mm256_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
	  _mm256_set1_epi64x(0xff00ff00ff00ff00UL),
	  _mm256_set1_epi64x(0xffff0000ffff0000UL),
	  _mm256_set1_epi64x(0xffffffff00000000UL),
	  _mm256_setr_epi64x(0UL,0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF),
	  _mm256_setr_epi64x(0UL,0UL,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF),
	};

	__m256i mask_r[8] = {
	  _mm256_set1_epi64x(0x5555555555555555UL),
	  _mm256_set1_epi64x(0x3333333333333333UL),
	  _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
	  _mm256_set1_epi64x(0x00ff00ff00ff00ffUL),
	  _mm256_set1_epi64x(0x0000ffff0000ffffUL),
	  _mm256_set1_epi64x(0x00000000ffffffffUL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF,0UL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0UL,0UL),
	};

	uint64_t *src64 = (uint64_t*)src;

#if __GATHER__==0
	for (uint32_t i=0;i<64;++i)
	{
		data[i] = _mm256_setr_epi64x(src64[i], src64[i+64], src64[i+128], src64[i+192]);
	}
#else
	const __m256i offsets = _mm256_setr_epi64x(0,64,128,192);
	for (uint32_t i=0;i<64;++i)
	{
		data[i] = _mm256_i64gather_epi64((const long long int *)src64+i, offsets, 8);
	}
#endif
	
	for (int i = 0; i < 6; i++) {
		int n = (1UL << i);
		for (int j = 0; j < 64; j += (2 * n))
			for (int k = 0; k < n; k++) {
				__m256i u = _mm256_and_si256(data[j + k], mask_r[i]);
				__m256i v = _mm256_and_si256(data[j + k], mask_l[i]);
				__m256i x = _mm256_and_si256(data[j + n + k], mask_r[i]);
				__m256i y = _mm256_and_si256(data[j + n + k], mask_l[i]);
				data[j + k] = _mm256_or_si256(u, _mm256_slli_epi64(x, n));
				data[j + n + k] = _mm256_or_si256(_mm256_srli_epi64(v, n), y);
				//if (i <= 5) {
				//	data[j + k] = _mm256_or_si256(u, _mm256_slli_epi64(x, n));
				//	data[j + n + k] = _mm256_or_si256(_mm256_srli_epi64(v, n), y);
				//}
				//else if (i == 6) {
				//	/* Note the "inversion" of srli and slli. */
				//	data[j + k] = _mm256_or_si256(u, _mm256_slli_si256(x, 8));
				//	data[j + n + k] = _mm256_or_si256(_mm256_srli_si256(v, 8), y);
				//}
				//else {
				//	data[j + k] = _mm256_or_si256(u, _mm256_permute2x128_si256(x, x, 1));
				//	data[j + n + k] = _mm256_or_si256(_mm256_permute2x128_si256(v, v, 1), y);
				//}
			}
	}
}


void real_ortho_from_64x64x4_modify(const __m256i* src, uint8_t *dst)
{
	__m256i mask_l[8] = {
	  _mm256_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
	  _mm256_set1_epi64x(0xccccccccccccccccUL),
	  _mm256_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
	  _mm256_set1_epi64x(0xff00ff00ff00ff00UL),
	  _mm256_set1_epi64x(0xffff0000ffff0000UL),
	  _mm256_set1_epi64x(0xffffffff00000000UL),
	  _mm256_setr_epi64x(0UL,0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF),
	  _mm256_setr_epi64x(0UL,0UL,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF),
	};

	__m256i mask_r[8] = {
	  _mm256_set1_epi64x(0x5555555555555555UL),
	  _mm256_set1_epi64x(0x3333333333333333UL),
	  _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
	  _mm256_set1_epi64x(0x00ff00ff00ff00ffUL),
	  _mm256_set1_epi64x(0x0000ffff0000ffffUL),
	  _mm256_set1_epi64x(0x00000000ffffffffUL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF,0UL),
	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0UL,0UL),
	};

	_ALIGN(32) __m256i data[64];
	for (uint32_t i=0;i<64;++i) data[i]=src[i];

	for (int i = 0; i < 6; i++) {
		int n = (1UL << i);
		for (int j = 0; j < 64; j += (2 * n))
			for (int k = 0; k < n; k++) {
				__m256i u = _mm256_and_si256(data[j + k], mask_r[i]);
				__m256i v = _mm256_and_si256(data[j + k], mask_l[i]);
				__m256i x = _mm256_and_si256(data[j + n + k], mask_r[i]);
				__m256i y = _mm256_and_si256(data[j + n + k], mask_l[i]);
				data[j + k] = _mm256_or_si256(u, _mm256_slli_epi64(x, n));
				data[j + n + k] = _mm256_or_si256(_mm256_srli_epi64(v, n), y);
				//if (i <= 5) {
				//	data[j + k] = _mm256_or_si256(u, _mm256_slli_epi64(x, n));
				//	data[j + n + k] = _mm256_or_si256(_mm256_srli_epi64(v, n), y);
				//}
				//else if (i == 6) {
				//	/* Note the "inversion" of srli and slli. */
				//	data[j + k] = _mm256_or_si256(u, _mm256_slli_si256(x, 8));
				//	data[j + n + k] = _mm256_or_si256(_mm256_srli_si256(v, 8), y);
				//}
				//else {
				//	data[j + k] = _mm256_or_si256(u, _mm256_permute2x128_si256(x, x, 1));
				//	data[j + n + k] = _mm256_or_si256(_mm256_permute2x128_si256(v, v, 1), y);
				//}
			}
	}
	
	uint64_t *dst64 = (uint64_t*)dst;
	for (uint32_t i=0;i<64;++i)
	{
		dst64[i] = _mm256_extract_epi64(data[i], 0);
		dst64[i+64] = _mm256_extract_epi64(data[i], 1);
		dst64[i+128] = _mm256_extract_epi64(data[i], 2);
		dst64[i+192] = _mm256_extract_epi64(data[i], 3);
	}
}


int TransposeUsuba_256x256_benchmark()
{
	printf("TransposeUsuba_256x256_benchmark\n");
	_ALIGN(32) __m256i data[256];
	for (uint32_t i=0;i<256;++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		real_ortho_256x256_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


int TransposeUsuba_256x128_benchmark()
{
	printf("TransposeUsuba_256x128_benchmark\n");
	_ALIGN(32) __m256i data[128*2];
	for (uint32_t i=0;i<128;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2,
			(uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1));
	}

	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		(i&1) ? real_ortho_from_128x128x2_modify((data+128), (uint8_t*)data) : real_ortho_to_128x128x2_modify((uint8_t*)data, (data+128));
		//real_ortho_128x128x2_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


int TransposeUsuba_256x64_benchmark()
{
	printf("TransposeUsuba_256x64_benchmark\n");
	_ALIGN(32) __m256i data[64*2];
	for (uint32_t i=0;i<64;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, 
			(uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1),
			(uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), 
			(uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3));
	}

	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		(i&1) ? real_ortho_from_64x64x4_modify((data+64), (uint8_t*)data) : real_ortho_to_64x64x4_modify((uint8_t*)data, (data+64));
		//real_ortho_128x128x2_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return 0;
}


// Сравнение, что все три варианта транспонирования выдают одинаковый результат
int compare_test_256x256()
{
	printf("compare_test_256x256 ");
	_ALIGN(32) __m256i data[256];
	for (uint32_t i=0;i<256;++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
	}
	_ALIGN(32) __m256i data2[256];
	for (uint32_t i=0;i<256;++i)
	{
		data2[i] = _mm256_set1_epi8((uint8_t)i);
	}
	
	TransposeMovemask_256x256_movemask_square((uint8_t*)data);
	real_ortho_256x256_modify(data2); //ConvertBitslice_256x256_square((uint8_t*)data); //ConvertToBitslice_1024x1024(i & 1 ? arr_dst : arr_src, i & 1 ? arr_src : arr_dst);
	
	for (uint32_t i=0;i<256;++i)
	{
		__m256i x = _mm256_xor_si256(data[i], data2[i]);
		if (!_mm256_testz_si256(x, x)) // 1, если все биты нулевые (т.е. a == b), иначе 0
		{
			printf("No equal\n");			
		}
	}
	
	for (uint32_t i=0;i<256;++i)
	{
		data2[i] = _mm256_set1_epi8((uint8_t)i);
	}
	Transpose8x8_256x256_square((uint8_t*)data2);

	for (uint32_t i=0;i<256;++i)
	{
		__m256i x = _mm256_xor_si256(data[i], data2[i]);
		if (!_mm256_testz_si256(x, x)) // 1, если все биты нулевые (т.е. a == b), иначе 0
		{
			printf("No equal\n");			
		}
	}
	
	printf("ok\n");
	
	return 0;
}


int compare_test_256x128()
{
	printf("compare_test_256x128 ");
	_ALIGN(32) __m256i data[128*2];
	for (uint32_t i=0;i<128;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2,
			(uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1));
	}
	_ALIGN(32) __m256i data2[128*2];
	for (uint32_t i=0;i<128;++i)
	{
		data2[i] = _mm256_setr_epi8((uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2,
			(uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1));
	}
	
	ConvertBitslice256x128_movemask((uint8_t*)data, (uint8_t*)(data+128));
	real_ortho_to_128x128x2_modify((uint8_t*)data2, data2+128); //ConvertBitslice_256x256_square((uint8_t*)data); //ConvertToBitslice_1024x1024(i & 1 ? arr_dst : arr_src, i & 1 ? arr_src : arr_dst);
	
	for (uint32_t i=0;i<128;++i)
	{
		__m256i x = _mm256_xor_si256(data[128+i], data2[128+i]);
		if (!_mm256_testz_si256(x, x)) // 1, если все биты нулевые (т.е. a == b), иначе 0
		{
			printf("No equal\n");			
		}
	}
	
	for (uint32_t i=0;i<128;++i)
	{
		data2[i] = _mm256_setr_epi8((uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2,
			(uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1));
	}
	ConvertBitslice256x128((uint8_t*)data2, (uint8_t*)(data2+128));

	for (uint32_t i=0;i<128;++i)
	{
		__m256i x = _mm256_xor_si256(data[128+i], data2[128+i]);
		if (!_mm256_testz_si256(x, x)) // 1, если все биты нулевые (т.е. a == b), иначе 0
		{
			printf("No equal\n");			
		}
	}
	
	printf("ok\n");
	
	return 0;
}


int compare_test_256x64()
{
	printf("compare_test_256x64 ");
	_ALIGN(32) __m256i data[64*2];
	for (uint32_t i=0;i<64;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, 
			(uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1),
			(uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), 
			(uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3));
	}
	_ALIGN(32) __m256i data2[64*2];
	for (uint32_t i=0;i<64;++i)
	{
		data2[i] = _mm256_setr_epi8((uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, 
			(uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1),
			(uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), 
			(uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3));
	}
	
	ConvertBitslice256x64_movemask((uint8_t*)data, (uint8_t*)(data+64));
	real_ortho_to_64x64x4_modify((uint8_t*)data2, data2+64); //ConvertBitslice_256x256_square((uint8_t*)data); //ConvertToBitslice_1024x1024(i & 1 ? arr_dst : arr_src, i & 1 ? arr_src : arr_dst);
	
	for (uint32_t i=0;i<64;++i)
	{
		__m256i x = _mm256_xor_si256(data[64+i], data2[64+i]);
		if (!_mm256_testz_si256(x, x)) // 1, если все биты нулевые (т.е. a == b), иначе 0
		{
			printf("No equal\n");			
		}
	}
	
	for (uint32_t i=0;i<64;++i)
	{
		data2[i] = _mm256_setr_epi8((uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, (uint8_t)i*4, 
			(uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1), (uint8_t)(i*4+1),
			(uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), (uint8_t)(i*4+2), 
			(uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3), (uint8_t)(i*4+3));
	}
	ConvertBitslice256x64((uint8_t*)data2, (uint8_t*)(data2+64));

	for (uint32_t i=0;i<64;++i)
	{
		__m256i x = _mm256_xor_si256(data[64+i], data2[64+i]);
		if (!_mm256_testz_si256(x, x)) // 1, если все биты нулевые (т.е. a == b), иначе 0
		{
			printf("No equal\n");			
		}
	}
	
	printf("ok\n");
	
	return 0;
}


int Transpose8x8_512x512_benchmark()
{
	printf("Transpose8x8_512x512_benchmark\n");
	__m256i* data = aligned_malloc(sizeof(__m256i) * 1024, 32);
	if (data == NULL)
	{
		printf("Error _aligned_malloc\n");
		return -1;
	}
	//_ALIGN(32) __m256i data[1024];
	for (uint32_t i = 0; i < 1024; ++i)
	{
		data[i] = _mm256_set1_epi32((uint8_t)i);
	}

	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		Transpose8x8_512x512_square((uint8_t*)data); //real_ortho_512x512_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	aligned_free(data);
	return 0;
}


int TransposeMovemask_512x512_benchmark()
{
	printf("TransposeMovemask_512x512_benchmark\n");
	__m256i* data = aligned_malloc(sizeof(__m256i) * 1024, 32);
	if (data == NULL)
	{
		printf("Error _aligned_malloc\n");
		return -1;
	}
	//_ALIGN(32) __m256i data[1024];
	for (uint32_t i = 0; i < 1024; ++i)
	{
		data[i] = _mm256_set1_epi32((uint8_t)i);
	}

	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		TransposeMovemask_512x512_square((uint8_t*)data); //real_ortho_512x512_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	aligned_free(data);
	return 0;
}


int compare_test_512x512()
{
	printf("compare_test_512x512 ");
	__m256i* data = aligned_malloc(sizeof(__m256i) * 1024 * 2, 32);
	if (data == NULL)
	{
		printf("Error _aligned_malloc\n");
		return -1;
	}
	__m256i* data2 = data + 1024;
	//_ALIGN(32) __m256i data[1024];
	for (uint32_t i = 0; i < 1024; ++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
		data2[i] = data[i];
	}
		
	real_ortho_512x512_modify(data);
	Transpose8x8_512x512_square((uint8_t*)data);
	TransposeMovemask_512x512_square((uint8_t*)data);
	real_ortho_512x512_modify(data);
	Transpose8x8_512x512_square((uint8_t*)data);
	real_ortho_512x512_modify(data);
	real_ortho_512x512_modify(data);
	TransposeMovemask_512x512_square((uint8_t*)data);
	Transpose8x8_512x512_square((uint8_t*)data);

	Transpose8x8_512x512_square((uint8_t*)data2);
	real_ortho_512x512_modify(data2);
	Transpose8x8_512x512_square((uint8_t*)data2);
	real_ortho_512x512_modify(data2);
	Transpose8x8_512x512_square((uint8_t*)data2);
	TransposeMovemask_512x512_square((uint8_t*)data2);
	real_ortho_512x512_modify(data2);

	for (uint32_t i = 0; i < 1024; ++i)
	{
		__m256i x = _mm256_xor_si256(data[i], data2[i]);
		if (!_mm256_testz_si256(x, x)) // 1, если все биты нулевые (т.е. a == b), иначе 0
		{
			printf("No equal\n");
			return -1;
		}
	}

	//for (uint32_t i = 0; i < 256; ++i)
	//{
	//	data2[i] = _mm256_set1_epi8((uint8_t)i);
	//}
	//Transpose8x8_256x256_square((uint8_t*)data2);

	//for (uint32_t i = 0; i < 256; ++i)
	//{
	//	__m256i x = _mm256_xor_si256(data[i], data2[i]);
	//	if (!_mm256_testz_si256(x, x)) // 1, если все биты нулевые (т.е. a == b), иначе 0
	//	{
	//		printf("No equal\n");
	//	}
	//}

	printf("ok\n");

	aligned_free(data);
	return 0;
}


int main()
{
	Transpose8x8_example1();
	TransposeMovemask_example2();
	compare_test_256x256();
	compare_test_256x128();
	compare_test_256x64();

	compare_test_512x512();
	TransposeUsuba_512x512_benchmark();
	Transpose8x8_512x512_benchmark();
	TransposeMovemask_512x512_benchmark();

	TransposeUsuba_256x256_benchmark();
	Transpose8x8_256x256_benchmark();
	TransposeMovemask_256x256_benchmark();

	TransposeUsuba_256x128_benchmark();
	Transpose8x8_256x128_benchmark();
	TransposeMovemask_256x128_benchmark();

	TransposeUsuba_256x64_benchmark();
	Transpose8x8_256x64_benchmark();
	TransposeMovemask_256x64_benchmark();

	return 0;
}
