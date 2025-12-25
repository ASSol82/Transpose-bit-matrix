// Transpose binary matrix, examples
// Author: Anatoly Solovyev, soloviov-anatoly@mail.ru


#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <malloc.h>


#define _OR(a,b) _mm256_or_si256((a), (b))
#define _AND(a,b) _mm256_and_si256((a), (b))
#define _SHIFTR64(a,count) _mm256_srli_epi64(a, count) // сдвиг в сторону младших разрядов в рамках 64-х битовых отрезков
#define _SHIFTL64(a,count) _mm256_slli_epi64(a, count) // сдвиг в сторону старших разрядов в рамках 64-х битовых отрезков


#define transpose8x8_4_macros(x) { \
	x = _OR(_OR(_AND(x, c1), _SHIFTL64(_AND(x, c2), 7)), _AND(_SHIFTR64(x, 7), c2)); \
	x = _OR(_OR(_AND(x, c3), _SHIFTL64(_AND(x, c4), 14)), _AND(_SHIFTR64(x, 14), c4)); \
	x = _OR(_OR(_AND(x, c5), _SHIFTL64(_AND(x, c6), 28)), _AND(_SHIFTR64(x, 28), c6)); }


#define DECL_CONST_C \
const __m256i c1 = _mm256_set1_epi64x(0xAA55AA55AA55AA55LL); \
const __m256i c2 = _mm256_set1_epi64x(0x00AA00AA00AA00AALL); \
const __m256i c3 = _mm256_set1_epi64x(0xCCCC3333CCCC3333LL); \
const __m256i c4 = _mm256_set1_epi64x(0x0000CCCC0000CCCCLL); \
const __m256i c5 = _mm256_set1_epi64x(0xF0F0F0F00F0F0F0FLL); \
const __m256i c6 = _mm256_set1_epi64x(0x00000000F0F0F0F0LL);


#define DECL_PERM \
const __m256i perm = _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0); \
const __m256i perm8x32 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);


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
	DECL_CONST_C DECL_PERM \
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
	DECL_CONST_C DECL_PERM \
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


#define ConvertBitslice_32x32_movemask_out_macros(w256, dst32, offset) \
{ \
	__m256i	tmp = _mm256_set_epi64x(_mm256_extract_epi64(w256[3], 0), _mm256_extract_epi64(w256[2], 0), _mm256_extract_epi64(w256[1], 0), _mm256_extract_epi64(w256[0], 0)); \
	movemask_8(dst32, 0, offset, tmp) \
	tmp = _mm256_set_epi64x(_mm256_extract_epi64(w256[3], 1), _mm256_extract_epi64(w256[2], 1), _mm256_extract_epi64(w256[1], 1), _mm256_extract_epi64(w256[0], 1)); \
	movemask_8(dst32, 8 * offset, offset, tmp) \
	tmp = _mm256_set_epi64x(_mm256_extract_epi64(w256[3], 2), _mm256_extract_epi64(w256[2], 2), _mm256_extract_epi64(w256[1], 2), _mm256_extract_epi64(w256[0], 2)); \
	movemask_8(dst32, 16 * offset, offset, tmp) \
	tmp = _mm256_set_epi64x(_mm256_extract_epi64(w256[3], 3), _mm256_extract_epi64(w256[2], 3), _mm256_extract_epi64(w256[1], 3), _mm256_extract_epi64(w256[0], 3)); \
	movemask_8(dst32, 24 * offset, offset, tmp) \
}


#define ConvertBitslice_32x32_movemask_macros(src32, p32, offsetRead, offsetWrite) \
{ \
	__m256i w256[4]; \
	Read_32x32_macros(w256, src32, offsetRead) \
	ConvertBitslice_32x32_movemask_out_macros(w256, p32, offsetWrite) \
}


#define CreateConvertBitsliceMovemask(fn, row, col) \
void fn(const uint8_t* src, uint8_t* dst) \
{ \
	DECL_PERM \
	for (uint32_t i = 0; i < ((row) / 32); ++i) \
	{ \
		uint32_t* p_src32 = (uint32_t*)src + i * (((col) / 32) * 32); \
		uint32_t* p_dst32 = (uint32_t*)dst + i; \
		for (uint32_t k = 0; k < ((col) / 32); ++k) \
		{ \
			ConvertBitslice_32x32_movemask_macros(p_src32, p_dst32, ((col) / 32), ((row) / 32)); \
			p_src32 += 1; \
			p_dst32 += (((row) / 32) * 32); \
		} \
	} \
}


// transpose only square matrix nxn, n divided by 32
#define CreateConvertBitsliceMovemask_square(fn, n) \
void fn(uint8_t* data) \
{ \
	DECL_CONST_C DECL_PERM \
	__m256i p1[8], * p2 = p1 + 4; \
	for (uint32_t i = 0; i < (n>>5); ++i) \
	{ \
		uint32_t* p_src32 = (uint32_t*)data + i * n + i, * p_dst32 = p_src32; \
		ConvertBitslice_32x32_movemask_macros(p_src32, p_src32, (n>>5), (n>>5)); \
		for (uint32_t j = i + 1; j < (n>>5); ++j) \
		{ \
			p_src32 += 1; p_dst32 += n; \
			Read_32x32_macros(p1, p_src32, (n>>5)); \
			Read_32x32_macros(p2, p_dst32, (n>>5)); \
			ConvertBitslice_32x32_movemask_out_macros(p2, p_src32, (n>>5)); \
			ConvertBitslice_32x32_movemask_out_macros(p1, p_dst32, (n>>5)); \
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


CreateConvertBitslice(ConvertBitslice256x128, 256, 128)
CreateConvertBitslice(ConvertBitslice128x256, 128, 256)


//manual create function, more effective then macros create
void ConvertBitslice_256x256_movemask(const uint8_t* src, uint8_t* dst)
{
	DECL_PERM
	for (uint32_t i = 0; i < 8; ++i)
	{
		uint32_t* p_src32 = (uint32_t*)src + i * 256;
		uint32_t* p_dst32 = (uint32_t*)dst + i;
		for (uint32_t k = 0; k < 8; ++k)
		{
			ConvertBitslice_32x32_movemask_macros(p_src32, p_dst32, 8, 8);
			p_src32 += 1;
			p_dst32 += 256;
		}
	}
}
//or create function ConvertBitslice_256x256 possible with macros
//CreateConvertBitsliceMovemask(ConvertBitslice_256x256_movemask, 256, 256)


CreateConvertBitsliceMovemask(ConvertBitslice256x128_movemask, 256, 128)
CreateConvertBitsliceMovemask(ConvertBitslice128x256_movemask, 128, 256)


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


int main()
{
	Transpose8x8_example1();
	TransposeMovemask_example2();

	return 0;
}
