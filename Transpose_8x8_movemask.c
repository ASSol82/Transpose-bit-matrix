

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <immintrin.h>
#include <malloc.h>
#include <time.h>
#include "Settings.h"
#include "TransposeBitMatrix_macros.h"
#include "MemoryAlign.h"


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
CreateConvertBitslice_square(Transpose_256x256_8x8_square, 256)


CreateConvertBitslice(ConvertBitslice256x128, 256, 128)
CreateConvertBitslice(ConvertBitslice128x256, 128, 256)


CreateConvertBitslice(ConvertBitslice256x64, 256, 64)
CreateConvertBitslice(ConvertBitslice64x256, 64, 256)


//manual create function, more effective then macros create
//void ConvertBitslice_256x256_movemask(const uint8_t* src, uint8_t* dst)
//{
//	DECL_PERM DECL_INDEX1((256)/8)
//	_ALIGN(32) __m256i w256[4], t;
//	for (uint32_t i = 0; i < 8; ++i)
//	{
//		uint32_t* p_src32 = (uint32_t*)src + i * 256;
//		uint32_t* p_dst32 = (uint32_t*)dst + i;
//		for (uint32_t k = 0; k < 8; ++k)
//		{
//			ConvertBitslice_32x32_movemask_macros(t, w256, p_src32, p_dst32, 8, 8);
//			p_src32 += 1;
//			p_dst32 += 256;
//		}
//	}
//}
//or create function ConvertBitslice_256x256 possible with macros
//CreateConvertBitsliceMovemask(ConvertBitslice_256x256_movemask, 256, 256)


CreateConvertBitsliceMovemask(ConvertBitslice256x128_movemask, 256, 128)
CreateConvertBitsliceMovemask(ConvertBitslice128x256_movemask, 128, 256)

CreateConvertBitsliceMovemask(ConvertBitslice256x64_movemask, 256, 64)
CreateConvertBitsliceMovemask(ConvertBitslice64x256_movemask, 64, 256)

CreateConvertBitsliceMovemask_square(Transpose_256x256_movemask_square, 256)


//void Transpose_256x256_movemask_square(uint8_t* data)
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
		uint8_t* src = malloc(8192), * dst = malloc(8192);//, * dst2 = malloc(8192); //uint8_t src[8192], dst[8192], dst2[8192];
		InitAr_uint8(src, 8192);
		InitAr_uint8(dst, 8192);
		Transpose_256x256_8x8_square(dst);
		//ConvertBitslice_256x256(src, dst);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 256; ++j)
			{
				if (GetBitItem(src, 32, i, j) != GetBitItem(dst, 32, j, i))
				{
					printf("Transpose_256x256_8x8_square error\n");
					//printf("ConvertBitslice_256x256 error\n");
					return -1;
				}
			}
		}

		Transpose_256x256_8x8_square(dst);
		//ConvertBitslice_256x256(dst, dst2);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 256; ++j)
			{
				if (GetBitItem(src, 32, i, j) != GetBitItem(dst, 32, i, j))
				{
					printf("Transpose_256x256_8x8_square error\n");
					//printf("ConvertBitslice_256x256 error\n");
					return -1;
				}
			}
		}
		printf("Transpose_256x256_8x8_square ok\n");
		//printf("ConvertBitslice_256x256 ok\n");
		free(src); free(dst); //free(dst2);
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
		uint8_t* src = malloc(8192), * dst = malloc(8192); //, * dst2 = malloc(8192);
		InitAr_uint8(src, 8192);
		InitAr_uint8(dst, 8192);
		Transpose_256x256_movemask_square(dst);
		//ConvertBitslice_256x256_movemask(src, dst);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 256; ++j)
			{
				if (GetBitItem(src, 32, i, j) != GetBitItem(dst, 32, j, i))
				{
					printf("Transpose_256x256_movemask_square error\n");
					//printf("ConvertBitslice_256x256_movemask error\n");
					return -1;
				}
			}
		}

		Transpose_256x256_movemask_square(dst);
		//ConvertBitslice_256x256_movemask(dst, dst2);
		for (uint32_t i = 0; i < 256; ++i)
		{
			for (uint32_t j = 0; j < 256; ++j)
			{
				if (GetBitItem(src, 32, i, j) != GetBitItem(dst, 32, i, j))
				{
					printf("Transpose_256x256_movemask_square error\n");
					//printf("ConvertBitslice_256x256_movemask error\n");
					return -1;
				}
			}
		}
		printf("Transpose_256x256_movemask_square ok\n");
		//printf("ConvertBitslice_256x256_movemask ok\n");
		free(src); free(dst); //free(dst2);
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


float Transpose_256x256_8x8_benchmark()
{
	//printf("Transpose_256x256_8x8_benchmark\n");
	_ALIGN(32) __m256i data[256];
	for (uint32_t i=0;i<256;++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		Transpose_256x256_8x8_square((uint8_t*)data); //ConvertToBitslice_1024x1024(i & 1 ? arr_dst : arr_src, i & 1 ? arr_src : arr_dst);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


float Transpose_256x128_8x8_benchmark()
{
	//printf("Transpose_256x128_8x8_benchmark\n");
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

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


float Transpose_256x64_8x8_benchmark()
{
	//printf("Transpose_256x64_8x8_benchmark\n");
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

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


float Transpose_256x256_Movemask_benchmark()
{
	//printf("Transpose_256x256_Movemask_benchmark\n");
	_ALIGN(32) __m256i data[256];
	for (uint32_t i=0;i<256;++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		Transpose_256x256_movemask_square((uint8_t*)data); //ConvertBitslice_256x256_square((uint8_t*)data); //ConvertToBitslice_1024x1024(i & 1 ? arr_dst : arr_src, i & 1 ? arr_src : arr_dst);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


float Transpose_256x128_Movemask_benchmark()
{
	//printf("Transpose_256x128_Movemask_benchmark\n");
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

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


float Transpose_256x64_Movemask_benchmark()
{
	//printf("Transpose_256x64_Movemask_benchmark\n");
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
		ConvertBitslice256x64_movemask((uint8_t*)(data), (uint8_t*)(data+64));
		ConvertBitslice64x256_movemask((uint8_t*)(data+64), (uint8_t*)data);
		//(i&1) ? ConvertBitslice64x256_movemask((uint8_t*)(data+64), (uint8_t*)data) : ConvertBitslice256x64_movemask((uint8_t*)(data), (uint8_t*)(data+64));
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


//#endif //__TRANSPOSE_256x256_ALG1__
