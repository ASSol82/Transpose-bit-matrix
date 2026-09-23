

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <immintrin.h>
#include <malloc.h>
#include <time.h>
#include "Settings.h"
#include "MemoryAlign.h"


//#ifdef __TRANSPOSE_256x256_ALG1__
//__attribute__((noinline))
void Transpose_256x256_Alg1(__m256i data[]) {

	__m256i mask_l[] = {
	  _mm256_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
	  _mm256_set1_epi64x(0xccccccccccccccccUL),
	  _mm256_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
	  _mm256_set1_epi64x(0xff00ff00ff00ff00UL),
	  _mm256_set1_epi64x(0xffff0000ffff0000UL),
	  _mm256_set1_epi64x(0xffffffff00000000UL)};//,
//	  _mm256_setr_epi64x(0UL,0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF),
//	  _mm256_setr_epi64x(0UL,0UL,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF),
//	};
	__m256i mask_r[] = {
	  _mm256_set1_epi64x(0x5555555555555555UL),
	  _mm256_set1_epi64x(0x3333333333333333UL),
	  _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
	  _mm256_set1_epi64x(0x00ff00ff00ff00ffUL),
	  _mm256_set1_epi64x(0x0000ffff0000ffffUL),
	  _mm256_set1_epi64x(0x00000000ffffffffUL)};//,
//	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0UL,0xFFFFFFFFFFFFFFFF,0UL),
//	  _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0UL,0UL),
//	};

//	const __m256i p8x32 = _mm256_set_epi32(7, 5, 6, 4, 3, 1, 2, 0);

	for (int i = 0; i < 6; i++) {
		int n = (1UL << i);
		for (int j = 0; j < 256; j += (2 * n))
			for (int k = 0; k < n; k++) {
				__m256i u = _mm256_and_si256(data[j + k], mask_r[i]);
				__m256i v = _mm256_and_si256(data[j + k], mask_l[i]);
				__m256i x = _mm256_and_si256(data[j + n + k], mask_r[i]);
				__m256i y = _mm256_and_si256(data[j + n + k], mask_l[i]);
//				if (i <= 5) {
					data[j + k] = _mm256_or_si256(u, _mm256_slli_epi64(x, n));
					data[j + n + k] = _mm256_or_si256(_mm256_srli_epi64(v, n), y);
//				}
//				else if (i == 6) {
//					/* Note the "inversion" of srli and slli. */
//					data[j + k] = _mm256_or_si256(u, _mm256_slli_si256(x, 8));
//					data[j + n + k] = _mm256_or_si256(_mm256_srli_si256(v, 8), y);
//				}
//				else {
//					data[j + k] = _mm256_or_si256(u, _mm256_permute2x128_si256(x, x, 1));
//					data[j + n + k] = _mm256_or_si256(_mm256_permute2x128_si256(v, v, 1), y);
//				}
			}
	}

//	// i=5
//	const int n = (1UL << 5); //32
//	for (int j = 0; j < 256; j += 64) {
//		for (int k = 0; k < 32; k++) {
//			__m256i u = _mm256_permutevar8x32_epi32(data[j + k], p8x32);
//			__m256i v = _mm256_permutevar8x32_epi32(data[j + 32 + k], p8x32);
//			data[j + k] = _mm256_unpacklo_epi32(data[j + k], v);
//			data[j + 32 + k] = _mm256_unpackhi_epi32(u, data[j + 32 + k]);
//		}
//	}
//	for (int j = 0; j < 256; j += 64) {
//		for (int k = 0; k < 32; k++) {
//			__m256i u = _mm256_srli_epi64(data[j + k], 32);
//			__m256i v = _mm256_slli_epi64(data[j + 32 + k], 32);
//			data[j + k] = _mm256_or_si256(_mm256_and_si256(data[j + k], mask_r[5]), v);
//			data[j + 32 + k] = _mm256_or_si256(_mm256_and_si256(data[j + 32 + k], mask_l[5]), u);
//		}
//	}
//	for (int j = 0; j < 256; j += 64) {
//		for (int k = 0; k < 32; k++) {
//			__m256i u = _mm256_srli_epi64(data[j + k], 32);
//			__m256i v = _mm256_slli_epi64(data[j + n + k], 32);
//			data[j + k] = _mm256_blend_epi32(data[j + k], v, 0xAA);
//			data[j + n + k] = _mm256_blend_epi32(u, data[j + n + k], 0xAA);
//		}
//	}
	// i=6
	//const uint32_t n = (1UL << 6); //64
	for (int j = 0; j < 256; j += 128)
	{
		for (int k = 0; k < 64; ++k)
		{
			__m256i u = data[j + k];
			data[j + k] = _mm256_unpacklo_epi64(u, data[j + 64 + k]);
			data[j + 64 + k] = _mm256_unpackhi_epi64(u, data[j + 64 + k]);
		}
	}
	// i=7
	//const uint32_t n = (1UL << 7); //n=128
	for (int k = 0; k < 128; ++k)
	{
		__m256i u = data[k];
		data[k] = _mm256_permute2x128_si256(data[k], data[128 + k], 32); // младшие половины data[k] || data[n+k]
		data[128 + k] = _mm256_permute2x128_si256(u, data[128 + k], 49); // // старшие половины data[k] || data[n+k]
	}
}


//__attribute__((noinline))
float Transpose_256x256_Alg1_benchmark()
{
	//printf("Transpose_256x256_Alg1_benchmark\n");
	_ALIGN(32) __m256i data[256];
	for (uint32_t i=0;i<256;++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		Transpose_256x256_Alg1(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed; //return 0;
}
//#endif


//Transpose_to_128x128x2_Alg1
void Transpose_to_256x128_Alg1(const uint8_t *src, __m256i data[]) //void real_ortho_to_128x128x2_modify(const uint8_t *src, __m256i data[])
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

	for (int i = 0; i < 6; i++) {
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
	//i=6
	for (int k = 0; k < 64; ++k)
	{
		__m256i u = data[k];
		data[k] = _mm256_unpacklo_epi64(u, data[64 + k]);
		data[64 + k] = _mm256_unpackhi_epi64(u, data[64 + k]);
	}
}

//Transpose_from_128x128x2_Alg1
void Transpose_from_256x128_Alg1(const __m256i *src, uint8_t *dst) //void real_ortho_from_128x128x2_modify(const __m256i *src, uint8_t *dst)
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

	for (int i = 0; i < 6; i++) {
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
//				else if (i == 6) {
//					/* Note the "inversion" of srli and slli. */
//					data[j + k] = _mm256_or_si256(u, _mm256_slli_si256(x, 8));
//					data[j + n + k] = _mm256_or_si256(_mm256_srli_si256(v, 8), y);
//				}
				//else {
				//	data[j + k] = _mm256_or_si256(u, _mm256_permute2x128_si256(x, x, 1));
				//	data[j + n + k] = _mm256_or_si256(_mm256_permute2x128_si256(v, v, 1), y);
				//}
			}
	}
	//i=6
	for (int k = 0; k < 64; ++k)
	{
		__m256i u = data[k];
		data[k] = _mm256_unpacklo_epi64(u, data[64 + k]);
		data[64 + k] = _mm256_unpackhi_epi64(u, data[64 + k]);
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


//Transpose_to_64x64x4_Alg1
void Transpose_to_256x64_Alg1(const uint8_t* src, __m256i data[]) //void real_ortho_to_64x64x4_modify(const uint8_t* src, __m256i data[])
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

	uint64_t* src64 = (uint64_t*)src;
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
			}
	}
}


//Transpose_from_64x64x4_Alg1
void Transpose_from_256x64_Alg1(const __m256i* src, uint8_t *dst) //void real_ortho_from_64x64x4_modify(const __m256i* src, uint8_t *dst)
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


float Transpose_256x128_Alg1_benchmark()
{
	//printf("Transpose_256x128_Alg1_benchmark\n");
	_ALIGN(32) __m256i data[128*2];
	for (uint32_t i=0;i<128;++i)
	{
		data[i] = _mm256_setr_epi8((uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2, (uint8_t)i*2,
			(uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1), (uint8_t)(i*2+1));
	}

	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_HALF; ++i)
	{
		Transpose_to_256x128_Alg1((uint8_t*)data, (data+128));
		Transpose_from_256x128_Alg1((data+128), (uint8_t*)data);
		//(i&1) ? Transpose_from_128x128x2_Alg1((data+128), (uint8_t*)data) : Transpose_to_128x128x2_Alg1((uint8_t*)data, (data+128));
		//real_ortho_128x128x2_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


float Transpose_256x64_Alg1_benchmark()
{
	//printf("Transpose_256x64_Alg1_benchmark\n");
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
		Transpose_to_256x64_Alg1((uint8_t*)data, (data+64));
		Transpose_from_256x64_Alg1((data+64), (uint8_t*)data);
		//(i&1) ? real_ortho_from_64x64x4_modify((data+64), (uint8_t*)data) : real_ortho_to_64x64x4_modify((uint8_t*)data, (data+64));
		//real_ortho_128x128x2_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}
