

#include "TransposeBitMatrix_512x512.h"
#include "MemoryAlign.h"
#include <stdio.h>
#include <stdint.h>
#include <time.h>


#if defined _MSC_VER
#define _ALIGN(x) __declspec(align(x))
#else
#define _ALIGN(x) __attribute__ ((__aligned__(x)))
#endif


void real_ortho_512x512_modify(__m256i data[]) {

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
		int n = (1UL << i); // êîëè÷åñòâî ñòðîê, ó÷àñòâóþùèõ â ñäâèãàõ/ïåðåñòàíîâêàõ
		for (int j = 0; j < 512; j += (2 * n))
			for (int k = 0; k < n; k++) {
				__m256i u[2] = { _mm256_and_si256(data[(j + k) * 2], mask_r[i]), _mm256_and_si256(data[(j + k) * 2 + 1], mask_r[i]) };
				__m256i v[2] = { _mm256_and_si256(data[(j + k) * 2], mask_l[i]), _mm256_and_si256(data[(j + k) * 2 + 1], mask_l[i]) };
				__m256i x[2] = { _mm256_and_si256(data[(j + n + k) * 2], mask_r[i]), _mm256_and_si256(data[(j + n + k) * 2 + 1], mask_r[i]) };
				__m256i y[2] = { _mm256_and_si256(data[(j + n + k) * 2], mask_l[i]), _mm256_and_si256(data[(j + n + k) * 2 + 1], mask_l[i]) };
				if (i <= 5) {
					data[(j + k) * 2] = _mm256_or_si256(u[0], _mm256_slli_epi64(x[0], n)); data[(j + k) * 2 + 1] = _mm256_or_si256(u[1], _mm256_slli_epi64(x[1], n));
					data[(j + n + k) * 2] = _mm256_or_si256(_mm256_srli_epi64(v[0], n), y[0]); data[(j + n + k) * 2 + 1] = _mm256_or_si256(_mm256_srli_epi64(v[1], n), y[1]);
				}
				else if (i == 6) {
					/* Note the "inversion" of srli and slli. */
					data[(j + k) * 2] = _mm256_or_si256(u[0], _mm256_slli_si256(x[0], 8)); data[(j + k) * 2 + 1] = _mm256_or_si256(u[1], _mm256_slli_si256(x[1], 8));
					data[(j + n + k) * 2] = _mm256_or_si256(_mm256_srli_si256(v[0], 8), y[0]); data[(j + n + k) * 2 + 1] = _mm256_or_si256(_mm256_srli_si256(v[1], 8), y[1]);
				}
				else {
					data[(j + k) * 2] = _mm256_or_si256(u[0], _mm256_permute2x128_si256(x[0], x[0], 1)); data[(j + k) * 2 + 1] = _mm256_or_si256(u[1], _mm256_permute2x128_si256(x[1], x[1], 1));
					data[(j + n + k) * 2] = _mm256_or_si256(_mm256_permute2x128_si256(v[0], v[0], 1), y[0]); data[(j + n + k) * 2 + 1] = _mm256_or_si256(_mm256_permute2x128_si256(v[1], v[1], 1), y[1]);
				}
			}
	}
	for (int j = 0; j < 256; ++j)
	{
		__m256i t = data[j * 2 + 1];
		data[j * 2 + 1] = data[j * 2 + 512];
		data[j * 2 + 512] = t;
	}
}


#define ITER_ (1<<20)


int TransposeUsuba_512x512_benchmark()
{
	printf("TransposeUsuba_512x512_benchmark\n");
	__m256i* data = aligned_malloc(sizeof(__m256i) * 1024, 32);
	if (data == NULL)
	{
		printf("Error _aligned_malloc\n");
		return -1;
	}
	//_ALIGN(32) __m256i data[256];
	for (uint32_t i = 0; i < 1024; ++i)
	{
		data[i] = _mm256_set1_epi32((uint8_t)i);
	}

	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		real_ortho_512x512_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	aligned_free(data);
	return 0;
}
