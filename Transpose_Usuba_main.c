#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <immintrin.h>
#include <malloc.h>
#include <time.h>


#define ITER_ (1<<20)
#define ITER_HALF (ITER_>>1)


//#ifdef __TRANSPOSE_256x64_USUBA__
/* Orthogonalization stuffs */
static uint64_t mask_l[6] = {
	0xaaaaaaaaaaaaaaaaUL,
	0xccccccccccccccccUL,
	0xf0f0f0f0f0f0f0f0UL,
	0xff00ff00ff00ff00UL,
	0xffff0000ffff0000UL,
	0xffffffff00000000UL
};

static uint64_t mask_r[6] = {
	0x5555555555555555UL,
	0x3333333333333333UL,
	0x0f0f0f0f0f0f0f0fUL,
	0x00ff00ff00ff00ffUL,
	0x0000ffff0000ffffUL,
	0x00000000ffffffffUL
};


void real_ortho(uint64_t data[]) {
  for (int i = 0; i < 6; i ++) {
    int n = (1UL << i);
    for (int j = 0; j < 64; j += (2 * n))
      for (int k = 0; k < n; k ++) {
        uint64_t u = data[j + k] & mask_l[i];
        uint64_t v = data[j + k] & mask_r[i];
        uint64_t x = data[j + n + k] & mask_l[i];
        uint64_t y = data[j + n + k] & mask_r[i];
        data[j + k] = u | (x >> n);
        data[j + n + k] = (v << n) | y;
      }
  }
}

void orthogonalize_256x64(uint64_t* data, __m256i* out) {
  real_ortho(data);
  real_ortho(&(data[64]));
  real_ortho(&(data[128]));
  real_ortho(&(data[192]));
  for (int i = 0; i < 64; i++)
    out[i] = _mm256_set_epi64x(data[i], data[64+i], data[128+i], data[192+i]);
}

void unorthogonalize_64x256(__m256i *in, uint64_t* data) {
  for (int i = 0; i < 64; i++) {
    uint64_t tmp[4];
    _mm256_store_si256 ((__m256i*)tmp, in[i]);
    data[i] = tmp[3];
    data[64+i] = tmp[2];
    data[128+i] = tmp[1];
    data[192+i] = tmp[0];
  }
  real_ortho(data);
  real_ortho(&(data[64]));
  real_ortho(&(data[128]));
  real_ortho(&(data[192]));
}
//#endif


//__attribute__((noinline))
void real_ortho_256x256(__m256i data[]) {

  __m256i mask_l[8] = {
    _mm256_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
    _mm256_set1_epi64x(0xccccccccccccccccUL),
    _mm256_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
    _mm256_set1_epi64x(0xff00ff00ff00ff00UL),
    _mm256_set1_epi64x(0xffff0000ffff0000UL),
    _mm256_set1_epi64x(0xffffffff00000000UL),
    _mm256_set_epi64x(0UL,0xFFFFFFFFFFFFFFFFUL,0UL,0xFFFFFFFFFFFFFFFFUL),
    _mm256_set_epi64x(0UL,0UL,0xFFFFFFFFFFFFFFFFUL,0xFFFFFFFFFFFFFFFFUL),

  };

  __m256i mask_r[8] = {
    _mm256_set1_epi64x(0x5555555555555555UL),
    _mm256_set1_epi64x(0x3333333333333333UL),
    _mm256_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
    _mm256_set1_epi64x(0x00ff00ff00ff00ffUL),
    _mm256_set1_epi64x(0x0000ffff0000ffffUL),
    _mm256_set1_epi64x(0x00000000ffffffffUL),
    _mm256_set_epi64x(0xFFFFFFFFFFFFFFFFUL,0UL,0xFFFFFFFFFFFFFFFFUL,0UL),
    _mm256_set_epi64x(0xFFFFFFFFFFFFFFFFUL,0xFFFFFFFFFFFFFFFFUL,0UL,0UL),
  };

  for (int i = 0; i < 8; i ++) {
    int n = (1UL << i);
    for (int j = 0; j < 256; j += (2 * n))
      for (int k = 0; k < n; k ++) {
        __m256i u = _mm256_and_si256(data[j + k], mask_l[i]);
        __m256i v = _mm256_and_si256(data[j + k], mask_r[i]);
        __m256i x = _mm256_and_si256(data[j + n + k], mask_l[i]);
        __m256i y = _mm256_and_si256(data[j + n + k], mask_r[i]);
        if (i <= 5) {
          data[j + k] = _mm256_or_si256(u, _mm256_srli_epi64(x, n));
          data[j + n + k] = _mm256_or_si256(_mm256_slli_epi64(v, n), y);
        } else if (i == 6) {
          /* Note the "inversion" of srli and slli. */
          data[j + k] = _mm256_or_si256(u, _mm256_slli_si256(x, 8));
          data[j + n + k] = _mm256_or_si256(_mm256_srli_si256(v, 8), y);
        } else {
          data[j + k] = _mm256_or_si256(u, _mm256_permute2x128_si256( x , x , 1));
          data[j + n + k] = _mm256_or_si256(_mm256_permute2x128_si256( v , v , 1), y);
        }
      }
  }
}


//;-funroll-loops;-fno-inline-functions-called-once
//__attribute__((noinline))
float Transpose_256x256_Usuba_benchmark()
{
	//printf("Transpose_256x256_Usuba_benchmark\n");
	__m256i data[256];
	for (uint32_t i=0;i<256;++i)
	{
		data[i] = _mm256_set1_epi8((uint8_t)i);
	}
	
	struct timespec startTime, endTime;
	timespec_get(&startTime, TIME_UTC);

	for (uint32_t i = 0; i < ITER_; ++i)
	{
		real_ortho_256x256(data); //real_ortho_256x256(data); //real_ortho_256x256_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


float Transpose_256x64_Usuba_benchmark()
{
	//printf("Transpose_256x64_Usuba_benchmark\n");
	__m256i data[64*2];
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
		orthogonalize_256x64((uint64_t*)data, (data+64));
		unorthogonalize_64x256((data+64), (uint64_t*)data);
		//(i&1) ? real_ortho_from_64x64x4_modify((data+64), (uint8_t*)data) : real_ortho_to_64x64x4_modify((uint8_t*)data, (data+64));
		//real_ortho_128x128x2_modify(data);
	}

	timespec_get(&endTime, TIME_UTC);
	double cpuTimeUsed = ((double)endTime.tv_sec - startTime.tv_sec) + ((double)endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

	//printf("Time elapsed variant timespec_get() %f\n", cpuTimeUsed);
	return (float)cpuTimeUsed;
}


int main()
{
	float t_256x64_Usuba = Transpose_256x64_Usuba_benchmark();
	float t_256x256_Usuba = Transpose_256x256_Usuba_benchmark();
	printf("Transpose_256x256_Usuba_benchmark\nTime elapsed variant timespec_get() %f\n", t_256x256_Usuba);
	printf("Transpose_256x64_Usuba_benchmark\nTime elapsed variant timespec_get() %f\n", t_256x64_Usuba);
//	printf("1\n");
	return 0;
}
