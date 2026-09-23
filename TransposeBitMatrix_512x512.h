

#ifndef __TRANSPOSE_512x512__
#define __TRANSPOSE_512x512__


#include <immintrin.h>


void real_ortho_512x512_modify(__m256i data[]);
//int compare_test_512x512();
int TransposeUsuba_512x512_benchmark();


#endif
