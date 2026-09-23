// Transpose binary matrix, examples
// Author: Anatoly Solovyev, soloviov-anatoly@mail.ru


#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <immintrin.h>
#include <malloc.h>
#include <time.h>
#include "Settings.h"
#include "TransposeBitMatrix_macros.h"
#include "TransposeBitMatrix_Alg1.h"
#include "Transpose_8x8_movemask.h"
//#include "Transpose_Usuba.h"
//#include "MemoryAlign.h"
//#include "Compare_test.h"


//#include "TransposeBitMatrix_512x512.h"
//#include "TransposeBitMatrix_512x512_movemask.h"


int main()
{
	printf("instructions %s\n", __GATHER__==0?"set":"gather");
//	Transpose8x8_example1();
//	TransposeMovemask_example2();

//	compare_test_256x256();
//	compare_test_256x128();
//	compare_test_256x64();

//	compare_test_512x512();
//	Transpose_512x512_Alg1_benchmark();
//	Transpose_512x512_8x8_benchmark();
//	Transpose_512x512_Movemask_benchmark();

	float time_256x256_Movemask = 0;
	float time_256x256_8x8 = 0;
	float time_256x256_Alg1 = 0;
	float time_256x128_Alg1 = 0;
	float time_256x128_8x8 = 0;
	float time_256x128_Movemask = 0;
	float time_256x64_Movemask = 0;
	float time_256x64_8x8 = 0;
	float time_256x64_Alg1 = 0;
//	float time_256x256_Usuba = 0;
//	float time_256x64_Usuba = 0;

	time_256x128_Alg1 = Transpose_256x128_Alg1_benchmark();
	time_256x128_8x8 = Transpose_256x128_8x8_benchmark();
	time_256x128_Movemask = Transpose_256x128_Movemask_benchmark();

	time_256x64_Movemask = Transpose_256x64_Movemask_benchmark();
	time_256x64_8x8 = Transpose_256x64_8x8_benchmark();
	time_256x64_Alg1 = Transpose_256x64_Alg1_benchmark();

	time_256x256_Movemask = Transpose_256x256_Movemask_benchmark();
	time_256x256_8x8 = Transpose_256x256_8x8_benchmark();
	time_256x256_Alg1 = Transpose_256x256_Alg1_benchmark();

//	time_256x256_Usuba = Transpose_256x256_Usuba_benchmark();
//	time_256x64_Usuba = Transpose_256x64_Usuba_benchmark();

//	printf("Transpose_256x256_Usuba_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x256_Usuba);
	printf("Transpose_256x256_Alg1_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x256_Alg1);
	printf("Transpose_256x256_8x8_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x256_8x8);
	printf("Transpose_256x256_Movemask_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x256_Movemask);

	printf("Transpose_256x128_Alg1_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x128_Alg1);
	printf("Transpose_256x128_8x8_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x128_8x8);
	printf("Transpose_256x128_Movemask_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x128_Movemask);

//	printf("Transpose_256x64_Usuba_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x64_Usuba);
	printf("Transpose_256x64_Alg1_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x64_Alg1);
	printf("Transpose_256x64_8x8_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x64_8x8);
	printf("Transpose_256x64_Movemask_benchmark\nTime elapsed variant timespec_get() %f\n", time_256x64_Movemask);

//	printf("instructions %s\n", __GATHER__==0?"set":"gather");

	return 0;
}
