#ifndef __MEMORY_ALIGN___
#define __MEMORY_ALIGN___


#include <stdint.h>
#include <stddef.h> // для определения size_t


#if defined _MSC_VER
#define _ALIGN(x) __declspec(align(x))
#else
#define _ALIGN(x) __attribute__ ((__aligned__(x)))
#endif


void *aligned_malloc(size_t size, uint32_t alignByte);
void aligned_free(void *ptr);


#endif // __MEMORY_ALIGN___
