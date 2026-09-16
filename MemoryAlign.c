

#include "MemoryAlign.h"


#ifdef _MSC_VER // для Visual Studio

#include <malloc.h>

void* aligned_malloc(size_t size, uint32_t alignByte) {
	return _aligned_malloc(size, alignByte);
}
void aligned_free(void* ptr) {
	_aligned_free(ptr);
}

#elif defined __GNUC__ //!!! например, для gcc, хотя уточни, там должно быть свое выравнивание

#include <stdlib.h>

void* aligned_malloc(size_t size, uint32_t alignByte)
{
	return aligned_alloc(alignByte, size);
}
void aligned_free(void* ptr) {
	free(ptr);
}

#else

void* aligned_malloc(size_t size, uint32_t alignByte)
{
	char* mem = malloc(size + alignByte + sizeof(mem));
	if (!mem) return 0;
	char** ptr = (char**)((uintptr_t)(mem + alignByte + sizeof(mem)) & ~(alignByte - 1));
	ptr[-1] = mem;
	return ptr;
}
void aligned_free(void* ptr) {
	free(((char**)ptr)[-1]);
}

#endif
