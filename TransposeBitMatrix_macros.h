#ifndef __TRANSPOSE_BIT_MATRIX_MACROS__
#define __TRANSPOSE_BIT_MATRIX_MACROS__


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


#endif //__TRANSPOSE_BIT_MATRIX_MACROS__
