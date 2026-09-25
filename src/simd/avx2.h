#pragma once

#include "core.h"

#include "box3d/types.h"

#include <immintrin.h>

// wide float holds 8 numbers
typedef __m256 b3FloatW8;

static inline b3FloatW8 b3ZeroW8( void )
{
	return _mm256_setzero_ps();
}

static inline b3FloatW8 b3SplatW8( float scalar )
{
	return _mm256_set1_ps( scalar );
}

static inline b3FloatW8 b3SetW8( float a, float b, float c, float d, float e, float f, float g, float h )
{
	return _mm256_setr_ps( a, b, c, d, e, f, g, h );
}

static inline b3FloatW8 b3LoadW8( const float* data )
{
	return _mm256_loadu_ps( data );
}

static inline void b3StoreW8( float* data, b3FloatW8 a )
{
	_mm256_storeu_ps( data, a );
}

static inline b3FloatW8 b3UnpackLoW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_unpacklo_ps( a, b );
}

static inline b3FloatW8 b3UnpackHiW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_unpackhi_ps( a, b );
}

static inline b3FloatW8 b3NegW8( b3FloatW8 a )
{
	// Create a mask with the sign bit set for each element
	__m256 mask = _mm256_set1_ps( -0.0f );

	// XOR the input with the mask to negate each element
	return _mm256_xor_ps( a, mask );
}

static inline b3FloatW8 b3AbsW8( b3FloatW8 a )
{
	__m256 mask = _mm256_set1_ps( -0.0f );
	return _mm256_andnot_ps( mask, a );
}

static inline b3FloatW8 b3AddW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_add_ps( a, b );
}

static inline b3FloatW8 b3SubW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_sub_ps( a, b );
}

static inline b3FloatW8 b3MulW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_mul_ps( a, b );
}

static inline b3FloatW8 b3DivW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_div_ps( a, b );
}

static inline b3FloatW8 b3SqrtW8( b3FloatW8 a )
{
	return _mm256_sqrt_ps( a );
}

// a + b * c
static inline b3FloatW8 b3MulAddW8( b3FloatW8 a, b3FloatW8 b, b3FloatW8 c )
{
	return _mm256_add_ps( a, _mm256_mul_ps( b, c ) );
}

static inline b3FloatW8 b3MinW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_min_ps( a, b );
}

static inline b3FloatW8 b3MaxW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_max_ps( a, b );
}

// Horizontal min over an 8-lane vector (result broadcast to all lanes).
static inline b3FloatW8 b3HorizontalMinW8( b3FloatW8 v )
{
	v = _mm256_min_ps( v, _mm256_permute_ps( v, _MM_SHUFFLE( 2, 3, 0, 1 ) ) );
	return _mm256_min_ps( v, _mm256_permute_ps( v, _MM_SHUFFLE( 1, 0, 3, 2 ) ) );
}

// clamp a to [-b, b]
static inline b3FloatW8 b3SymClampW8( b3FloatW8 a, b3FloatW8 b )
{
	b3FloatW8 nb = b3NegW8( b );
	b3FloatW8 c = b3MaxW8( nb, a );
	return b3MinW8( c, b );
}

static inline b3FloatW8 b3AndW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_and_ps( a, b );
}

static inline b3FloatW8 b3OrW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_or_ps( a, b );
}

// a & ~b
static inline b3FloatW8 b3AndNotW8( b3FloatW8 a, b3FloatW8 b )
{
	// Arguments are reversed
	return _mm256_andnot_ps( b, a );
}

// This is used to optimize selection of contact softness.
static inline b3FloatW8 b3SoftMaskW8( const int* indexA, const int* indexB )
{
	__m256i zero = _mm256_setzero_si256();
	__m256i a = _mm256_cmpeq_epi32( _mm256_loadu_si256( (const __m256i*)indexA ), zero );
	__m256i b = _mm256_cmpeq_epi32( _mm256_loadu_si256( (const __m256i*)indexB ), zero );
	return _mm256_castsi256_ps( _mm256_or_si256( a, b ) );
}

static inline b3FloatW8 b3GreaterThanW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_cmp_ps( a, b, _CMP_GT_OQ );
}

static inline b3FloatW8 b3LessThanW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_cmp_ps( a, b, _CMP_LT_OQ );
}

static inline b3FloatW8 b3EqualsW8( b3FloatW8 a, b3FloatW8 b )
{
	return _mm256_cmp_ps( a, b, _CMP_EQ_OQ );
}

static inline bool b3AllZeroW8( b3FloatW8 a )
{
	// Compare each element with zero
	b3FloatW8 zero = _mm256_setzero_ps();
	b3FloatW8 cmp = _mm256_cmp_ps( a, zero, _CMP_EQ_OQ );

	// Create a mask from the comparison results
	int mask = _mm256_movemask_ps( cmp );

	// If all elements are zero, the mask will be 0xF (1111 in binary)
	return mask == 0xF;
}

static inline bool b3AnyTrueW8( b3FloatW8 mask )
{
	return _mm256_movemask_ps( mask ) != 0;
}

// component-wise returns mask ? b : a
static inline b3FloatW8 b3BlendW8( b3FloatW8 a, b3FloatW8 b, b3FloatW8 mask )
{
	return _mm256_or_ps( _mm256_and_ps( mask, b ), _mm256_andnot_ps( mask, a ) );
}

static inline b3FloatW8 b3Dot3W8( b3FloatW8 ax, b3FloatW8 ay, b3FloatW8 az, b3FloatW8 bx, b3FloatW8 by, b3FloatW8 bz )
{
	return _mm256_add_ps( _mm256_mul_ps( ax, bx ), _mm256_add_ps( _mm256_mul_ps( ay, by ), _mm256_mul_ps( az, bz ) ) );
}

// Replace the low bitCount mantissa bits of each lane with baseIndex + lane. The value must be
// positive so the embedded index sorts with the value, and ties fall to the lower index.
static inline b3FloatW8 b3EmbedIndexW8( b3FloatW8 value, int baseIndex, int bitCount )
{
	int mask = ( 1 << bitCount ) - 1;
	__m256i index = _mm256_add_epi32( _mm256_set1_epi32( baseIndex ), _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 ) );
	__m256 clearLow = _mm256_castsi256_ps( _mm256_set1_epi32( ~mask ) );
	return _mm256_or_ps( _mm256_and_ps( value, clearLow ), _mm256_castsi256_ps( index ) );
}

// Recovers the index embedded by b3EmbedIndexW8 from the lane holding the minimum.
static inline int b3MinIndexW8( b3FloatW8 a, int bitCount )
{
	a = _mm256_min_ps( a, _mm256_permute_ps( a, _MM_SHUFFLE( 2, 3, 0, 1 ) ) );
	a = _mm256_min_ps( a, _mm256_permute_ps( a, _MM_SHUFFLE( 1, 0, 3, 2 ) ) );
	return _mm256_cvtsi256_si32( _mm256_castps_si256( a ) ) & ( ( 1 << bitCount ) - 1 );
}

B3_FORCE_INLINE void b3TransposeW8( b3FloatW8 r0, b3FloatW8 r1, b3FloatW8 r2, b3FloatW8 r3, b3FloatW8 r4, b3FloatW8 r5, b3FloatW8 r6, b3FloatW8 r7, b3FloatW8* c0,
									   b3FloatW8* c1, b3FloatW8* c2, b3FloatW8* c3, b3FloatW8* c4, b3FloatW8* c5, b3FloatW8* c6, b3FloatW8* c7 )
{
	// I don't think the following code is correct.

	b3FloatW8 t0 = b3UnpackLoW8( r0, r4 );
	b3FloatW8 t1 = b3UnpackLoW8( r1, r5 );
	b3FloatW8 t2 = b3UnpackLoW8( r2, r6 );
	b3FloatW8 t3 = b3UnpackLoW8( r3, r7 );
	b3FloatW8 t4 = b3UnpackHiW8( r0, r4 );
	b3FloatW8 t5 = b3UnpackHiW8( r1, r5 );
	b3FloatW8 t6 = b3UnpackHiW8( r2, r6 );
	b3FloatW8 t7 = b3UnpackHiW8( r3, r7 );

	*c0 = b3UnpackLoW8( t0, t1 );
	*c1 = b3UnpackHiW8( t0, t1 );
	*c2 = b3UnpackLoW8( t2, t3 );
	*c3 = b3UnpackHiW8( t2, t3 );
	*c4 = b3UnpackLoW8( t4, t5 );
	*c5 = b3UnpackHiW8( t4, t5 );
	*c6 = b3UnpackLoW8( t6, t7 );
	*c7 = b3UnpackHiW8( t6, t7 );
}
