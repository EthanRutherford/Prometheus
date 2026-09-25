#pragma once

#include "core.h"

#include "box3d/types.h"

#include <emmintrin.h>

// wide float holds 4 numbers
typedef __m128 b3FloatW4;

static inline b3FloatW4 b3ZeroW4( void )
{
	return _mm_setzero_ps();
}

static inline b3FloatW4 b3SplatW4( float scalar )
{
	return _mm_set1_ps( scalar );
}

static inline b3FloatW4 b3SetW4( float a, float b, float c, float d )
{
	return _mm_setr_ps( a, b, c, d );
}

static inline b3FloatW4 b3LoadW4( const float* data )
{
	return _mm_loadu_ps( data );
}

static inline void b3StoreW4( float* data, b3FloatW4 a )
{
	_mm_storeu_ps( data, a );
}

static inline b3FloatW4 b3UnpackLoW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_unpacklo_ps( a, b );
}

static inline b3FloatW4 b3UnpackHiW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_unpackhi_ps( a, b );
}

static inline b3FloatW4 b3NegW4( b3FloatW4 a )
{
	// Create a mask with the sign bit set for each element
	__m128 mask = _mm_set1_ps( -0.0f );

	// XOR the input with the mask to negate each element
	return _mm_xor_ps( a, mask );
}

static inline b3FloatW4 b3AbsW4( b3FloatW4 a )
{
	return _mm_andnot_ps( _mm_set1_ps( -0.0f ), a );
}

static inline b3FloatW4 b3AddW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_add_ps( a, b );
}

static inline b3FloatW4 b3SubW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_sub_ps( a, b );
}

static inline b3FloatW4 b3MulW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_mul_ps( a, b );
}

static inline b3FloatW4 b3DivW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_div_ps( a, b );
}

static inline b3FloatW4 b3SqrtW4( b3FloatW4 a )
{
	return _mm_sqrt_ps( a );
}

// a + b * c
static inline b3FloatW4 b3MulAddW4( b3FloatW4 a, b3FloatW4 b, b3FloatW4 c )
{
	return _mm_add_ps( a, _mm_mul_ps( b, c ) );
}

static inline b3FloatW4 b3MinW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_min_ps( a, b );
}

static inline b3FloatW4 b3MaxW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_max_ps( a, b );
}

// Horizontal min over a 4-lane vector (result broadcast to all lanes).
static inline b3FloatW4 b3HorizontalMinW4( b3FloatW4 v )
{
	v = _mm_min_ps( v, _mm_shuffle_ps( v, v, _MM_SHUFFLE( 2, 3, 0, 1 ) ) );
	return _mm_min_ps( v, _mm_shuffle_ps( v, v, _MM_SHUFFLE( 1, 0, 3, 2 ) ) );
}

// clamp a to [-b, b]
static inline b3FloatW4 b3SymClampW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 nb = b3NegW4( b );
	b3FloatW4 c = b3MaxW4( nb, a );
	return b3MinW4( c, b );
}

static inline b3FloatW4 b3AndW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_and_ps( a, b );
}

static inline b3FloatW4 b3OrW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_or_ps( a, b );
}

// a & ~b
static inline b3FloatW4 b3AndNotW4( b3FloatW4 a, b3FloatW4 b )
{
	// Arguments are reversed
	return _mm_andnot_ps( b, a );
}

// This is used to optimize selection of contact softness.
static inline b3FloatW4 b3SoftMaskW4( const int* indexA, const int* indexB )
{
	__m128i zero = _mm_setzero_si128();
	__m128i a = _mm_cmpeq_epi32( _mm_loadu_si128( (const __m128i*)indexA ), zero );
	__m128i b = _mm_cmpeq_epi32( _mm_loadu_si128( (const __m128i*)indexB ), zero );
	return _mm_castsi128_ps( _mm_or_si128( a, b ) );
}

static inline b3FloatW4 b3GreaterThanW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_cmpgt_ps( a, b );
}

static inline b3FloatW4 b3LessThanW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_cmplt_ps( a, b );
}

static inline b3FloatW4 b3EqualsW4( b3FloatW4 a, b3FloatW4 b )
{
	return _mm_cmpeq_ps( a, b );
}

static inline bool b3AllZeroW4( b3FloatW4 a )
{
	// Compare each element with zero
	b3FloatW4 zero = _mm_setzero_ps();
	b3FloatW4 cmp = _mm_cmpeq_ps( a, zero );

	// Create a mask from the comparison results
	int mask = _mm_movemask_ps( cmp );

	// If all elements are zero, the mask will be 0xF (1111 in binary)
	return mask == 0xF;
}

static inline bool b3AnyTrueW4( b3FloatW4 mask )
{
	return _mm_movemask_ps( mask ) != 0;
}

// component-wise returns mask ? b : a
static inline b3FloatW4 b3BlendW4( b3FloatW4 a, b3FloatW4 b, b3FloatW4 mask )
{
	return _mm_or_ps( _mm_and_ps( mask, b ), _mm_andnot_ps( mask, a ) );
}

static inline b3FloatW4 b3Dot3W4( b3FloatW4 ax, b3FloatW4 ay, b3FloatW4 az, b3FloatW4 bx, b3FloatW4 by,
										b3FloatW4 bz )
{
	return _mm_add_ps( _mm_mul_ps( ax, bx ), _mm_add_ps( _mm_mul_ps( ay, by ), _mm_mul_ps( az, bz ) ) );
}

// Replace the low bitCount mantissa bits of each lane with baseIndex + lane. The value must be
// positive so the embedded index sorts with the value, and ties fall to the lower index.
static inline b3FloatW4 b3EmbedIndexW4( b3FloatW4 value, int baseIndex, int bitCount )
{
	int mask = ( 1 << bitCount ) - 1;
	__m128i index = _mm_add_epi32( _mm_set1_epi32( baseIndex ), _mm_setr_epi32( 0, 1, 2, 3 ) );
	__m128 clearLow = _mm_castsi128_ps( _mm_set1_epi32( ~mask ) );
	return _mm_or_ps( _mm_and_ps( value, clearLow ), _mm_castsi128_ps( index ) );
}

// Recovers the index embedded by b3EmbedIndexW4 from the lane holding the minimum.
static inline int b3MinIndexW4( b3FloatW4 a, int bitCount )
{
	a = _mm_min_ps( a, _mm_shuffle_ps( a, a, _MM_SHUFFLE( 2, 3, 0, 1 ) ) );
	a = _mm_min_ps( a, _mm_shuffle_ps( a, a, _MM_SHUFFLE( 1, 0, 3, 2 ) ) );
	return _mm_cvtsi128_si32( _mm_castps_si128( a ) ) & ( ( 1 << bitCount ) - 1 );
}

B3_FORCE_INLINE void b3TransposeW4( b3FloatW4 r0, b3FloatW4 r1, b3FloatW4 r2, b3FloatW4 r3, b3FloatW4* c0,
									   b3FloatW4* c1, b3FloatW4* c2, b3FloatW4* c3 )
{
	b3FloatW4 t0 = b3UnpackLoW4( r0, r2 );
	b3FloatW4 t1 = b3UnpackLoW4( r1, r3 );
	b3FloatW4 t2 = b3UnpackHiW4( r0, r2 );
	b3FloatW4 t3 = b3UnpackHiW4( r1, r3 );
	*c0 = b3UnpackLoW4( t0, t1 );
	*c1 = b3UnpackHiW4( t0, t1 );
	*c2 = b3UnpackLoW4( t2, t3 );
	*c3 = b3UnpackHiW4( t2, t3 );
}
