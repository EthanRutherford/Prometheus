#pragma once

// this file contains SIMD definitions for operations where a wide register is
// used to store a single 3D vector, rather than SOA data for proper "wide" SIMD.
// For such uses, there's no benefit to using AVX2 8-wide registers; we're only using 3 lanes.
// For that reason, we unconditionally use SSE, even if AVX2 is available.

#include "core.h"

#include <emmintrin.h>

// wide float holds 4 numbers
typedef __m128 b3V32;

typedef union b3128
{
	b3V32 v;
	float f[4];
} b3128;

#if defined( _MSC_VER ) && !defined( __clang__ )

static const b3V32 b3_zeroV = { { 0.0f, 0.0f, 0.0f, 0.0f } };
static const b3V32 b3_halfV = { { 0.5f, 0.5f, 0.5f, 0.5f } };
static const b3V32 b3_oneV = { { 1.0f, 1.0f, 1.0f, 1.0f } };

#else

static const b3V32 b3_zeroV = { 0.0f, 0.0f, 0.0f, 0.0f };
static const b3V32 b3_halfV = { 0.5f, 0.5f, 0.5f, 0.5f };
static const b3V32 b3_oneV = { 1.0f, 1.0f, 1.0f, 1.0f };

#endif

static inline b3V32 b3AddV( b3V32 a, b3V32 b )
{
	return _mm_add_ps( a, b );
}

static inline b3V32 b3SubV( b3V32 a, b3V32 b )
{
	return _mm_sub_ps( a, b );
}

static inline b3V32 b3MulV( b3V32 a, b3V32 b )
{
	return _mm_mul_ps( a, b );
}

static inline b3V32 b3DivV( b3V32 a, b3V32 b )
{
	return _mm_div_ps( a, b );
}

static inline b3V32 b3NegV( b3V32 a )
{
	return _mm_sub_ps( _mm_setzero_ps(), a );
}

static inline b3V32 b3LoadV( const float* src )
{
	// Loads exactly 12 bytes: 8 via movsd, 4 via movss.
	// Result lane 3 is implicitly zero from the partial loads.
	__m128 xy = _mm_castpd_ps( _mm_load_sd( (const double*)( src ) ) );
	__m128 z = _mm_load_ss( src + 2 );
	return _mm_movelh_ps( xy, z ); // { src[0], src[1], src[2], 0.0f }
}

static inline b3V32 b3ZeroV( void )
{
	return _mm_setzero_ps();
}

static inline float b3GetXV( b3V32 a )
{
	return _mm_cvtss_f32( a );
}

static inline float b3GetYV( b3V32 a )
{
	return _mm_cvtss_f32( _mm_shuffle_ps( a, a, _MM_SHUFFLE( 1, 1, 1, 1 ) ) );
}

static inline float b3GetZV( b3V32 a )
{
	return _mm_cvtss_f32( _mm_shuffle_ps( a, a, _MM_SHUFFLE( 2, 2, 2, 2 ) ) );
}

static inline float b3GetV( b3V32 a, int index )
{
	b3128 b;
	b.v = a;
	return b.f[index];
}

static inline b3V32 b3SplatV( float x )
{
	return _mm_set_ps1( x );
}

static inline b3V32 b3AbsV( b3V32 a )
{
	// Abs( V ) = Max( -V, V )
	b3V32 zero = _mm_setzero_ps();
	return _mm_max_ps( _mm_sub_ps( zero, a ), a );
}

static inline b3V32 b3MinV( b3V32 a, b3V32 b )
{
	return _mm_min_ps( a, b );
}

static inline b3V32 b3MaxV( b3V32 a, b3V32 b )
{
	return _mm_max_ps( a, b );
}

static inline b3V32 b3CrossV( b3V32 a, b3V32 b )
{
	b3V32 yzX1 = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 3, 0, 2, 1 ) );
	b3V32 zxY1 = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 3, 1, 0, 2 ) );
	b3V32 yzX2 = _mm_shuffle_ps( b, b, _MM_SHUFFLE( 3, 0, 2, 1 ) );
	b3V32 zxY2 = _mm_shuffle_ps( b, b, _MM_SHUFFLE( 3, 1, 0, 2 ) );

	return _mm_sub_ps( _mm_mul_ps( yzX1, zxY2 ), _mm_mul_ps( zxY1, yzX2 ) );
}

static inline b3V32 b3ModifiedCrossV( b3V32 a, b3V32 b )
{
	b3V32 yzX1 = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 3, 0, 2, 1 ) );
	b3V32 zxY1 = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 3, 1, 0, 2 ) );
	b3V32 yzX2 = _mm_shuffle_ps( b, b, _MM_SHUFFLE( 3, 0, 2, 1 ) );
	b3V32 zxY2 = _mm_shuffle_ps( b, b, _MM_SHUFFLE( 3, 1, 0, 2 ) );

	return _mm_add_ps( _mm_mul_ps( yzX1, zxY2 ), _mm_mul_ps( zxY1, yzX2 ) );
}

static inline bool b3AnyLess3V( b3V32 a, b3V32 b )
{
	b3V32 v = _mm_cmplt_ps( a, b );
	return ( _mm_movemask_ps( v ) & 0x07 ) != 0;
}

static inline bool b3AnyLessEq3V( b3V32 a, b3V32 b )
{
	b3V32 v = _mm_cmple_ps( a, b );
	return ( _mm_movemask_ps( v ) & 0x07 ) != 0;
}

static inline bool b3AnyGreater3V( b3V32 a, b3V32 b )
{
	b3V32 v = _mm_cmpgt_ps( a, b );
	return ( _mm_movemask_ps( v ) & 0x07 ) != 0;
}

static inline bool b3AllLessEq3V( b3V32 a, b3V32 b )
{
	b3V32 v = _mm_cmple_ps( a, b );
	return ( _mm_movemask_ps( v ) & 0x07 ) == 0x07;
}


typedef struct b3AABBV
{
	__m128 lower;
	__m128 upper;
} b3AABBV;

B3_FORCE_INLINE b3AABBV b3LoadAABBV( const b3AABB* aabb )
{
	const float* base = &aabb->lowerBound.x;
	// Same offset as Neon to avoid UB.
	__m128 v1 = _mm_loadu_ps( base + 2 );
	b3AABBV result;
	result.lower = _mm_loadu_ps( base );
	result.upper = _mm_shuffle_ps( v1, v1, _MM_SHUFFLE( 3, 3, 2, 1 ) );
	return result;
}

B3_FORCE_INLINE bool b3OverlapAABBV( b3AABBV a, b3AABBV b )
{
	__m128 test = _mm_and_ps( _mm_cmple_ps( a.lower, b.upper ), _mm_cmple_ps( b.lower, a.upper ) );
	return ( _mm_movemask_ps( test ) & 0x7 ) == 0x7;
}

B3_FORCE_INLINE bool b3OverlapNode( b3AABBV av, const b3TreeNode* node )
{
	return b3OverlapAABBV( av, b3LoadAABBV( &node->aabb ) );
}

B3_FORCE_INLINE bool b3OverlapV( const b3AABB* a, const b3AABB* b )
{
	return b3OverlapAABBV( b3LoadAABBV( a ), b3LoadAABBV( b ) );
}

B3_FORCE_INLINE b3AABBV b3UnionAABBV( b3AABBV a, b3AABBV b )
{
	b3AABBV result;
	result.lower = _mm_min_ps( a.lower, b.lower );
	result.upper = _mm_max_ps( a.upper, b.upper );
	return result;
}

B3_FORCE_INLINE b3AABBV b3UnionPairV( const b3TreeNode* pair )
{
	return b3UnionAABBV( b3LoadAABBV( &pair[0].aabb ), b3LoadAABBV( &pair[1].aabb ) );
}

// Conditionally store an AABB. Avoids a branch.
B3_FORCE_INLINE void b3StoreAABBV( b3AABB* aabb, b3AABBV value, bool condition )
{
	__m128 lane3 = _mm_castsi128_ps( _mm_set_epi32( -1, 0, 0, 0 ) );
	__m128 uxSplat = _mm_shuffle_ps( value.upper, value.upper, _MM_SHUFFLE( 0, 0, 0, 0 ) );
	__m128 raw0 = _mm_or_ps( _mm_andnot_ps( lane3, value.lower ), _mm_and_ps( lane3, uxSplat ) );
	__m128 raw1 = _mm_shuffle_ps( raw0, value.upper, _MM_SHUFFLE( 2, 1, 3, 2 ) );

	float* base = &aabb->lowerBound.x;
	__m128 mask = _mm_castsi128_ps( _mm_set1_epi32( condition ? -1 : 0 ) );
	__m128 old0 = _mm_loadu_ps( base );
	__m128 old1 = _mm_loadu_ps( base + 2 );
	_mm_storeu_ps( base, _mm_or_ps( _mm_and_ps( mask, raw0 ), _mm_andnot_ps( mask, old0 ) ) );
	_mm_storeu_ps( base + 2, _mm_or_ps( _mm_and_ps( mask, raw1 ), _mm_andnot_ps( mask, old1 ) ) );
}
