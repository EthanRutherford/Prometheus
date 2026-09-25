#pragma once

#include "core.h"

#include "box3d/types.h"

#include <arm_neon.h>

// wide float holds 4 numbers
typedef float32x4_t b3FloatW4;

// I don't expect the use case of b3V32 to benefit from Neon code.
// In particular the cross product is very complex in Neon.

// scalar math
typedef struct b3V32
{
	float x, y, z;
} b3V32;

typedef union b3128
{
	b3V32 v;
	float f[3];
} b3128;

static const b3V32 b3_zeroV = { 0.0f, 0.0f, 0.0f };
static const b3V32 b3_halfV = { 0.5f, 0.5f, 0.5f };
static const b3V32 b3_oneV = { 1.0f, 1.0f, 1.0f };

static inline b3V32 b3AddV( b3V32 a, b3V32 b )
{
	return B3_LITERAL( b3V32 ){
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
	};
}

static inline b3V32 b3SubV( b3V32 a, b3V32 b )
{
	return B3_LITERAL( b3V32 ){
		a.x - b.x,
		a.y - b.y,
		a.z - b.z,
	};
}

static inline b3V32 b3MulV( b3V32 a, b3V32 b )
{
	return B3_LITERAL( b3V32 ){
		a.x * b.x,
		a.y * b.y,
		a.z * b.z,
	};
}

static inline b3V32 b3DivV( b3V32 a, b3V32 b )
{
	return B3_LITERAL( b3V32 ){
		a.x / b.x,
		a.y / b.y,
		a.z / b.z,
	};
}

static inline b3V32 b3NegV( b3V32 a )
{
	return B3_LITERAL( b3V32 ){
		-a.x,
		-a.y,
		-a.z,
	};
}

// Unaligned loads are much faster on recent hardware with little to no penalty
static inline b3V32 b3LoadV( const float* src )
{
	return B3_LITERAL( b3V32 ){ src[0], src[1], src[2] };
}

static inline b3V32 b3ZeroV( void )
{
	return B3_LITERAL( b3V32 ){ 0.0f, 0.0f, 0.0f };
}

static inline float b3GetXV( b3V32 a )
{
	return a.x;
}

static inline float b3GetYV( b3V32 a )
{
	return a.y;
}

static inline float b3GetZV( b3V32 a )
{
	return a.z;
}

static inline float b3GetV( b3V32 a, int index )
{
	b3128 b;
	b.v = a;
	return b.f[index];
}

static inline b3V32 b3SplatV( float x )
{
	return B3_LITERAL( b3V32 ){ x, x, x };
}

static inline b3V32 b3AbsV( b3V32 a )
{
	return B3_LITERAL( b3V32 ){
		a.x < 0.0f ? -a.x : a.x,
		a.y < 0.0f ? -a.y : a.y,
		a.z < 0.0f ? -a.z : a.z,
	};
}

static inline b3V32 b3MinV( b3V32 a, b3V32 b )
{
	return B3_LITERAL( b3V32 ){
		a.x < b.x ? a.x : b.x,
		a.y < b.y ? a.y : b.y,
		a.z < b.z ? a.z : b.z,
	};
}

static inline b3V32 b3MaxV( b3V32 a, b3V32 b )
{
	return B3_LITERAL( b3V32 ){
		a.x > b.x ? a.x : b.x,
		a.y > b.y ? a.y : b.y,
		a.z > b.z ? a.z : b.z,
	};
}

static inline b3V32 b3CrossV( b3V32 a, b3V32 b )
{
	b3V32 c;
	c.x = a.y * b.z - a.z * b.y;
	c.y = a.z * b.x - a.x * b.z;
	c.z = a.x * b.y - a.y * b.x;
	return c;
}

static inline b3V32 b3ModifiedCrossV( b3V32 a, b3V32 b )
{
	b3V32 c;
	c.x = a.y * b.z + a.z * b.y;
	c.y = a.z * b.x + a.x * b.z;
	c.z = a.x * b.y + a.y * b.x;
	return c;
}

static inline bool b3AnyLess3V( b3V32 a, b3V32 b )
{
	return a.x < b.x || a.y < b.y || a.z < b.z;
}

static inline bool b3AnyLessEq3V( b3V32 a, b3V32 b )
{
	return a.x <= b.x || a.y <= b.y || a.z <= b.z;
}

static inline bool b3AnyGreater3V( b3V32 a, b3V32 b )
{
	return a.x > b.x || a.y > b.y || a.z > b.z;
}

static inline bool b3AllLessEq3V( b3V32 a, b3V32 b )
{
	return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

static inline b3FloatW4 b3ZeroW4( void )
{
	return vdupq_n_f32( 0.0f );
}

static inline b3FloatW4 b3SplatW4( float scalar )
{
	return vdupq_n_f32( scalar );
}

static inline b3FloatW4 b3SetW4( float a, float b, float c, float d )
{
	float32_t array[4] = { a, b, c, d };
	return vld1q_f32( array );
}

static inline b3FloatW4 b3LoadW4( const float* data )
{
	return vld1q_f32( data );
}

static inline void b3StoreW4( float* data, b3FloatW4 a )
{
	vst1q_f32( data, a );
}

static inline b3FloatW4 b3UnpackLoW4( b3FloatW4 a, b3FloatW4 b )
{
	return vzip1q_f32( a, b );
}

static inline b3FloatW4 b3UnpackHiW4( b3FloatW4 a, b3FloatW4 b )
{
	return vzip2q_f32( a, b );
}

static inline b3FloatW4 b3NegW4( b3FloatW4 a )
{
	return vnegq_f32( a );
}

static inline b3FloatW4 b3AbsW4( b3FloatW4 a )
{
	return vabsq_f32( a );
}

static inline b3FloatW4 b3AddW4( b3FloatW4 a, b3FloatW4 b )
{
	return vaddq_f32( a, b );
}

static inline b3FloatW4 b3SubW4( b3FloatW4 a, b3FloatW4 b )
{
	return vsubq_f32( a, b );
}

static inline b3FloatW4 b3MulW4( b3FloatW4 a, b3FloatW4 b )
{
	return vmulq_f32( a, b );
}

static inline b3FloatW4 b3DivW4( b3FloatW4 a, b3FloatW4 b )
{
	return vdivq_f32( a, b );
}

static inline b3FloatW4 b3SqrtW4( b3FloatW4 a )
{
	return vsqrtq_f32( a );
}

// Cannot use real FMA because it doesn't match the non-SIMD path
static inline b3FloatW4 b3MulAddW4( b3FloatW4 a, b3FloatW4 b, b3FloatW4 c )
{
	return vaddq_f32( a, vmulq_f32( b, c ) );
}

static inline b3FloatW4 b3MinW4( b3FloatW4 a, b3FloatW4 b )
{
	return vminq_f32( a, b );
}

static inline b3FloatW4 b3MaxW4( b3FloatW4 a, b3FloatW4 b )
{
	return vmaxq_f32( a, b );
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
	return vreinterpretq_f32_u32( vandq_u32( vreinterpretq_u32_f32( a ), vreinterpretq_u32_f32( b ) ) );
}

static inline b3FloatW4 b3OrW4( b3FloatW4 a, b3FloatW4 b )
{
	return vreinterpretq_f32_u32( vorrq_u32( vreinterpretq_u32_f32( a ), vreinterpretq_u32_f32( b ) ) );
}

// a & ~b
static inline b3FloatW4 b3AndNotW4( b3FloatW4 a, b3FloatW4 b )
{
	return vreinterpretq_f32_u32( vbicq_u32( vreinterpretq_u32_f32( a ), vreinterpretq_u32_f32( b ) ) );
}

static inline b3FloatW4 b3SoftMaskW4( const int* indexA, const int* indexB )
{
	int32x4_t zero = vdupq_n_s32( 0 );
	uint32x4_t a = vceqq_s32( vld1q_s32( (const int32_t*)indexA ), zero );
	uint32x4_t b = vceqq_s32( vld1q_s32( (const int32_t*)indexB ), zero );
	return vreinterpretq_f32_u32( vorrq_u32( a, b ) );
}

static inline b3FloatW4 b3GreaterThanW4( b3FloatW4 a, b3FloatW4 b )
{
	return vreinterpretq_f32_u32( vcgtq_f32( a, b ) );
}

static inline b3FloatW4 b3LessThanW4( b3FloatW4 a, b3FloatW4 b )
{
	return vreinterpretq_f32_u32( vcltq_f32( a, b ) );
}

static inline b3FloatW4 b3EqualsW4( b3FloatW4 a, b3FloatW4 b )
{
	return vreinterpretq_f32_u32( vceqq_f32( a, b ) );
}

static inline bool b3AllZeroW4( b3FloatW4 a )
{
	// Create a zero vector for comparison
	b3FloatW4 zero = vdupq_n_f32( 0.0f );

	// Compare the input vector with zero
	uint32x4_t cmp_result = vceqq_f32( a, zero );

// Check if all comparison results are non-zero using vminvq
#ifdef __ARM_FEATURE_SVE
	// ARM v8.2+ has horizontal minimum instruction
	return vminvq_u32( cmp_result ) != 0;
#else
	// For older ARM architectures, we need to manually check all lanes
	return vgetq_lane_u32( cmp_result, 0 ) != 0 && vgetq_lane_u32( cmp_result, 1 ) != 0 && vgetq_lane_u32( cmp_result, 2 ) != 0 &&
		   vgetq_lane_u32( cmp_result, 3 ) != 0;
#endif
}

// _mm_movemask_ps equivalent, compatible with ARM v7.
static inline bool b3AnyTrueW4( b3FloatW4 mask )
{
	uint32x4_t m = vreinterpretq_u32_f32( mask );
	uint32x2_t p = vorr_u32( vget_low_u32( m ), vget_high_u32( m ) );
	return ( vget_lane_u32( p, 0 ) | vget_lane_u32( p, 1 ) ) != 0;
}

// component-wise returns mask ? b : a
static inline b3FloatW4 b3BlendW4( b3FloatW4 a, b3FloatW4 b, b3FloatW4 mask )
{
	uint32x4_t mask32 = vreinterpretq_u32_f32( mask );
	return vbslq_f32( mask32, b, a );
}

static inline b3FloatW4 b3Dot3W4( b3FloatW4 ax, b3FloatW4 ay, b3FloatW4 az, b3FloatW4 bx,
										  b3FloatW4 by, b3FloatW4 bz )
{
	return vaddq_f32( vmulq_f32( ax, bx ), vaddq_f32( vmulq_f32( ay, by ), vmulq_f32( az, bz ) ) );
}

static inline b3FloatW4 b3EmbedIndexW4( b3FloatW4 value, int baseIndex, int bitCount )
{
	uint32_t mask = ( 1u << bitCount ) - 1;
	const int32_t lanes[4] = { 0, 1, 2, 3 };
	int32x4_t index = vaddq_s32( vdupq_n_s32( baseIndex ), vld1q_s32( lanes ) );
	uint32x4_t clearLow = vdupq_n_u32( ~mask );
	uint32x4_t bits = vorrq_u32( vandq_u32( vreinterpretq_u32_f32( value ), clearLow ), vreinterpretq_u32_s32( index ) );
	return vreinterpretq_f32_u32( bits );
}

static inline int b3MinIndexW4( b3FloatW4 a, int bitCount )
{
	float32x2_t m = vmin_f32( vget_low_f32( a ), vget_high_f32( a ) );
	m = vpmin_f32( m, m );
	uint32_t bits = vget_lane_u32( vreinterpret_u32_f32( m ), 0 );
	return (int)( bits & ( ( 1u << bitCount ) - 1 ) );
}

typedef struct b3AABBV4
{
	float32x4_t lower;
	float32x4_t upper;
} b3AABBV4;

B3_FORCE_INLINE b3AABBV4 b3LoadAABBV4( const b3AABB* aabb )
{
	const float* base = &aabb->lowerBound.x;

	// Offset to avoid reading off the end (avoid UB).
	// [lz ux uy uz]
	float32x4_t v1 = vld1q_f32( base + 2 );
	b3AABBV4 result;
	// [lx ly lz -]
	result.lower = vld1q_f32( base );
	// [ux uy uz -]
	result.upper = vextq_f32( v1, v1, 1 );
	return result;
}

B3_FORCE_INLINE bool b3OverlapAABBV4( b3AABBV4 a, b3AABBV4 b )
{
	static const uint32_t laneMask[4] = { 0, 0, 0, 0xFFFFFFFFu };
	uint32x4_t test = vandq_u32( vcleq_f32( a.lower, b.upper ), vcleq_f32( b.lower, a.upper ) );
	return vminvq_u32( vorrq_u32( test, vld1q_u32( laneMask ) ) ) != 0;
}

B3_FORCE_INLINE bool b3OverlapNode4( b3AABBV4 av, const b3TreeNode* node )
{
	return b3OverlapAABBV4( av, b3LoadAABBV4( &node->aabb ) );
}

B3_FORCE_INLINE bool b3OverlapV4( const b3AABB* a, const b3AABB* b )
{
	return b3OverlapAABBV4( b3LoadAABBV4( a ), b3LoadAABBV4( b ) );
}

B3_FORCE_INLINE b3AABBV4 b3UnionAABBV4( b3AABBV4 a, b3AABBV4 b )
{
	b3AABBV4 result;
	result.lower = vminq_f32( a.lower, b.lower );
	result.upper = vmaxq_f32( a.upper, b.upper );
	return result;
}

B3_FORCE_INLINE b3AABBV4 b3UnionPairV4( const b3TreeNode* pair )
{
	return b3UnionAABBV4( b3LoadAABBV4( &pair[0].aabb ), b3LoadAABBV4( &pair[1].aabb ) );
}

B3_FORCE_INLINE void b3StoreAABBV4( b3AABB* aabb, b3AABBV4 value, bool condition )
{
	float32x4_t raw0 = vsetq_lane_f32( vgetq_lane_f32( value.upper, 0 ), value.lower, 3 );
	float32x4_t rotated = vextq_f32( value.upper, value.upper, 1 );
	float32x4_t raw1 = vcombine_f32( vget_high_f32( raw0 ), vget_low_f32( rotated ) );

	float* base = &aabb->lowerBound.x;
	uint32x4_t mask = vdupq_n_u32( condition ? 0xFFFFFFFFu : 0u );
	vst1q_f32( base, vbslq_f32( mask, raw0, vld1q_f32( base ) ) );
	vst1q_f32( base + 2, vbslq_f32( mask, raw1, vld1q_f32( base + 2 ) ) );
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
