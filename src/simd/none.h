#pragma once

#include "core.h"

#include "box3d/types.h"

#include <math.h>
#include <string.h>

// scalar math
typedef struct b3FloatW4
{
	float x, y, z, w;
} b3FloatW4;

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
	return (b3FloatW4){ 0.0f, 0.0f, 0.0f, 0.0f };
}

static inline b3FloatW4 b3SplatW4( float scalar )
{
	return (b3FloatW4){ scalar, scalar, scalar, scalar };
}

static inline b3FloatW4 b3SetW4( float a, float b, float c, float d )
{
	return (b3FloatW4){ a, b, c, d };
}

static inline b3FloatW4 b3LoadW4( const float* data )
{
	return (b3FloatW4){ data[0], data[1], data[2], data[3] };
}

static inline void b3StoreW4( float* data, b3FloatW4 a )
{
	data[0] = a.x;
	data[1] = a.y;
	data[2] = a.z;
	data[3] = a.w;
}

static inline b3FloatW4 b3UnpackLoW4( b3FloatW4 a, b3FloatW4 b )
{
	return (b3FloatW4){ a.x, b.x, a.y, b.y };
}

static inline b3FloatW4 b3UnpackHiW4( b3FloatW4 a, b3FloatW4 b )
{
	return (b3FloatW4){ a.z, b.z, a.w, b.w };
}

static inline b3FloatW4 b3NegW4( b3FloatW4 a )
{
	return (b3FloatW4){ -a.x, -a.y, -a.z, -a.w };
}

static inline b3FloatW4 b3AbsW4( b3FloatW4 a )
{
	return (b3FloatW4){ a.x < 0.0f ? -a.x : a.x, a.y < 0.0f ? -a.y : a.y, a.z < 0.0f ? -a.z : a.z, a.w < 0.0f ? -a.w : a.w };
}

static inline b3FloatW4 b3AddW4( b3FloatW4 a, b3FloatW4 b )
{
	return (b3FloatW4){ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

static inline b3FloatW4 b3SubW4( b3FloatW4 a, b3FloatW4 b )
{
	return (b3FloatW4){ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

static inline b3FloatW4 b3MulW4( b3FloatW4 a, b3FloatW4 b )
{
	return (b3FloatW4){ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

static inline b3FloatW4 b3DivW4( b3FloatW4 a, b3FloatW4 b )
{
	return (b3FloatW4){ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

static inline b3FloatW4 b3SqrtW4( b3FloatW4 a )
{
	return (b3FloatW4){ sqrtf( a.x ), sqrtf( a.y ), sqrtf( a.z ), sqrtf( a.w ) };
}

static inline b3FloatW4 b3MulAddW4( b3FloatW4 a, b3FloatW4 b, b3FloatW4 c )
{
	return (b3FloatW4){ a.x + b.x * c.x, a.y + b.y * c.y, a.z + b.z * c.z, a.w + b.w * c.w };
}

static inline b3FloatW4 b3MinW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x <= b.x ? a.x : b.x;
	r.y = a.y <= b.y ? a.y : b.y;
	r.z = a.z <= b.z ? a.z : b.z;
	r.w = a.w <= b.w ? a.w : b.w;
	return r;
}

static inline b3FloatW4 b3MaxW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x >= b.x ? a.x : b.x;
	r.y = a.y >= b.y ? a.y : b.y;
	r.z = a.z >= b.z ? a.z : b.z;
	r.w = a.w >= b.w ? a.w : b.w;
	return r;
}

// clamp a to [-b, b]
static inline b3FloatW4 b3SymClampW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x <= b.x ? a.x : b.x;
	r.y = a.y <= b.y ? a.y : b.y;
	r.z = a.z <= b.z ? a.z : b.z;
	r.w = a.w <= b.w ? a.w : b.w;
	r.x = r.x <= -b.x ? -b.x : r.x;
	r.y = r.y <= -b.y ? -b.y : r.y;
	r.z = r.z <= -b.z ? -b.z : r.z;
	r.w = r.w <= -b.w ? -b.w : r.w;
	return r;
}

// Logical operations on the scalar path are 0/1 float values. Not bit-wise like SIMD.

static inline b3FloatW4 b3AndW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x != 0.0f && b.x != 0.0f ? 1.0f : 0.0f;
	r.y = a.y != 0.0f && b.y != 0.0f ? 1.0f : 0.0f;
	r.z = a.z != 0.0f && b.z != 0.0f ? 1.0f : 0.0f;
	r.w = a.w != 0.0f && b.w != 0.0f ? 1.0f : 0.0f;
	return r;
}

static inline b3FloatW4 b3OrW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x != 0.0f || b.x != 0.0f ? 1.0f : 0.0f;
	r.y = a.y != 0.0f || b.y != 0.0f ? 1.0f : 0.0f;
	r.z = a.z != 0.0f || b.z != 0.0f ? 1.0f : 0.0f;
	r.w = a.w != 0.0f || b.w != 0.0f ? 1.0f : 0.0f;
	return r;
}

// a & ~b
static inline b3FloatW4 b3AndNotW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x != 0.0f && b.x == 0.0f ? 1.0f : 0.0f;
	r.y = a.y != 0.0f && b.y == 0.0f ? 1.0f : 0.0f;
	r.z = a.z != 0.0f && b.z == 0.0f ? 1.0f : 0.0f;
	r.w = a.w != 0.0f && b.w == 0.0f ? 1.0f : 0.0f;
	return r;
}

static inline b3FloatW4 b3SoftMaskW4( const int* indexA, const int* indexB )
{
	b3FloatW4 r;
	r.x = indexA[0] == 0 || indexB[0] == 0 ? 1.0f : 0.0f;
	r.y = indexA[1] == 0 || indexB[1] == 0 ? 1.0f : 0.0f;
	r.z = indexA[2] == 0 || indexB[2] == 0 ? 1.0f : 0.0f;
	r.w = indexA[3] == 0 || indexB[3] == 0 ? 1.0f : 0.0f;
	return r;
}

static inline b3FloatW4 b3GreaterThanW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x > b.x ? 1.0f : 0.0f;
	r.y = a.y > b.y ? 1.0f : 0.0f;
	r.z = a.z > b.z ? 1.0f : 0.0f;
	r.w = a.w > b.w ? 1.0f : 0.0f;
	return r;
}

static inline b3FloatW4 b3LessThanW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x < b.x ? 1.0f : 0.0f;
	r.y = a.y < b.y ? 1.0f : 0.0f;
	r.z = a.z < b.z ? 1.0f : 0.0f;
	r.w = a.w < b.w ? 1.0f : 0.0f;
	return r;
}

static inline b3FloatW4 b3EqualsW4( b3FloatW4 a, b3FloatW4 b )
{
	b3FloatW4 r;
	r.x = a.x == b.x ? 1.0f : 0.0f;
	r.y = a.y == b.y ? 1.0f : 0.0f;
	r.z = a.z == b.z ? 1.0f : 0.0f;
	r.w = a.w == b.w ? 1.0f : 0.0f;
	return r;
}

static inline bool b3AllZeroW4( b3FloatW4 a )
{
	return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f && a.w == 0.0f;
}

static inline bool b3AnyTrueW4( b3FloatW4 mask )
{
	return mask.x != 0.0f || mask.y != 0.0f || mask.z != 0.0f || mask.w != 0.0f;
}

// component-wise returns mask ? b : a
static inline b3FloatW4 b3BlendW4( b3FloatW4 a, b3FloatW4 b, b3FloatW4 mask )
{
	b3FloatW4 r;
	r.x = mask.x != 0.0f ? b.x : a.x;
	r.y = mask.y != 0.0f ? b.y : a.y;
	r.z = mask.z != 0.0f ? b.z : a.z;
	r.w = mask.w != 0.0f ? b.w : a.w;
	return r;
}

static inline b3FloatW4 b3Dot3W4( b3FloatW4 ax, b3FloatW4 ay, b3FloatW4 az, b3FloatW4 bx,
										  b3FloatW4 by, b3FloatW4 bz )
{
	b3FloatW4 r;
	r.x = ax.x * bx.x + ( ay.x * by.x + az.x * bz.x );
	r.y = ax.y * bx.y + ( ay.y * by.y + az.y * bz.y );
	r.z = ax.z * bx.z + ( ay.z * by.z + az.z * bz.z );
	r.w = ax.w * bx.w + ( ay.w * by.w + az.w * bz.w );
	return r;
}

static inline b3FloatW4 b3EmbedIndexW4( b3FloatW4 value, int baseIndex, int bitCount )
{
	uint32_t mask = ( 1u << bitCount ) - 1;
	float lanes[4] = { value.x, value.y, value.z, value.w };

	for ( int i = 0; i < 4; ++i )
	{
		uint32_t bits;
		memcpy( &bits, lanes + i, sizeof( bits ) );
		bits = ( bits & ~mask ) | (uint32_t)( baseIndex + i );
		memcpy( lanes + i, &bits, sizeof( bits ) );
	}

	return (b3FloatW4){ lanes[0], lanes[1], lanes[2], lanes[3] };
}

static inline int b3MinIndexW4( b3FloatW4 a, int bitCount )
{
	float m = a.x;
	m = a.y < m ? a.y : m;
	m = a.z < m ? a.z : m;
	m = a.w < m ? a.w : m;

	uint32_t bits;
	memcpy( &bits, &m, sizeof( bits ) );
	return (int)( bits & ( ( 1u << bitCount ) - 1 ) );
}

typedef b3AABB b3AABBV4;

B3_FORCE_INLINE b3AABBV4 b3LoadAABBV4( const b3AABB* aabb )
{
	return *aabb;
}

B3_FORCE_INLINE bool b3OverlapAABBV4( b3AABBV4 a, b3AABBV4 b )
{
	return a.lowerBound.x <= b.upperBound.x && a.lowerBound.y <= b.upperBound.y && a.lowerBound.z <= b.upperBound.z &&
		   b.lowerBound.x <= a.upperBound.x && b.lowerBound.y <= a.upperBound.y && b.lowerBound.z <= a.upperBound.z;
}

B3_FORCE_INLINE bool b3OverlapNode4( b3AABBV4 av, const b3TreeNode* node )
{
	return b3OverlapAABBV4( av, node->aabb );
}

B3_FORCE_INLINE bool b3OverlapV4( const b3AABB* a, const b3AABB* b )
{
	return b3OverlapAABBV( *a, *b );
}

B3_FORCE_INLINE b3AABBV4 b3UnionAABBV4( b3AABBV4 a, b3AABBV4 b )
{
	return b3AABB_Union( a, b );
}

B3_FORCE_INLINE b3AABBV4 b3UnionPairV4( const b3TreeNode* pair )
{
	return b3AABB_Union( pair[0].aabb, pair[1].aabb );
}

B3_FORCE_INLINE void b3StoreAABBV4( b3AABB* aabb, b3AABBV4 value, bool condition )
{
	if ( condition )
	{
		*aabb = value;
	}
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
