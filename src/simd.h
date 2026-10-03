// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "core.h"

#include "box3d/types.h"

#include <stdbool.h>

#if defined( B3_SIMD_ENABLED )

#if defined( B3_SIMD_NEON )
#include "simd_neon.h"
#endif

#if defined( B3_SIMD_SSE2 )
#include "simd_sse2.h"
#include "simd_v32_sse2.h"
#endif

#if defined( B3_SIMD_AVX2 )
#include "simd_avx2.h"
#include "simd_v32_sse2.h"
#endif

#else

#include "simd_scalar.h"

#endif

int b3GetSIMDWidth( void );
void b3SetSIMDWidth( int width );

// Wide vec3
typedef struct b3Vec3W4
{
	b3FloatW4 X, Y, Z;
} b3Vec3W4;

static inline b3Vec3W4 b3LoadVW4( const float* x, const float* y, const float* z )
{
	return (b3Vec3W4){ b3LoadW4( x ), b3LoadW4( y ), b3LoadW4( z ) };
}

static inline void b3StoreVW4( float* x, float* y, float* z, b3Vec3W4 v )
{
	b3StoreW4( x, v.X );
	b3StoreW4( y, v.Y );
	b3StoreW4( z, v.Z );
}

static inline b3Vec3W4 b3SplatVW4( b3Vec3 v )
{
	return (b3Vec3W4){ b3SplatW4( v.x ), b3SplatW4( v.y ), b3SplatW4( v.z ) };
}

static inline b3Vec3W4 b3NegVW4( b3Vec3W4 a )
{
	return (b3Vec3W4){ b3NegW4( a.X ), b3NegW4( a.Y ), b3NegW4( a.Z ) };
}

// s * a
static inline b3Vec3W4 b3MulSVW4( b3FloatW4 s, b3Vec3W4 a )
{
	return (b3Vec3W4){ b3MulW4( s, a.X ), b3MulW4( s, a.Y ), b3MulW4( s, a.Z ) };
}

// a - s * b
static inline b3Vec3W4 b3MulSubSVW4( b3Vec3W4 a, b3FloatW4 s, b3Vec3W4 b )
{
	return (b3Vec3W4){ b3SubW4( a.X, b3MulW4( s, b.X ) ), b3SubW4( a.Y, b3MulW4( s, b.Y ) ), b3SubW4( a.Z, b3MulW4( s, b.Z ) ) };
}

// a + s * b
static inline b3Vec3W4 b3MulAddSVW4( b3Vec3W4 a, b3FloatW4 s, b3Vec3W4 b )
{
	return (b3Vec3W4){ b3AddW4( a.X, b3MulW4( s, b.X ) ), b3AddW4( a.Y, b3MulW4( s, b.Y ) ), b3AddW4( a.Z, b3MulW4( s, b.Z ) ) };
}

// a - b
static inline b3Vec3W4 b3SubVW4( b3Vec3W4 a, b3Vec3W4 b )
{
	return (b3Vec3W4){
		b3SubW4( a.X, b.X ),
		b3SubW4( a.Y, b.Y ),
		b3SubW4( a.Z, b.Z ),
	};
}

// a + b
static inline b3Vec3W4 b3AddVW4( b3Vec3W4 a, b3Vec3W4 b )
{
	return (b3Vec3W4){
		b3AddW4( a.X, b.X ),
		b3AddW4( a.Y, b.Y ),
		b3AddW4( a.Z, b.Z ),
	};
}

static inline b3FloatW4 b3DotW4( b3Vec3W4 a, b3Vec3W4 b )
{
	return b3AddW4( b3AddW4( b3MulW4( a.X, b.X ), b3MulW4( a.Y, b.Y ) ), b3MulW4( a.Z, b.Z ) );
}

static inline b3Vec3W4 b3CrossW4( b3Vec3W4 a, b3Vec3W4 b )
{
	b3Vec3W4 c;
	c.X = b3SubW4( b3MulW4( a.Y, b.Z ), b3MulW4( a.Z, b.Y ) );
	c.Y = b3SubW4( b3MulW4( a.Z, b.X ), b3MulW4( a.X, b.Z ) );
	c.Z = b3SubW4( b3MulW4( a.X, b.Y ), b3MulW4( a.Y, b.X ) );
	return c;
}

#if defined( B3_SIMD_AVX2 )

// Wide vec3
typedef struct b3Vec3W8
{
	b3FloatW8 X, Y, Z;
} b3Vec3W8;

static inline b3Vec3W8 b3LoadVW8( const float* x, const float* y, const float* z )
{
	return (b3Vec3W8){ b3LoadW8( x ), b3LoadW8( y ), b3LoadW8( z ) };
}

static inline void b3StoreVW8( float* x, float* y, float* z, b3Vec3W8 v )
{
	b3StoreW8( x, v.X );
	b3StoreW8( y, v.Y );
	b3StoreW8( z, v.Z );
}

static inline b3Vec3W8 b3SplatVW8( b3Vec3 v )
{
	return (b3Vec3W8){ b3SplatW8( v.x ), b3SplatW8( v.y ), b3SplatW8( v.z ) };
}

static inline b3Vec3W8 b3NegVW8( b3Vec3W8 a )
{
	return (b3Vec3W8){ b3NegW8( a.X ), b3NegW8( a.Y ), b3NegW8( a.Z ) };
}

// s * a
static inline b3Vec3W8 b3MulSVW8( b3FloatW8 s, b3Vec3W8 a )
{
	return (b3Vec3W8){ b3MulW8( s, a.X ), b3MulW8( s, a.Y ), b3MulW8( s, a.Z ) };
}

// a - s * b
static inline b3Vec3W8 b3MulSubSVW8( b3Vec3W8 a, b3FloatW8 s, b3Vec3W8 b )
{
	return (b3Vec3W8){ b3SubW8( a.X, b3MulW8( s, b.X ) ), b3SubW8( a.Y, b3MulW8( s, b.Y ) ), b3SubW8( a.Z, b3MulW8( s, b.Z ) ) };
}

// a + s * b
static inline b3Vec3W8 b3MulAddSVW8( b3Vec3W8 a, b3FloatW8 s, b3Vec3W8 b )
{
	return (b3Vec3W8){ b3AddW8( a.X, b3MulW8( s, b.X ) ), b3AddW8( a.Y, b3MulW8( s, b.Y ) ), b3AddW8( a.Z, b3MulW8( s, b.Z ) ) };
}

// a - b
static inline b3Vec3W8 b3SubVW8( b3Vec3W8 a, b3Vec3W8 b )
{
	return (b3Vec3W8){
		b3SubW8( a.X, b.X ),
		b3SubW8( a.Y, b.Y ),
		b3SubW8( a.Z, b.Z ),
	};
}

// a + b
static inline b3Vec3W8 b3AddVW8( b3Vec3W8 a, b3Vec3W8 b )
{
	return (b3Vec3W8){
		b3AddW8( a.X, b.X ),
		b3AddW8( a.Y, b.Y ),
		b3AddW8( a.Z, b.Z ),
	};
}

static inline b3FloatW8 b3DotW8( b3Vec3W8 a, b3Vec3W8 b )
{
	return b3AddW8( b3AddW8( b3MulW8( a.X, b.X ), b3MulW8( a.Y, b.Y ) ), b3MulW8( a.Z, b.Z ) );
}

static inline b3Vec3W8 b3CrossW8( b3Vec3W8 a, b3Vec3W8 b )
{
	b3Vec3W8 c;
	c.X = b3SubW8( b3MulW8( a.Y, b.Z ), b3MulW8( a.Z, b.Y ) );
	c.Y = b3SubW8( b3MulW8( a.Z, b.X ), b3MulW8( a.X, b.Z ) );
	c.Z = b3SubW8( b3MulW8( a.X, b.Y ), b3MulW8( a.Y, b.X ) );
	return c;
}

#endif

static inline bool b3TestBoundsOverlap( b3V32 nodeMin1, b3V32 nodeMax1, b3V32 nodeMin2, b3V32 nodeMax2 )
{
	b3V32 separation = b3MaxV( b3SubV( nodeMin2, nodeMax1 ), b3SubV( nodeMin1, nodeMax2 ) );
	return b3AllLessEq3V( separation, b3_zeroV );
}

// Test a ray for edge separation with an AABB (Gino, p80).
static inline bool b3TestBoundsRayOverlap( b3V32 nodeMin, b3V32 nodeMax, b3V32 rayStart, b3V32 rayDelta )
{
	// Setup node
	b3V32 nodeCenter = b3MulV( b3_halfV, b3AddV( nodeMin, nodeMax ) );
	b3V32 nodeExtent = b3SubV( nodeMax, nodeCenter );

	// Setup ray
	rayStart = b3SubV( rayStart, nodeCenter );

	// SAT: Edge separation
	b3V32 edgeSeparation = b3SubV( b3AbsV( b3CrossV( rayDelta, rayStart ) ), b3ModifiedCrossV( b3AbsV( rayDelta ), nodeExtent ) );
	return b3AllLessEq3V( edgeSeparation, b3_zeroV );
}

bool b3TestBoundsTriangleOverlap( b3V32 nodeCenter, b3V32 nodeExtent, b3V32 vertex1, b3V32 vertex2, b3V32 vertex3 );
float b3IntersectRayTriangle( b3V32 rayStart, b3V32 rayDelta, b3V32 vertex1, b3V32 vertex2, b3V32 vertex3 );

B3_FORCE_INLINE b3AABB b3UnionV( b3AABB a, b3AABB b )
{
	b3AABB result;
	b3StoreAABBV( &result, b3UnionAABBV( b3LoadAABBV( &a ), b3LoadAABBV( &b ) ), true );
	return result;
}
