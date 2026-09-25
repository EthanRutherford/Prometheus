// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "core.h"

#include "box3d/types.h"

#include <stdbool.h>

#if defined( B3_SIMD_ENABLED )

#if defined( B3_SIMD_NEON )
#include "simd/neon.h"
#endif

#if defined( B3_SIMD_SSE2 )
#include "simd/sse.h"
#include "simd/x86_common.h"
#endif

#if defined( B3_SIMD_AVX2 )
#include "simd/avx2.h"
#include "simd/x86_common.h"
#endif

#if defined( B3_SIMD_DYNAMIC_DISPATCH )
#include "simd/feature_detection.h"
#endif

#else

#include "simd/none.h"

#endif

static inline int b3_GetSIMDWidth()
{
#if defined( B3_SIMD_DYNAMIC_DISPATCH )
	return b3_supportsW8() ? 8 : 4;
#elif defined( B3_SIMD_HAS_WIDTH_8 )
	return 8;
#elif defined( B3_SIMD_HAS_WIDTH_4 )
	return 4;
#else
	return 1;
#endif
}

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
