#include "algorithm.h"
#include "convex_manifold.h"
#include "shape.h"
#include "simd.h"

#if defined( B3_SIMD_HAS_WIDTH_4 )
#define B3_SIMD_WIDTH 4

// Transform a SoA point/normal stream (already split into X/Y/Z) by out = -(R*v (+t)).
// The inputs come straight from the hull's stored SoA arrays, so there's no transpose here.
static inline void b3NegativeTransformFromSoAW4( b3Matrix3 R, b3Vec3 p, const float* inX, const float* inY, const float* inZ,
												 int n, float* outX, float* outY, float* outZ, bool isPoint )
{
	// row-column
	b3Vec3W4 r0 = { b3SplatW4( R.cx.x ), b3SplatW4( R.cy.x ), b3SplatW4( R.cz.x ) };
	b3Vec3W4 r1 = { b3SplatW4( R.cx.y ), b3SplatW4( R.cy.y ), b3SplatW4( R.cz.y ) };
	b3Vec3W4 r2 = { b3SplatW4( R.cx.z ), b3SplatW4( R.cy.z ), b3SplatW4( R.cz.z ) };

	b3Vec3W4 t = { b3ZeroW4(), b3ZeroW4(), b3ZeroW4() };
	if ( isPoint )
	{
		t = b3SplatVW4( p );
	}

	for ( int i = 0; i < n; i += B3_SIMD_WIDTH )
	{
		b3Vec3W4 v = b3LoadVW4( inX + i, inY + i, inZ + i );

		// Rotate four vectors at a time
		b3Vec3W4 out = { b3DotW4( r0, v ), b3DotW4( r1, v ), b3DotW4( r2, v ) };

		if ( isPoint )
		{
			out = b3AddVW4( out, t );
		}

		b3StoreVW4( outX + i, outY + i, outZ + i, b3NegVW4( out ) );
	}
}

_Static_assert( B3_MAX_HULL_VERTICES == 128, "must be 128" );

#define B3_HULL_BIT_COUNT 7

// SIMD support point calculation using a SoA vertex array padded to a multiple of 8.
//
// This minimizes (bias - dot), where the caller is expected to provide a bias that makes this always positive.
// It can be direction dependent. The bias should be just big enough to ensure the value is positive because
// an excessive bias causes a precision loss in the support calculation.
//
// The vertex index is embedded in the low B3_HULL_BIT_COUNT mantissa bits of the value. By minimizing a value that
// is always positive, the minimum carries the smallest index so that padded SoA values will never win. This is the
// purpose of using the bias instead of maximizing the dot directly.
//
// The support is then recomputed exactly as dot(normal, vertex), without the embedded index.
// todo consider using this for GJK
static inline void b3GetSupportW4( b3Vec3 normal, const float* vx, const float* vy, const float* vz, int n, float bias,
								   float* support, int* vertexIndex )
{
	const b3Vec3W4 normalW = b3SplatVW4( normal );
	const b3FloatW4 biasV = b3SplatW4( bias );

	// Start the minimum at a large value.
	b3FloatW4 minValue = b3SplatW4( INFINITY );

	// Tail lanes hold vertex 0 with index bits >= vertexCount, so they never become the min value.
	for ( int i = 0; i < n; i += B3_SIMD_WIDTH )
	{
		b3Vec3W4 v = b3LoadVW4( vx + i, vy + i, vz + i );
		b3FloatW4 d = b3DotW4( normalW, v );

		// This is always positive.
		b3FloatW4 value = b3SubW4( biasV, d );
		b3FloatW4 augmentedValue = b3EmbedIndexW4( value, i, B3_HULL_BIT_COUNT );
		minValue = b3MinW4( minValue, augmentedValue );
	}

	// One horizontal min, the winning lane's value and index bits ride through.
	int vi = b3MinIndexW4( minValue, B3_HULL_BIT_COUNT );

	// Exact support for the chosen vertex.
	*vertexIndex = vi;

	// Dot product
	*support = normal.x * vx[vi] + normal.y * vy[vi] + normal.z * vz[vi];
}

static inline float b3GetFaceSeparationW4( b3Vec3 direction, float planeSeparation, const float* vx, const float* vy,
										   const float* vz, int n, b3Vec3 center, b3Vec3 extents, int* vertexIndex )
{
	float bias = b3Dot( direction, center ) + 1.0625f * b3Dot( b3Abs( direction ), extents );
	float support;
	b3GetSupportW4( direction, vx, vy, vz, n, bias, &support, vertexIndex );
	return planeSeparation - support;
}

// Wide dot(n, d) for all face normals n of the hull, padded to the SIMD width.
static inline void b3GetFaceDotsW4( const b3HullData* hull, b3Vec3 d, float* dots )
{
	int soaFaceCount = ( hull->faceCount + 7 ) & ~( 7 );
	const float* nx = b3GetHullSoaNormals( hull );
	const float* ny = nx + soaFaceCount;
	const float* nz = ny + soaFaceCount;

	b3Vec3W4 dW = b3SplatVW4( d );

	for ( int i = 0; i < soaFaceCount; i += B3_SIMD_WIDTH )
	{
		// dot product per lane
		b3FloatW4 m = b3DotW4( b3LoadVW4( nx + i, ny + i, nz + i ), dW );
		b3StoreW4( dots + i, m );
	}
}

#define B3_PARALLEL_TOL 1e-4f

// A hull edge is bounded by two face normals. On the Gauss map the edge becomes an arc between
// those two face normals. An edge can only build the best separating axis if a normal on that arc
// can beat the best separation value seen so far. This function determines if the upper bound
// separation on that arc can possibly beat the current maximum face separation.
//
// This follows the upper bound used for hull faces:
// separation_upper_bound = dot(axis, centerB - centerA) - innerRadiusA - innerRadiusB
//
// Inputs:
// d the vector connecting the hull centers
// a1 = dot(n1, d)
// a2 = dot(n2, d)
// c = dot(n1, n2)
// bound = maxFaceSeparation + radiusBound
//
// This returns 1 if the edges is a candidate and 0 otherwise.
static inline int b3TestEdgeCandidateW4( float a1, float a2, float c, float bound )
{
	// c = cos(theta), the angle between the normals.
	// s = sin(theta)^2 >= 0
	float s = 1.0f - c * c;

	// This is coincidentally the law of cosines. Break out your protractor.
	float t = a1 * a1 + a2 * a2 - 2.0f * a1 * a2 * c;

	// Exterior conditions:
	// Can either face normal beat the best separation? These cover the case where
	// d is outside the arc. Note that d pointing outside the arc does not cull this
	// edge. It can still generate the maximum separation (happens commonly).
	// Note: when earlyReturn == false, bound == -INFINITY and this is always true.
	int exterior = b3MaxFloat( a1, a2 ) >= bound;

	// Project d into the plane that holds both n1 and n2, call that vector dp.
	// Introduce the coordinates b1 and b2 (these have units of length).
	//
	// dp = b1*n1 + b2*n2
	//
	// Since dp is the projection of d into the plane formed by the cross(n1, n2):
	// dot(n1, dp) == dot(n1, d) == a1
	// dot(n2, dp) == dot(n2, d) == a2
	//
	// Dot the equation with n1 and n2:
	// a1 = b1 + b2*c
	// a2 = b1*c + b2
	//
	// Solve for b1 and b2 using Cramer's Rule
	// b1 = (a1 - a2 * c) / s
	// b2 = (a2 - a1 * c) / s
	//
	// s = 1 - c * c is positive, so b1 and b2 must be positive for d to live in
	// the arc between n1 and n2.
	//
	// The peak value along d is then norm(dp):
	// dot(dp, dp) = dot(b1*n1 + b2*n2, d)
	//             = b1*a1 + b2*a2
	//             = (a1*a1 - a1*a2*c + a2*a2 - a1*a2*c) / s
	//             = (a1*a1 + a2*a2 - 2*a1*a2*c) / s
	//             = t / s
	//
	// The interior is a candidate if:
	// norm(dp) > bound
	// sqrt(t / s) > bound
	// If bound < 0 this is always true. Otherwise
	// t > bound^2 * s
	//
	// Interior conditions:
	// b1 and b2 positive (interior arc): a1 >= c * a2 & a2 >= c * a1
	// bound <= 0.0: the interior arc is automatically a candidate because it is positive
	// s < B3_PARALLEL_TOL : n1 and n2 are nearly parallel so give up and pass the edge to the next stage
	// t >= bound * bound * s : bound is positive and the interior normal direction is a candidate
	//
	// Using bit ops here to avoid branches.

	int interior =
		( a1 >= c * a2 ) & ( a2 >= c * a1 ) & ( ( bound <= 0.0f ) | ( s < B3_PARALLEL_TOL ) | ( t >= bound * bound * s ) );

	return exterior | interior;
}

// Temporary abbreviations for convenience.
#define NE ( B3_MAX_HULL_EDGES + B3_SIMD_WIDTH )
#define NF ( B3_MAX_HULL_FACES + B3_SIMD_WIDTH )
#define NV ( B3_MAX_HULL_VERTICES + B3_SIMD_WIDTH )

// SIMD separating axis test based on an implementation developed by Cairn Overturf.
// See his article: https://cairno.substack.com/p/improvements-to-the-separating-axis
b3AxisQuery b3ComputeSeparatingAxisW4( const b3HullData* hullA, const b3HullData* hullB, b3Transform xfB, bool earlyReturn )
{
	b3Matrix3 R = b3MakeMatrixFromQuat( xfB.q );
	b3Matrix3 invR = b3Transpose( R );

	float speculativeDistance = B3_SPECULATIVE_DISTANCE;

	b3AxisQuery res = {
		.faceA =
			{
				.normal = b3Vec3_zero,
				.separation = -INFINITY,
				.indexA = B3_NULL_INDEX,
				.indexB = B3_NULL_INDEX,
				.type = b3_faceAxisA,
			},
		.faceB =
			{
				.normal = b3Vec3_zero,
				.separation = -INFINITY,
				.indexA = B3_NULL_INDEX,
				.indexB = B3_NULL_INDEX,
				.type = b3_faceAxisB,
			},
		.edge =
			{
				.normal = b3Vec3_zero,
				.separation = -INFINITY,
				.indexA = B3_NULL_INDEX,
				.indexB = B3_NULL_INDEX,
				.type = b3_edgePairAxis,
			},
		.separatedFeature = b3_invalidAxis,
	};

	int faceCountA = hullA->faceCount;
	const b3Plane* planesA = b3GetHullPlanes( hullA );

	int soaVertexCountB = ( hullB->vertexCount + 7 ) & ~( 7 );
	const float* vxB = b3GetHullSoaVertices( hullB );
	const float* vyB = vxB + soaVertexCountB;
	const float* vzB = vyB + soaVertexCountB;

	b3Vec3 cB = b3AABB_Center( hullB->aabb );
	b3Vec3 hB = b3AABB_Extents( hullB->aabb );

	// The hulls have a precomputed inner radius and centroid.
	// A given axis cannot achieve a separation larger than:
	// dot(axis, centerB - centerA) - innerRadiusA - innerRadiusB
	// So this is the upper bound for the separation of a candidate axis.
	// An axis can be skipped if it has an upper bound that is less than the current
	// best separation. This lets me skip many of the candidates without computing
	// support points.

	b3Vec3 deltaCenter = b3Sub( b3Add( b3MulMV( R, hullB->center ), xfB.p ), hullA->center );
	float centerDistance = b3Length( deltaCenter );
	float radius = hullA->innerRadius + hullB->innerRadius;

	// Adjust the radius to ensure the best axis isn't skipped.
	float radiusBound = radius - ( B3_LINEAR_SLOP + 0.001f * ( centerDistance + radius ) );

	// Compute dot(normalA, centerDelta) for all face normals of hullA.
	_Alignas( 16 ) float dotA[NF];
	b3GetFaceDotsW4( hullA, deltaCenter, dotA );

	// Find the face of hullA that most aligns with centerDelta.
	int seedIndexA = 0;
	float maxDotA = dotA[0];
	for ( int i = 1; i < faceCountA; ++i )
	{
		if ( dotA[i] > maxDotA )
		{
			maxDotA = dotA[i];
			seedIndexA = i;
		}
	}

	// Use the seed to get a lower bound on the separation for the faces of hullA.
	float floorA = -INFINITY;
	if ( earlyReturn )
	{
		b3Plane plane = planesA[seedIndexA];
		b3Vec3 direction = b3Neg( b3MulMV( invR, plane.normal ) );
		float planeSeparation = b3Dot( plane.normal, xfB.p ) - plane.offset;
		int vertexIndex;
		float separation =
			b3GetFaceSeparationW4( direction, planeSeparation, vxB, vyB, vzB, soaVertexCountB, cB, hB, &vertexIndex );
		floorA = b3MinFloat( separation, speculativeDistance );
	}

	// Test A's face planes against B's vertices.
	for ( int i = 0; i < faceCountA; ++i )
	{
		// The bound offset ensures the seed will be evaluated.
		if ( dotA[i] - radiusBound < b3MaxFloat( floorA, res.faceA.separation ) )
		{
			continue;
		}

		b3Plane plane = planesA[i];
		b3Vec3 direction = b3Neg( b3MulMV( invR, plane.normal ) );
		float planeSeparation = b3Dot( plane.normal, xfB.p ) - plane.offset;
		int vertexIndex;
		float separation =
			b3GetFaceSeparationW4( direction, planeSeparation, vxB, vyB, vzB, soaVertexCountB, cB, hB, &vertexIndex );
		if ( separation > res.faceA.separation )
		{
			res.faceA.normal = plane.normal;
			res.faceA.separation = separation;
			res.faceA.indexA = i;
			res.faceA.indexB = vertexIndex;
			if ( separation > speculativeDistance && earlyReturn )
			{
				res.separatedFeature = b3_faceAxisA;
				return res;
			}
		}
	}

	B3_VALIDATE( res.faceA.indexA != B3_NULL_INDEX );

	int faceCountB = hullB->faceCount;
	const b3Plane* planesB = b3GetHullPlanes( hullB );

	int soaVertexCountA = ( hullA->vertexCount + 7 ) & ~( 7 );
	const float* vxA = b3GetHullSoaVertices( hullA );
	const float* vyA = vxA + soaVertexCountA;
	const float* vzA = vyA + soaVertexCountA;

	b3Vec3 cA = b3AABB_Center( hullA->aabb );
	b3Vec3 hA = b3AABB_Extents( hullA->aabb );

	// Similarly, find the face of hullB that most aligns with the vector pointing from centerB to centerA.
	_Alignas( 16 ) float dotB[NF];
	b3GetFaceDotsW4( hullB, b3Neg( b3MulMV( invR, deltaCenter ) ), dotB );

	int seedIndexB = 0;
	float maxDotB = dotB[0];
	for ( int i = 1; i < faceCountB; ++i )
	{
		if ( dotB[i] > maxDotB )
		{
			maxDotB = dotB[i];
			seedIndexB = i;
		}
	}

	// Get a lower bound on the separation for the faces of hullB.
	float floorB = -INFINITY;
	if ( earlyReturn )
	{
		b3Plane plane = planesB[seedIndexB];
		b3Vec3 direction = b3Neg( b3MulMV( R, plane.normal ) );
		float planeSeparation = b3Dot( direction, xfB.p ) - plane.offset;
		int vertexIndex;
		float separation =
			b3GetFaceSeparationW4( direction, planeSeparation, vxA, vyA, vzA, soaVertexCountA, cA, hA, &vertexIndex );

		// Include the floor set by hull A faces.
		floorB = b3MaxFloat( separation, res.faceA.separation );
		floorB = b3MinFloat( floorB, speculativeDistance );
	}

	// Test B's face planes against A's vertices.
	for ( int i = 0; i < faceCountB; ++i )
	{
		if ( dotB[i] - radiusBound < b3MaxFloat( floorB, res.faceB.separation ) )
		{
			continue;
		}

		b3Plane plane = planesB[i];
		b3Vec3 direction = b3Neg( b3MulMV( R, plane.normal ) );
		float planeSeparation = b3Dot( direction, xfB.p ) - plane.offset;
		int vertexIndex;
		float separation =
			b3GetFaceSeparationW4( direction, planeSeparation, vxA, vyA, vzA, soaVertexCountA, cA, hA, &vertexIndex );
		if ( separation > res.faceB.separation )
		{
			res.faceB.normal = direction;
			res.faceB.separation = separation;
			res.faceB.indexA = vertexIndex;
			res.faceB.indexB = i;
			if ( separation > speculativeDistance && earlyReturn )
			{
				res.separatedFeature = b3_faceAxisB;
				return res;
			}
		}
	}

	// Transform B into A's space once, into SoA arrays. Extra space so tail can be set to zero.
	_Static_assert( ( B3_MAX_HULL_EDGES & ( B3_SIMD_WIDTH - 1 ) ) == 0, "must be multiple of SIMD width" );
	_Static_assert( ( B3_MAX_HULL_FACES & ( B3_SIMD_WIDTH - 1 ) ) == 0, "must be multiple of SIMD width" );
	_Static_assert( ( B3_MAX_HULL_VERTICES & ( B3_SIMD_WIDTH - 1 ) ) == 0, "must be multiple of SIMD width" );

	// Bound to skip edge tests. Derived from:
	// dot(edgeNormal, deltaCenter) - radiusBound > maxSep
	float edgeBound = earlyReturn ? b3MaxFloat( res.faceA.separation, res.faceB.separation ) + radiusBound : -INFINITY;

	B3_VALIDATE( earlyReturn == false || centerDistance >= edgeBound );

	// Gather edges of A that can feasibly create a separating axis that beats the maximum face separation.
	int halfEdgeCountA = hullA->edgeCount;
	const b3HullHalfEdge* halfEdgesA = b3GetHullEdges( hullA );
	int edgeIndicesA[NE];
	int na = 0;
	for ( int i = 0; i < halfEdgeCountA; i += 2 )
	{
		int i1 = halfEdgesA[i].face;
		int i2 = halfEdgesA[i + 1].face;
		float c = b3Dot( planesA[i1].normal, planesA[i2].normal );
		edgeIndicesA[na] = i;
		na += b3TestEdgeCandidateW4( dotA[i1], dotA[i2], c, edgeBound );
	}

	// Similar for edges of B.
	int halfEdgeCountB = hullB->edgeCount;
	const b3HullHalfEdge* halfEdgesB = b3GetHullEdges( hullB );
	int edgeIndicesB[B3_MAX_HULL_EDGES];
	int nb = 0;
	for ( int i = 0; i < halfEdgeCountB; i += 2 )
	{
		int i1 = halfEdgesB[i].face;
		int i2 = halfEdgesB[i + 1].face;
		float c = b3Dot( planesB[i1].normal, planesB[i2].normal );
		edgeIndicesB[nb] = i;
		nb += b3TestEdgeCandidateW4( dotB[i1], dotB[i2], c, edgeBound );
	}

	if ( na == 0 || nb == 0 )
	{
		// No edge candidates found.
		return res;
	}

	// The alignments below are not necessary, but they don't hurt.

	// B face normals in A space, negated.
	_Alignas( 16 ) float bFNx[NF];
	_Alignas( 16 ) float bFNy[NF];
	_Alignas( 16 ) float bFNz[NF];

	// B vertices in A space, negated.
	_Alignas( 16 ) float bWx[NV];
	_Alignas( 16 ) float bWy[NV];
	_Alignas( 16 ) float bWz[NV];

	int soaFaceCountB = ( faceCountB + 7 ) & ~( 7 );
	const float* nxB = b3GetHullSoaNormals( hullB );
	const float* nyB = nxB + soaFaceCountB;
	const float* nzB = nyB + soaFaceCountB;

	b3NegativeTransformFromSoAW4( R, xfB.p, nxB, nyB, nzB, soaFaceCountB, bFNx, bFNy, bFNz, false );
	b3NegativeTransformFromSoAW4( R, xfB.p, vxB, vyB, vzB, soaVertexCountB, bWx, bWy, bWz, true );

	// Per A edge data, already in A's space so just gathered. n0 and n1 are the two face
	// normals, d the edge vector av1-av0, v0 the first vertex. Tol is the
	// parallel edge tolerance, scaled by the edge length.
	_Alignas( 16 ) float aN0x[NE];
	_Alignas( 16 ) float aN0y[NE];
	_Alignas( 16 ) float aN0z[NE];
	_Alignas( 16 ) float aN1x[NE];
	_Alignas( 16 ) float aN1y[NE];
	_Alignas( 16 ) float aN1z[NE];
	// dir = av1 - av0
	_Alignas( 16 ) float aDx[NE];
	_Alignas( 16 ) float aDy[NE];
	_Alignas( 16 ) float aDz[NE];
	_Alignas( 16 ) float aV0x[NE];
	_Alignas( 16 ) float aV0y[NE];
	_Alignas( 16 ) float aV0z[NE];
	_Alignas( 16 ) float aTol[NE];

	float squaredTol = B3_PARALLEL_EDGE_TOL * B3_PARALLEL_EDGE_TOL;
	for ( int k = 0; k < na; ++k )
	{
		const b3HullHalfEdge* edge = halfEdgesA + edgeIndicesA[k];
		const b3HullHalfEdge* twin = edge + 1;

		b3Vec3 A = planesA[edge->face].normal;
		b3Vec3 B = planesA[twin->face].normal;
		aN0x[k] = A.x;
		aN0y[k] = A.y;
		aN0z[k] = A.z;
		aN1x[k] = B.x;
		aN1y[k] = B.y;
		aN1z[k] = B.z;

		int v0 = edge->origin;
		int v1 = twin->origin;

		aDx[k] = vxA[v1] - vxA[v0];
		aDy[k] = vyA[v1] - vyA[v0];
		aDz[k] = vzA[v1] - vzA[v0];
		aV0x[k] = vxA[v0];
		aV0y[k] = vyA[v0];
		aV0z[k] = vzA[v0];

		aTol[k] = squaredTol * ( aDx[k] * aDx[k] + aDy[k] * aDy[k] + aDz[k] * aDz[k] );
	}

	// Zero the tail lanes.
	b3FloatW4 zero = b3ZeroW4();
	b3Vec3W4 zeroV = { zero, zero, zero };
	b3StoreVW4( aN0x + na, aN0y + na, aN0z + na, zeroV );
	b3StoreVW4( aN1x + na, aN1y + na, aN1z + na, zeroV );
	b3StoreVW4( aDx + na, aDy + na, aDz + na, zeroV );
	b3StoreVW4( aV0x + na, aV0y + na, aV0z + na, zeroV );

	b3StoreW4( aTol + na, zero );

	float linearSlop = B3_LINEAR_SLOP;

#if defined( B3_SIMD_NONE )

	// The SIMD emulated version of this code is very slow. This is a purely scalar version
	// for platforms that don't have SIMD capability. It is much faster than SIMD emulation.
	// WARNING: this math needs to match the SIMD version for cross platform determinism.

	const float EPS = -linearSlop * linearSlop;

	for ( int j = 0; j < nb; ++j )
	{
		const b3HullHalfEdge* edge = halfEdgesB + edgeIndicesB[j];
		const b3HullHalfEdge* twin = edge + 1;
		int f0 = edge->face;
		int f1 = twin->face;
		int v0 = edge->origin;
		int v1 = twin->origin;

		b3Vec3 C = { bFNx[f0], bFNy[f0], bFNz[f0] };
		b3Vec3 D = { bFNx[f1], bFNy[f1], bFNz[f1] };
		b3Vec3 bv0 = { bWx[v0], bWy[v0], bWz[v0] };
		b3Vec3 bv1 = { bWx[v1], bWy[v1], bWz[v1] };
		b3Vec3 DC = b3Sub( bv1, bv0 );

		for ( int i = 0; i < na; ++i )
		{
			b3Vec3 d = { aDx[i], aDy[i], aDz[i] };

			// CBA = C.dir, DBA = D.dir, where dir = B_x_A
			float CBA = b3Dot( C, d );
			float DBA = b3Dot( D, d );
			if ( CBA * DBA >= EPS )
			{
				continue;
			}

			b3Vec3 n0 = { aN0x[i], aN0y[i], aN0z[i] };
			b3Vec3 n1 = { aN1x[i], aN1y[i], aN1z[i] };

			// ADC = n0.DC, BDC = n1.DC, where DC = D_x_C
			float ADC = b3Dot( n0, DC );
			float BDC = b3Dot( n1, DC );
			if ( ADC * BDC >= EPS || CBA * BDC >= EPS )
			{
				continue;
			}

			// Reject near parallel edges
			float maxCD = b3MaxFloat( CBA * CBA, DBA * DBA );
			if ( maxCD <= aTol[i] )
			{
				continue;
			}

			// t = -CBA / (DBA - CBA)
			float t = -CBA / ( DBA - CBA );

			// normal = lerp(t, C, D) = C + (D-C)*t
			b3Vec3 n = b3MulAdd( C, t, b3Sub( D, C ) );
			float len2 = b3Dot( n, n );

			float inv = 1.0f / sqrtf( len2 );
			n = b3MulSV( inv, n );

			// separation = -dot(normal, av0 + bv0)
			b3Vec3 av0 = { aV0x[i], aV0y[i], aV0z[i] };
			float separation = -b3Dot( b3Add( av0, bv0 ), n );

			if ( separation > res.edge.separation )
			{
				res.edge.normal = n;
				res.edge.separation = separation;

				// Half edge index
				res.edge.indexA = edgeIndicesA[i];
				res.edge.indexB = edgeIndicesB[j];

				if ( separation > speculativeDistance && earlyReturn )
				{
					res.separatedFeature = b3_edgePairAxis;
					return res;
				}
			}
		}
	}

#else

	// This tolerance can skip edges shorter than 1cm.
	const b3FloatW4 EPS = b3SplatW4( -linearSlop * linearSlop );
	const b3FloatW4 INF = b3SplatW4( INFINITY );

	for ( int j = 0; j < nb; ++j )
	{
		const b3HullHalfEdge* edge = halfEdgesB + edgeIndicesB[j];
		const b3HullHalfEdge* twin = edge + 1;
		int f0 = edge->face;
		int f1 = twin->face;
		int v0 = edge->origin;
		int v1 = twin->origin;

		b3Vec3 nC = { bFNx[f0], bFNy[f0], bFNz[f0] };
		b3Vec3 nD = { bFNx[f1], bFNy[f1], bFNz[f1] };
		b3Vec3 pB = { bWx[v0], bWy[v0], bWz[v0] };
		b3Vec3 qB = { bWx[v1], bWy[v1], bWz[v1] };

		const b3Vec3W4 C = b3SplatVW4( nC );
		const b3Vec3W4 D = b3SplatVW4( nD );
		const b3Vec3W4 DC = b3SplatVW4( b3Sub( qB, pB ) );
		const b3Vec3W4 bv0 = b3SplatVW4( pB );

		for ( int i = 0; i < na; i += B3_SIMD_WIDTH )
		{
			b3Vec3W4 n0 = b3LoadVW4( aN0x + i, aN0y + i, aN0z + i );
			b3Vec3W4 n1 = b3LoadVW4( aN1x + i, aN1y + i, aN1z + i );
			b3Vec3W4 d = b3LoadVW4( aDx + i, aDy + i, aDz + i );
			b3Vec3W4 av0 = b3LoadVW4( aV0x + i, aV0y + i, aV0z + i );

			b3FloatW4 tol = b3LoadW4( aTol + i );

			// CBA = C.dir, DBA = D.dir, where dir = B_x_A
			b3FloatW4 CBA = b3DotW4( C, d );
			b3FloatW4 DBA = b3DotW4( D, d );
			// ADC = n0.DC, BDC = n1.DC, where DC = D_x_C
			b3FloatW4 ADC = b3DotW4( n0, DC );
			b3FloatW4 BDC = b3DotW4( n1, DC );

			// Gauss map arc crossing test, CBA*DBA<eps and ADC*BDC<eps and CBA*BDC<eps
			b3FloatW4 m1 = b3LessThanW4( b3MulW4( CBA, DBA ), EPS );
			b3FloatW4 m2 = b3LessThanW4( b3MulW4( ADC, BDC ), EPS );
			b3FloatW4 m3 = b3LessThanW4( b3MulW4( CBA, BDC ), EPS );

			// Reject near parallel edges. The arc lerp is ill conditioned when both of B's normals are nearly
			// perpendicular to edge A, a scale invariant sine threshold relative to the edge length.
			b3FloatW4 maxCD = b3MaxW4( b3MulW4( CBA, CBA ), b3MulW4( DBA, DBA ) );
			b3FloatW4 notParallel = b3GreaterThanW4( maxCD, tol );
			b3FloatW4 mask = b3AndW4( b3AndW4( m1, m2 ), b3AndW4( m3, notParallel ) );

			// Most A-edges fail the Gauss test, so skip the divide, sqrt and support work when no
			// lane passed.
			if ( b3AnyTrueW4( mask ) == false )
			{
				continue;
			}

			// t = -CBA / (DBA - CBA)
			b3FloatW4 t = b3DivW4( b3NegW4( CBA ), b3SubW4( DBA, CBA ) );

			// normal = lerp(t, C, D) = C + (D-C)*t
			b3Vec3W4 n = b3MulAddSVW4( C, t, b3SubVW4( D, C ) );

			// normalize
			b3FloatW4 len2 = b3DotW4( n, n );
			b3FloatW4 inv = b3DivW4( b3SplatW4( 1.0f ), b3SqrtW4( len2 ) );
			n = b3MulSVW4( inv, n );

			// support = dot(normal, av0 + bv0)
			b3FloatW4 support = b3DotW4( b3AddVW4( av0, bv0 ), n );

			// Lanes that fail the Gauss test can never win.
			support = b3BlendW4( INF, support, mask );
			b3FloatW4 separation = b3NegW4( support );

			// Test all B3_SIMD_WIDTH supports against the running best at once. If none beats it, skip the
			// store and scalar reduction.
			b3FloatW4 improves = b3GreaterThanW4( separation, b3SplatW4( res.edge.separation ) );
			if ( b3AnyTrueW4( improves ) == false )
			{
				continue;
			}

			_Alignas( 16 ) float sA[B3_SIMD_WIDTH];
			_Alignas( 16 ) float nxA[B3_SIMD_WIDTH];
			_Alignas( 16 ) float nyA[B3_SIMD_WIDTH];
			_Alignas( 16 ) float nzA[B3_SIMD_WIDTH];
			b3StoreW4( sA, separation );
			b3StoreVW4( nxA, nyA, nzA, n );

			// Reduce in lane order so ties keep the first edge and the early out takes the first
			// improving support below zero. Padded tail lanes carry +INF support, so they never
			// update or index edges out of range.
			for ( int lane = 0; lane < B3_SIMD_WIDTH; lane++ )
			{
				int ei = i + lane;
				float s = sA[lane];
				if ( s > res.edge.separation )
				{
					res.edge.normal = (b3Vec3){ nxA[lane], nyA[lane], nzA[lane] };
					res.edge.separation = s;

					// Half edge index
					res.edge.indexA = edgeIndicesA[ei];
					res.edge.indexB = edgeIndicesB[j];

					if ( s > speculativeDistance && earlyReturn )
					{
						res.separatedFeature = b3_edgePairAxis;
						return res;
					}
				}
			}
		}
	}
#endif

	return res;
}

#undef B3_SIMD_WIDTH
#undef NE
#undef NF
#undef NV

void b3CollideHullsW4( b3LocalManifold* manifold, int capacity, const b3HullData* hullA, const b3HullData* hullB,
					   b3Transform transformBtoA, b3SATCache* cache )
{
	manifold->pointCount = 0;

	if ( capacity < 4 )
	{
		return;
	}

	// Work in shapeA coordinates
	float speculativeDistance = B3_SPECULATIVE_DISTANCE;

	float linearSlop = B3_LINEAR_SLOP;
	const b3HullHalfEdge* edgesA = b3GetHullEdges( hullA );
	const b3Plane* planesA = b3GetHullPlanes( hullA );
	const b3Vec3* pointsA = b3GetHullPoints( hullA );

	const b3HullHalfEdge* edgesB = b3GetHullEdges( hullB );
	const b3Plane* planesB = b3GetHullPlanes( hullB );
	const b3Vec3* pointsB = b3GetHullPoints( hullB );

	cache->hit = 0;

	// Attempt to use the cache to speed up collision
	switch ( cache->type )
	{
		case b3_invalidAxis:
			break;

		case b3_faceAxisA:
		{
			B3_ASSERT( cache->indexA < hullA->faceCount );

			// Check for separation using cached face
			b3Plane plane = planesA[cache->indexA];
			b3Vec3 searchDirectionInB = b3Neg( b3InvRotateVector( transformBtoA.q, plane.normal ) );

			int vertexIndex = b3FindHullSupportVertex( hullB, searchDirectionInB );
			b3Vec3 support = b3TransformPoint( transformBtoA, pointsB[vertexIndex] );
			float separation = b3PlaneSeparation( plane, support );

			if ( separation >= speculativeDistance )
			{
				// Cache hit, shapes are separated
				cache->hit = 1;
				return;
			}

			// Attempt face contact using cached feature
			b3SeparatingAxis faceQuery;
			faceQuery.normal = plane.normal;
			faceQuery.separation = 0.0f;
			faceQuery.indexA = cache->indexA;
			faceQuery.indexB = vertexIndex;
			faceQuery.type = b3_faceAxisA;

			b3SATCache localCache = { 0 };
			bool touching = b3BuildFaceAContact( manifold, capacity, hullA, hullB, transformBtoA, faceQuery, &localCache );
			if ( touching == true && b3AbsFloat( cache->separation - localCache.separation ) < linearSlop )
			{
				// Cache hit, contact points generated
				cache->hit = 1;
				return;
			}
		}
		break;

		case b3_faceAxisB:
		{
			B3_ASSERT( cache->indexB < hullB->faceCount );

			// Check for separation using cached face
			b3Plane plane = planesB[cache->indexB];
			b3Vec3 searchDirectionInA = b3Neg( b3RotateVector( transformBtoA.q, plane.normal ) );

			// todo use b3GetSupportW4
			int vertexIndex = b3FindHullSupportVertex( hullA, searchDirectionInA );
			b3Vec3 support = b3InvTransformPoint( transformBtoA, pointsA[vertexIndex] );
			float separation = b3PlaneSeparation( plane, support );

			if ( separation >= speculativeDistance )
			{
				// Cache hit, shapes are separated
				cache->hit = 1;
				return;
			}

			// Attempt face contact using cached feature
			b3SeparatingAxis faceQuery;
			faceQuery.normal = b3Neg( plane.normal );
			faceQuery.separation = 0.0f;
			faceQuery.indexA = vertexIndex;
			faceQuery.indexB = cache->indexB;
			faceQuery.type = b3_faceAxisB;

			b3SATCache localCache = { 0 };
			bool touching = b3BuildFaceBContact( manifold, capacity, hullA, hullB, transformBtoA, faceQuery, &localCache );
			if ( touching == true && b3AbsFloat( cache->separation - localCache.separation ) < linearSlop )
			{
				// Cache hit, contact points generated
				cache->hit = 1;
				return;
			}
		}
		break;

		case b3_edgePairAxis:
		{
			int indexA = cache->indexA;
			const b3HullHalfEdge* edge1 = edgesA + indexA;
			const b3HullHalfEdge* twin1 = edgesA + indexA + 1;
			B3_ASSERT( edge1->twin == indexA + 1 && twin1->twin == indexA );

			b3Vec3 pA = pointsA[edge1->origin];
			b3Vec3 qA = pointsA[twin1->origin];
			b3Vec3 eA = b3Sub( qA, pA );

			b3Vec3 uA = planesA[edge1->face].normal;
			b3Vec3 vA = planesA[twin1->face].normal;

			int indexB = cache->indexB;
			const b3HullHalfEdge* edge2 = edgesB + indexB;
			const b3HullHalfEdge* twin2 = edgesB + indexB + 1;
			B3_ASSERT( edge2->twin == indexB + 1 && twin2->twin == indexB );

			b3Vec3 pB = b3TransformPoint( transformBtoA, pointsB[edge2->origin] );
			b3Vec3 qB = b3TransformPoint( transformBtoA, pointsB[twin2->origin] );
			b3Vec3 eB = b3Sub( qB, pB );

			b3Vec3 uB = b3RotateVector( transformBtoA.q, planesB[edge2->face].normal );
			b3Vec3 vB = b3RotateVector( transformBtoA.q, planesB[twin2->face].normal );

			// flipping the signs of u2 and v2
			// cross(v2, u2) == cross(-v2, -u2)
			// so we still use -e2
			// but we can also use e1 = cross(u1, v1) and e2 = cross(u2, v2)
			float cba = b3Dot( uB, eA );
			float dba = b3Dot( vB, eA );
			float adc = -b3Dot( uA, eB );
			float bdc = -b3Dot( vA, eB );

			if ( cba * dba < 0.0f && adc * bdc < 0.0f && cba * bdc > 0.0f )
			{
				// Avoid nearly parallel edges that may lead to invalid separation values at the noise floor.
				float squaredTolerance = B3_PARALLEL_EDGE_TOL * B3_PARALLEL_EDGE_TOL;
				if ( b3MaxFloat( cba * cba, dba * dba ) >= squaredTolerance * b3LengthSquared( eA ) )
				{
					// Transform reference center of the first hull into local space of the second hull
					float t = cba / ( cba - dba );
					b3Vec3 axis = b3Lerp( uB, vB, t );
					B3_VALIDATE( b3LengthSquared( axis ) > 1000.0f * FLT_MIN );
					axis = b3Normalize( axis );
					float separation = b3Dot( axis, b3Sub( qA, qB ) );

					if ( separation > speculativeDistance )
					{
						// Cache hit, shapes are separated
						cache->hit = 1;
						return;
					}

					// Try to rebuild contact from last features
					b3SeparatingAxis edgeQuery = { 0 };
					edgeQuery.normal = b3Neg( axis );
					edgeQuery.separation = 0.0f;
					edgeQuery.indexA = cache->indexA;
					edgeQuery.indexB = cache->indexB;
					edgeQuery.type = b3_edgePairAxis;

					b3SATCache localCache = { 0 };
					bool touching = b3BuildEdgeContact( manifold, hullA, hullB, transformBtoA, edgeQuery, &localCache );

					// This separation tolerance may have a big impact on performance in some benchmarks.
					if ( touching && b3AbsFloat( cache->separation - localCache.separation ) < linearSlop )
					{
						// Cache hit, contact point generated
						cache->hit = 1;
						return;
					}
				}
			}
		}
		break;

			// This case is for testing
		case b3_manualFaceAxisA:
		{
			b3AxisQuery axisQuery = b3ComputeSeparatingAxisW4( hullA, hullB, transformBtoA, false );
			b3SeparatingAxis faceQuery = axisQuery.faceA;
			b3BuildFaceAContact( manifold, capacity, hullA, hullB, transformBtoA, faceQuery, cache );
			return;
		}

			// This case is for testing
		case b3_manualFaceAxisB:
		{
			b3AxisQuery axisQuery = b3ComputeSeparatingAxisW4( hullA, hullB, transformBtoA, false );
			b3SeparatingAxis faceQuery = axisQuery.faceB;
			b3BuildFaceBContact( manifold, capacity, hullA, hullB, transformBtoA, faceQuery, cache );
			return;
		}

			// This case is for testing
		case b3_manualEdgePairAxis:
		{
			b3AxisQuery axisQuery = b3ComputeSeparatingAxisW4( hullA, hullB, transformBtoA, false );
			b3SeparatingAxis edgeQuery = axisQuery.edge;
			if ( edgeQuery.indexA != B3_NULL_INDEX )
			{
				b3BuildEdgeContact( manifold, hullA, hullB, transformBtoA, edgeQuery, cache );
			}
			return;
		}

		default:
			B3_ASSERT( false );
			break;
	}

	manifold->pointCount = 0;
	*cache = (b3SATCache){ 0 };

	b3AxisQuery axisQuery = b3ComputeSeparatingAxisW4( hullA, hullB, transformBtoA, true );

	if ( axisQuery.separatedFeature != b3_invalidAxis )
	{
		// We found a separating axis
		cache->type = axisQuery.separatedFeature;

		if ( axisQuery.separatedFeature == b3_faceAxisA )
		{
			B3_VALIDATE( axisQuery.faceA.separation > speculativeDistance );
			cache->separation = axisQuery.faceA.separation;
			cache->indexA = (uint8_t)axisQuery.faceA.indexA;
			cache->indexB = (uint8_t)axisQuery.faceA.indexB;
		}
		else if ( axisQuery.separatedFeature == b3_faceAxisB )
		{
			B3_VALIDATE( axisQuery.faceB.separation > speculativeDistance );
			cache->separation = axisQuery.faceB.separation;
			cache->indexA = (uint8_t)axisQuery.faceB.indexA;
			cache->indexB = (uint8_t)axisQuery.faceB.indexB;
		}
		else
		{
			B3_ASSERT( axisQuery.separatedFeature == b3_edgePairAxis );
			B3_VALIDATE( axisQuery.edge.separation > speculativeDistance );
			cache->separation = axisQuery.edge.separation;
			cache->indexA = (uint8_t)axisQuery.edge.indexA;
			cache->indexB = (uint8_t)axisQuery.edge.indexB;
		}
		return;
	}

	B3_VALIDATE( axisQuery.faceA.separation <= speculativeDistance || axisQuery.faceB.separation <= speculativeDistance ||
				 axisQuery.edge.separation <= speculativeDistance );

	if ( axisQuery.faceA.separation > axisQuery.faceB.separation )
	{
		b3SeparatingAxis faceQuery = axisQuery.faceA;
		B3_VALIDATE( 0 <= faceQuery.indexA && faceQuery.indexA < hullA->faceCount );
		B3_VALIDATE( 0 <= faceQuery.indexB && faceQuery.indexB < hullB->vertexCount );

		// Face contact A
		b3BuildFaceAContact( manifold, capacity, hullA, hullB, transformBtoA, faceQuery, cache );

		B3_VALIDATE( cache->indexA < hullA->faceCount );
		B3_VALIDATE( cache->indexB < hullB->vertexCount );
	}
	else
	{
		b3SeparatingAxis faceQuery = axisQuery.faceB;
		B3_VALIDATE( 0 <= faceQuery.indexA && faceQuery.indexA < hullA->vertexCount );
		B3_VALIDATE( 0 <= faceQuery.indexB && faceQuery.indexB < hullB->faceCount );

		// Face contact B
		b3BuildFaceBContact( manifold, capacity, hullA, hullB, transformBtoA, faceQuery, cache );

		B3_VALIDATE( cache->indexA < hullA->vertexCount );
		B3_VALIDATE( cache->indexB < hullB->faceCount );
	}

	b3SeparatingAxis edgeQuery = axisQuery.edge;

	if ( edgeQuery.indexA == B3_NULL_INDEX )
	{
		// There are no valid edge pairs (all edges parallel)
		return;
	}

	float faceSeparation = b3MaxFloat( axisQuery.faceA.separation, axisQuery.faceB.separation );
	float clipSeparation = cache->separation;
	float edgeTol = linearSlop;

	// Face contact can be empty if it is not the axis of maximum separation. It can also
	// be empty in narrow cases in the speculative region. If that case was important then
	// a GJK fallback would be used. So far it doesn't seem important.
	// Create edge contact if face contact fails or edge contact is significantly better.
	if ( ( manifold->pointCount == 0 && edgeQuery.separation > faceSeparation ) ||
		 edgeQuery.separation > clipSeparation + edgeTol )
	{
		B3_ASSERT( 0 <= edgeQuery.indexA && edgeQuery.indexA < hullA->edgeCount );
		B3_ASSERT( 0 <= edgeQuery.indexB && edgeQuery.indexB < hullB->edgeCount );

		// Edge contact
		b3LocalManifold edgeManifold = { 0 };
		b3LocalManifoldPoint edgePoint = { 0 };
		edgeManifold.points = &edgePoint;

		b3SATCache edgeCache = { 0 };
		b3BuildEdgeContact( &edgeManifold, hullA, hullB, transformBtoA, edgeQuery, &edgeCache );

		// It is possible with speculation to have vertex-vertex collision that is missed by SAT,
		// so edge contact yields no points. In that case perhaps the face contact has some points.
		if ( edgeManifold.pointCount == 1 )
		{
			// Copy edge manifold out, being careful to preserve manifold point buffer.
			b3LocalManifoldPoint* points = manifold->points;
			*manifold = edgeManifold;
			manifold->points = points;
			manifold->points[0] = edgePoint;
			*cache = edgeCache;
		}
	}
}

#undef B3_SIMD_WIDTH
#endif
