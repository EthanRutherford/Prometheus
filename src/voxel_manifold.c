// SPDX-FileCopyrightText: 2026 Ethan Rutherford
// SPDX-License-Identifier: MIT

// a note about boundary checks within this file; when checking against voxel boundaries, we treat inside as
// min <= inside < max. This is so that contacts that land on the boundary between multiple voxels will
// avoid duplicating the contact point on each voxel; only one voxel will "see" the contact position and add
// it as a contact point, ensuring that we avoid duplicated contact points.

#include "bits.h"
#include "contact.h"
#include "manifold.h"
#include "physics_world.h"
#include "reduce_cluster.h"
#include "shape.h"
#include "simd.h"

#include "box3d/types.h"

#include <stdlib.h>

#define POINT_BUFFER_CAPACITY 256
#define POINT_RECYCLE_TOL_2 ( B3_LINEAR_SLOP * B3_LINEAR_SLOP )

// predefine some SIMD constants.
static const b3FloatW zeroW = B3_STATIC_FLOAT_W( 0.0f );
static const b3FloatW halfW = B3_STATIC_FLOAT_W( 0.5f );
static const b3FloatW oneW = B3_STATIC_FLOAT_W( 1.0f );
static const b3FloatW epsilonW = B3_STATIC_FLOAT_W( 1000.0f * FLT_MIN );
static const b3Vec3W zeroVW = { B3_STATIC_FLOAT_W( 0.0f ), B3_STATIC_FLOAT_W( 0.0f ), B3_STATIC_FLOAT_W( 0.0f ) };
static const b3Vec3W oneVW = { B3_STATIC_FLOAT_W( 1.0f ), B3_STATIC_FLOAT_W( 1.0f ), B3_STATIC_FLOAT_W( 1.0f ) };

// helper for extracting floats/ints from SIMD lanes or vector components.
#define FLT( v, i ) ( ( (float*)&( v ) )[i] )
#define INT( v, i ) ( ( (int*)&( v ) )[i] )
#define COLUMN( mat, i ) ( ( (b3Vec3*)&( mat ) )[i] )

static const uint32_t negAxisNeighbors[3] = { b3_negXNeighbor, b3_negYNeighbor, b3_negZNeighbor };
static const uint32_t posAxisNeighbors[3] = { b3_posXNeighbor, b3_posYNeighbor, b3_posZNeighbor };

typedef struct VoxelWide
{
	b3Vec3W min;
	b3Vec3W max;
	b3Vec3W point;
	b3Vec3W normal;
	b3FloatW separation;
	b3FloatW flags;
	b3FloatW accepted;
} VoxelWide;

typedef struct CacheRefreshContext
{
	const b3VoxelData* voxels;
	b3VoxelContact* contact;
} CacheRefreshContext;

typedef struct VoxCandidatePoint
{
	b3Vec3 point;
	b3Vec3 normal;
	float separation;
} VoxCandidatePoint;

typedef struct VoxCollideContext
{
	b3VoxelContact* contact;

	VoxCandidatePoint* pointBuffer;
	int pointCount;

	b3Voxels voxelsA;

	union
	{
		const b3Sphere* sphereB;
		const b3Capsule* capsuleB;
		const b3HullData* hullB;
		b3Voxels voxelsB;
	};
} VoxCollideContext;

static b3Vec3 transformPointMat( b3Matrix3 mat, b3Vec3 t, b3Vec3 p )
{
	return b3Add( b3MulMV( mat, p ), t );
}

static b3Vec3 invTransformPointMat( b3Matrix3 invMat, b3Vec3 t, b3Vec3 p )
{
	return b3MulMV( invMat, b3Sub( p, t ) );
}

// builds a mask that can be used to detect if a voxel has a neighbor that is closer to the candidate point than itself.
// This is used to cull contact points early, knowing that there is at least one coplanar voxel that can generate a deeper
// contact point. This early culling helps reduce load on the later clustering algorithm, which can cull additional points.
static b3FloatW getNeighborMaskW( const b3Vec3W candidate, const b3Vec3W voxMin, const b3Vec3W voxMax )
{
	static const b3FloatW b3_negXNeighborW = B3_STATIC_MASK_W( b3_negXNeighbor );
	static const b3FloatW b3_posXNeighborW = B3_STATIC_MASK_W( b3_posXNeighbor );
	static const b3FloatW b3_negYNeighborW = B3_STATIC_MASK_W( b3_negYNeighbor );
	static const b3FloatW b3_posYNeighborW = B3_STATIC_MASK_W( b3_posYNeighbor );
	static const b3FloatW b3_negZNeighborW = B3_STATIC_MASK_W( b3_negZNeighbor );
	static const b3FloatW b3_posZNeighborW = B3_STATIC_MASK_W( b3_posZNeighbor );

	// this is effectively an AABB SAT test, resulting in a mask of the separating axes.
	// if a voxel has a neighbor along a separating axis, that neighbor is closer to the candidate point.
	// An extra bonus, this also filters out any contact points which would have a normal pointed into
	// a neighboring voxel, which is not a valid contact point for collision resolution.
	b3FloatW neighborMask = { 0, 0, 0, 0 };
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_negXNeighborW, b3LessThanW( candidate.X, voxMin.X ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posXNeighborW, b3GreaterOrEqualW( candidate.X, voxMax.X ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_negYNeighborW, b3LessThanW( candidate.Y, voxMin.Y ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posYNeighborW, b3GreaterOrEqualW( candidate.Y, voxMax.Y ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_negZNeighborW, b3LessThanW( candidate.Z, voxMin.Z ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posZNeighborW, b3GreaterOrEqualW( candidate.Z, voxMax.Z ) ) );
	return neighborMask;
}

// Clip the query bounds to the voxel grid bounds, matching the behavior of b3QueryVoxels.
// This stabilizes the voxel contact cache when the query bounds shift by less than a voxel size.
static b3AABB computeVoxelBounds( const b3VoxelData* voxels, b3Vec3 lower, b3Vec3 upper )
{
	b3Vec3 lowerBound = b3Max( b3Floor( lower ), voxels->bounds.lowerBound );
	b3Vec3 upperBound = b3Min( b3Ceil( upper ), voxels->bounds.upperBound );
	return (b3AABB){ lowerBound, upperBound };
}

static void cacheRefreshCallback( uint64_t code, uint32_t index, void* context )
{
	CacheRefreshContext* ctx = (CacheRefreshContext*)context;

	uint32_t voxelFlags = b3GetVoxelAttrs( ctx->voxels )[index].flags;

	// skip occluded voxels, we only want to gather surface voxels
	if ( ( voxelFlags & b3_voxOccludedMask ) == b3_voxOccludedMask )
		return;

	b3VoxelCache* cache = ctx->contact->voxelCache.data + ctx->contact->voxelCache.count;
	ctx->contact->voxelCache.count++;

	cache->min.x = (float)b3DecodeVoxelX( code );
	cache->min.y = (float)b3DecodeVoxelY( code );
	cache->min.z = (float)b3DecodeVoxelZ( code );
	cache->flags = voxelFlags;
}

static void refreshVoxCache( b3VoxelContact* contact, const b3VoxelData* voxels, b3AABB bounds )
{
	// if the bounds are inside the cached bounds, we're done
	if ( b3AABB_Contains( contact->queryBounds, bounds ) )
		return;

	// compute maximum possible voxels returned.
	int maxVoxels = (int)( ( bounds.upperBound.x - bounds.lowerBound.x ) * ( bounds.upperBound.y - bounds.lowerBound.y ) *
						   ( bounds.upperBound.z - bounds.lowerBound.z ) );

	// clear the cache, reserve space, and gather new voxels
	contact->queryBounds = bounds;
	contact->voxelCache.count = 0;
	b3Array_Reserve( contact->voxelCache, maxVoxels );

	CacheRefreshContext ctx = { .voxels = voxels, .contact = contact };
	b3QueryVoxels( voxels, bounds, cacheRefreshCallback, &ctx );
}

static void collideVoxSphereW( VoxCollideContext* context, b3Transform bToA, b3Arena arena )
{
	// get the center, radius, and speculative distance of the sphere in voxel space/scale
	const b3Voxels voxelsA = context->voxelsA;
	float invScale = 1.0f / voxelsA.scale;
	float specDist = B3_SPECULATIVE_DISTANCE * invScale;
	float radius = context->sphereB->radius * invScale;
	float maxDistSqr = ( radius + specDist ) * ( radius + specDist );
	b3Vec3 center = b3MulSV( invScale, b3TransformPoint( bToA, context->sphereB->center ) );

	// compute the query bounds for the voxel grid. This is the AABB of the sphere expanded by the speculative distance.
	b3Vec3 extent = b3Vec3Of( radius + specDist );
	b3AABB queryBounds = computeVoxelBounds( voxelsA.data, b3Sub( center, extent ), b3Add( center, extent ) );

	// refresh the voxel cache
	refreshVoxCache( context->contact, voxelsA.data, queryBounds );

	// early exit if no voxels are in the query bounds
	if ( context->contact->voxelCache.count == 0 )
		return;

	// create and initialize the wide voxel array for SIMD processing
	int wideCount = ( context->contact->voxelCache.count + B3_SIMD_WIDTH - 1 ) / B3_SIMD_WIDTH;
	VoxelWide* wideVoxels = b3Bump( &arena, wideCount * sizeof( VoxelWide ) );
	for ( int i = 0; i < wideCount; i++ )
	{
		VoxelWide* vox = &wideVoxels[i];
		for ( int lane = 0; lane < B3_SIMD_WIDTH; lane++ )
		{
			int index = i * B3_SIMD_WIDTH + lane;
			if ( index >= context->contact->voxelCache.count )
			{
				INT( vox->flags, lane ) = 0;
				continue;
			}

			b3VoxelCache* cache = &context->contact->voxelCache.data[index];
			FLT( vox->min.X, lane ) = cache->min.x;
			FLT( vox->min.Y, lane ) = cache->min.y;
			FLT( vox->min.Z, lane ) = cache->min.z;
			INT( vox->flags, lane ) = cache->flags;
		}
	}

	// create wide vectors for intersection parameters
	b3FloatW maxDistSqrW = b3SplatW( maxDistSqr );
	b3FloatW scale = b3SplatW( voxelsA.scale );
	b3FloatW radiusW = b3SplatW( radius );
	b3Vec3W centerW = { b3SplatW( center.x ), b3SplatW( center.y ), b3SplatW( center.z ) };

	// Collide Step 1: filter gathered candidates using neighbor masks.
	for ( int i = 0; i < wideCount; i++ )
	{
		VoxelWide* vox = &wideVoxels[i];
		vox->max = b3AddVW( vox->min, oneVW );

		// if there is a neighboring voxel which is closer to the sphere center than this voxel, then skip this one.
		// A neighboring voxel means we are part of an edge/surface, and we ideally only generate one contact point per
		// edge/surface. This reduces the number of contact points the manifold clustering algorithm needs to process.
		b3FloatW neighborMask = getNeighborMaskW( centerW, vox->min, vox->max );
		b3FloatW neighborResults = b3AndW( vox->flags, neighborMask );
		if ( b3AllTrueW( neighborResults ) )
			continue;

		vox->accepted = b3EqualsW( neighborResults, zeroW );
	}

	// Step 2: compact the accepted candidates down in-place
	int acceptedCount = 0;
	for ( int i = 0; i < context->contact->voxelCache.count; i++ )
	{
		int wi = i / B3_SIMD_WIDTH;
		int li = i % B3_SIMD_WIDTH;
		if ( INT( wideVoxels[wi].accepted, li ) != 0 )
		{
			if ( i != acceptedCount )
			{
				int wj = acceptedCount / B3_SIMD_WIDTH;
				int lj = acceptedCount % B3_SIMD_WIDTH;
				FLT( wideVoxels[wj].min.X, lj ) = FLT( wideVoxels[wi].min.X, li );
				FLT( wideVoxels[wj].min.Y, lj ) = FLT( wideVoxels[wi].min.Y, li );
				FLT( wideVoxels[wj].min.Z, lj ) = FLT( wideVoxels[wi].min.Z, li );

				FLT( wideVoxels[wj].max.X, lj ) = FLT( wideVoxels[wi].max.X, li );
				FLT( wideVoxels[wj].max.Y, lj ) = FLT( wideVoxels[wi].max.Y, li );
				FLT( wideVoxels[wj].max.Z, lj ) = FLT( wideVoxels[wi].max.Z, li );

				INT( wideVoxels[wj].flags, lj ) = INT( wideVoxels[wi].flags, li );
			}

			acceptedCount++;
		}
	}

	// recompute the wide count and clear the accepted flags for any overflow lanes
	wideCount = ( acceptedCount + B3_SIMD_WIDTH - 1 ) / B3_SIMD_WIDTH;
	int overflowLanes = acceptedCount % B3_SIMD_WIDTH;
	if ( overflowLanes > 0 )
	{
		for ( int lane = overflowLanes; lane < B3_SIMD_WIDTH; lane++ )
		{
			INT( wideVoxels[wideCount - 1].flags, lane ) = 0;
		}
	}

	// Step 3: compute the closest point on each voxel to the sphere center, and compute the separation.
	for ( int i = 0; i < wideCount; i++ )
	{
		VoxelWide* vox = &wideVoxels[i];

		// compute the closest point on the voxel bounds to the sphere center
		b3Vec3W closestPoint = b3ClampVW( centerW, vox->min, vox->max );

		// compute the squared distance from the closest point to the sphere center
		b3Vec3W d = b3SubVW( centerW, closestPoint );
		b3FloatW distSqr = b3DotW( d, d );
		vox->accepted = b3AndW( b3GreaterThanW( distSqr, epsilonW ), b3LessThanW( distSqr, maxDistSqrW ) );
		if ( !b3AnyTrueW( vox->accepted ) )
			continue;

		// compute normal and closest point on sphere.
		// contact point is midpoint between closest points
		b3FloatW dist = b3SqrtW( distSqr );
		vox->normal = b3MulSVW( b3DivW( oneW, dist ), d );
		b3Vec3W closestPointSphere = b3SubVW( centerW, b3MulSVW( radiusW, vox->normal ) );

		// descale the point and compute separation
		vox->point = b3MulSVW( scale, b3MulSVW( halfW, b3AddVW( closestPoint, closestPointSphere ) ) );
		vox->separation = b3MulW( scale, b3SubW( dist, radiusW ) );
	}

	// Step 4: add a candidate point for all valid lanes
	for ( int i = 0; i < wideCount; i++ )
	{
		VoxelWide* vox = &wideVoxels[i];

		for ( int lane = 0; lane < B3_SIMD_WIDTH; lane++ )
		{
			if ( INT( vox->accepted, lane ) == 0 )
				continue;

			VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
			cp->point.x = FLT( vox->point.X, lane );
			cp->point.y = FLT( vox->point.Y, lane );
			cp->point.z = FLT( vox->point.Z, lane );

			cp->normal.x = FLT( vox->normal.X, lane );
			cp->normal.y = FLT( vox->normal.Y, lane );
			cp->normal.z = FLT( vox->normal.Z, lane );

			cp->separation = ( FLT( vox->separation, lane ) );
		}
	}
}

static uint32_t getNeighborMask( const b3Vec3 candidate, const b3Vec3 voxMin, const b3Vec3 voxMax )
{
	// this is effectively an AABB SAT test, resulting in a mask of the separating axes.
	// if a voxel has a neighbor along a separating axis, that neighbor is closer to the candidate point.
	// An extra bonus, this also filters out any contact points which would have a normal pointed into
	// a neighboring voxel, which is not a valid contact point for collision resolution.
	uint32_t neighborMask = 0;
	neighborMask |= candidate.x < voxMin.x ? b3_negXNeighbor : 0;
	neighborMask |= candidate.x >= voxMax.x ? b3_posXNeighbor : 0;
	neighborMask |= candidate.y < voxMin.y ? b3_negYNeighbor : 0;
	neighborMask |= candidate.y >= voxMax.y ? b3_posYNeighbor : 0;
	neighborMask |= candidate.z < voxMin.z ? b3_negZNeighbor : 0;
	neighborMask |= candidate.z >= voxMax.z ? b3_posZNeighbor : 0;
	return neighborMask;
}

static void collideVoxCapsule( VoxCollideContext* context, b3Transform bToA, b3Arena arena )
{
	// This will probably be used when we simd
	B3_UNUSED( arena );

	// get the center, radius, and speculative distance of the capsule in voxel space/scale
	const b3Voxels voxelsA = context->voxelsA;
	float invScale = 1.0f / voxelsA.scale;
	float specDist = B3_SPECULATIVE_DISTANCE * invScale;
	float radius = context->capsuleB->radius * invScale;
	float maxDistSqr = ( radius + specDist ) * ( radius + specDist );
	b3Vec3 center1 = b3MulSV( invScale, b3TransformPoint( bToA, context->capsuleB->center1 ) );
	b3Vec3 center2 = b3MulSV( invScale, b3TransformPoint( bToA, context->capsuleB->center2 ) );
	b3Vec3 dir = b3Sub( center2, center1 );

	// compute the query bounds for the voxel grid. This is the AABB of the capsule expanded by the speculative distance.
	b3Vec3 extent = b3Vec3Of( radius + specDist );
	b3Vec3 capsuleMin = b3Min( center1, center2 );
	b3Vec3 capsuleMax = b3Max( center1, center2 );
	b3AABB queryBounds = computeVoxelBounds( voxelsA.data, b3Sub( capsuleMin, extent ), b3Add( capsuleMax, extent ) );

	// refresh the voxel cache
	refreshVoxCache( context->contact, voxelsA.data, queryBounds );

	for ( int i = 0; i < context->contact->voxelCache.count; i++ )
	{
		b3Vec3 voxMin = context->contact->voxelCache.data[i].min;
		b3Vec3 voxMax = b3Add( voxMin, b3Vec3Of( 1.0f ) );
		uint32_t flags = context->contact->voxelCache.data[i].flags;

		// similar to the sphere case, we can skip voxels that are coplanar with nearer voxels
		// in this case, we're actually using the bounding box of the capsule segment, rather than a
		// single point, which is a conservative check that may miss some voxels that could be skipped.
		// However, this can prevent computing the closest point calculation below.
		uint32_t neighborMask = 0;
		neighborMask |= max( center1.x, center2.x ) < voxMin.x ? b3_negXNeighbor : 0;
		neighborMask |= min( center1.x, center2.x ) >= voxMax.x ? b3_posXNeighbor : 0;
		neighborMask |= max( center1.y, center2.y ) < voxMin.y ? b3_negYNeighbor : 0;
		neighborMask |= min( center1.y, center2.y ) >= voxMax.y ? b3_posYNeighbor : 0;
		neighborMask |= max( center1.z, center2.z ) < voxMin.z ? b3_negZNeighbor : 0;
		neighborMask |= min( center1.z, center2.z ) >= voxMax.z ? b3_posZNeighbor : 0;
		if ( ( flags & neighborMask ) != 0 )
			continue;

		// find the closest point between the capsule segment and the voxel bounding box
		float segLengthSqr = b3LengthSquared( dir );
		b3Vec3 clip1 = b3Clamp( center1, voxMin, voxMax );
		b3Vec3 clip2 = b3Clamp( center2, voxMin, voxMax );
		float distSqr1 = b3LengthSquared( b3Sub( center1, clip1 ) );
		float distSqr2 = b3LengthSquared( b3Sub( center2, clip2 ) );
		float t = b3ClampFloat( b3Dot( b3Sub( clip1, center1 ), dir ) / segLengthSqr, 0.0f, 1.0f );
		b3Vec3 segProj = b3Add( center1, b3MulSV( t, dir ) );
		b3Vec3 voxProj = b3Clamp( segProj, voxMin, voxMax );
		float dist3Sqr = b3LengthSquared( b3Sub( segProj, voxProj ) );

		b3Vec3 closestSeg, closestVox;
		float distSqr;
		if ( dist3Sqr <= distSqr1 && dist3Sqr <= distSqr2 )
		{
			closestSeg = segProj;
			closestVox = voxProj;
			distSqr = dist3Sqr;
		}
		else if ( distSqr1 <= distSqr2 )
		{
			closestSeg = center1;
			closestVox = clip1;
			distSqr = distSqr1;
		}
		else
		{
			closestSeg = center2;
			closestVox = clip2;
			distSqr = distSqr2;
		}

		// if closest point is too far from the voxel, skip this voxel
		if ( distSqr > maxDistSqr )
			continue;

		// detect if the capsule is near parallel with the reference face
		b3Vec3 delta = b3Sub( closestSeg, closestVox );
		b3Vec3 absDelta = b3Abs( delta );
		int maxAxis = absDelta.x > absDelta.y ? ( absDelta.x > absDelta.z ? 0 : 2 ) : ( absDelta.y > absDelta.z ? 1 : 2 );
		float maxDelta = FLT( absDelta, maxAxis );
		float epsSqr = distSqr * 0.998f * 0.998f;

		// check for multiple contact points due to near parallelism
		if ( maxDelta * maxDelta > epsSqr )
		{
			// get the normal of the reference face
			b3Vec3 faceNormal = (b3Vec3){ 0.0f, 0.0f, 0.0f };
			FLT( faceNormal, maxAxis ) = copysignf( 1.0f, FLT( delta, maxAxis ) );
			int altAxisA = ( maxAxis + 1 ) % 3;
			int altAxisB = ( maxAxis + 2 ) % 3;
			float t1 = 0.0f, t2 = 1.0f;

			// clip center1 to the reference face
			{
				float minA = FLT( voxMin, altAxisA );
				float maxA = FLT( voxMax, altAxisA );
				float vA1 = FLT( center1, altAxisA );
				float vA2 = FLT( center2, altAxisA );
				float dA = vA2 - vA1;
				if ( vA1 < minA )
				{
					float tn = ( minA - vA1 ) / dA;
					t1 = max( t1, tn );
				}
				else if ( vA1 > maxA )
				{
					float tn = ( vA1 - maxA ) / dA;
					t1 = max( t1, tn );
				}

				float minB = FLT( voxMin, altAxisB );
				float maxB = FLT( voxMax, altAxisB );
				float vB1 = FLT( center1, altAxisB );
				float vB2 = FLT( center2, altAxisB );
				float dB = vB2 - vB1;
				if ( vB1 < minB )
				{
					float tn = ( minB - vB1 ) / dB;
					t1 = max( t1, tn );
				}
				else if ( vB1 > maxB )
				{
					float tn = ( vB1 - maxB ) / dB;
					t1 = max( t1, tn );
				}

				b3Vec3 cp1 = b3Add( center1, b3MulSV( t1, dir ) );

				// do a neighbor test along the edge of the face this point was clipped to
				uint32_t neighborMask1 = 0;
				if ( t1 > 0.0f )
				{
					neighborMask1 |= cp1.x == voxMax.x ? b3_posXNeighbor : 0;
					neighborMask1 |= cp1.x == voxMin.x ? b3_negXNeighbor : 0;
					neighborMask1 |= cp1.y == voxMax.y ? b3_posYNeighbor : 0;
					neighborMask1 |= cp1.y == voxMin.y ? b3_negYNeighbor : 0;
					neighborMask1 |= cp1.z == voxMax.z ? b3_posZNeighbor : 0;
					neighborMask1 |= cp1.z == voxMin.z ? b3_negZNeighbor : 0;
				}

				if ( ( flags & neighborMask1 ) == 0 )
				{
					// add a candidate point for the contact
					b3Vec3 vp1 = b3Clamp( cp1, voxMin, voxMax );
					b3Vec3 d = b3Sub( cp1, vp1 );
					float distSqrn = b3LengthSquared( d );

					b3Vec3 capsulePt = b3Sub( cp1, b3MulSV( radius, faceNormal ) );
					b3Vec3 point = b3MulSV( 0.5f, b3Add( capsulePt, vp1 ) );

					VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
					cp->point = b3MulSV( voxelsA.scale, point );
					cp->normal = faceNormal;
					cp->separation = ( sqrtf( distSqrn ) - radius ) * voxelsA.scale;
				}
			}

			// clip center2 to the reference face
			{
				float minA = FLT( voxMin, altAxisA );
				float maxA = FLT( voxMax, altAxisA );
				float vA1 = FLT( center1, altAxisA );
				float vA2 = FLT( center2, altAxisA );
				float dA = vA2 - vA1;
				if ( vA2 < minA )
				{
					float tn = 1.0f - ( minA - vA2 ) / dA;
					t2 = min( t2, tn );
				}
				else if ( vA2 > maxA )
				{
					float tn = 1.0f - ( vA2 - maxA ) / dA;
					t2 = min( t2, tn );
				}

				float minB = FLT( voxMin, altAxisB );
				float maxB = FLT( voxMax, altAxisB );
				float vB1 = FLT( center1, altAxisB );
				float vB2 = FLT( center2, altAxisB );
				float dB = vB2 - vB1;
				if ( vB2 < minB )
				{
					float tn = 1.0f - ( minB - vB2 ) / dB;
					t2 = min( t2, tn );
				}
				else if ( vB2 > maxB )
				{
					float tn = 1.0f - ( vB2 - maxB ) / dB;
					t2 = min( t2, tn );
				}

				// if t1 and t2 are effectively the same, don't add a second contact point
				if ( fabsf( t2 - t1 ) < FLT_EPSILON )
				{
					continue;
				}

				b3Vec3 cp2 = b3Add( center1, b3MulSV( t2, dir ) );

				uint32_t neighborMask2 = 0;
				if ( t2 < 1.0f )
				{
					neighborMask2 |= cp2.x == voxMax.x ? b3_posXNeighbor : 0;
					neighborMask2 |= cp2.x == voxMin.x ? b3_negXNeighbor : 0;
					neighborMask2 |= cp2.y == voxMax.y ? b3_posYNeighbor : 0;
					neighborMask2 |= cp2.y == voxMin.y ? b3_negYNeighbor : 0;
					neighborMask2 |= cp2.z == voxMax.z ? b3_posZNeighbor : 0;
					neighborMask2 |= cp2.z == voxMin.z ? b3_negZNeighbor : 0;
				}

				if ( ( flags & neighborMask2 ) == 0 )
				{
					// add a candidate point for the contact
					b3Vec3 vp2 = b3Clamp( cp2, voxMin, voxMax );
					b3Vec3 d = b3Sub( cp2, vp2 );
					float distSqrn = b3LengthSquared( d );

					b3Vec3 capsulePt = b3Sub( cp2, b3MulSV( radius, faceNormal ) );
					b3Vec3 point = b3MulSV( 0.5f, b3Add( capsulePt, vp2 ) );

					VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
					cp->point = b3MulSV( voxelsA.scale, point );
					cp->normal = faceNormal;
					cp->separation = ( sqrtf( distSqrn ) - radius ) * voxelsA.scale;
				}
			}

			continue;
		}

		// if there's a closer neighbor voxel, skip
		neighborMask = getNeighborMask( closestSeg, voxMin, voxMax );
		if ( ( flags & neighborMask ) != 0 )
			continue;

		float dist = sqrtf( distSqr );
		b3Vec3 normal = b3MulSV( 1.0f / dist, delta );
		b3Vec3 closestCaps = b3Sub( closestSeg, b3MulSV( radius, normal ) );
		b3Vec3 point = b3MulSV( 0.5f, b3Add( closestCaps, closestVox ) );

		// add a candidate point for the contact
		VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
		cp->point = b3MulSV( voxelsA.scale, point );
		cp->normal = normal;
		cp->separation = ( dist - radius ) * voxelsA.scale;
	}
}

static inline bool voxFacesVsHullSAT( b3Vec3 voxMin, b3Vec3 voxMax, uint32_t flags, b3AABB hullAABB, float specDist,
									  float* bestSep, b3Vec3* bestNormal, int* normalAxis )
{
	// box face axis separation is just the distance between the voxel faces and the corresponding hull AABB faces.
	float sepNegX = voxMin.x - hullAABB.upperBound.x;
	float sepPosX = hullAABB.lowerBound.x - voxMax.x;
	bool hasNegXNeighbor = flags & b3_negXNeighbor;
	bool hasPosXNeighbor = flags & b3_posXNeighbor;
	float tolNegX = hasNegXNeighbor ? 0.0f : specDist;
	float tolPosX = hasPosXNeighbor ? 0.0f : specDist;
	if ( sepPosX > tolPosX || sepNegX > tolNegX )
		return true;

	float sepNegY = voxMin.y - hullAABB.upperBound.y;
	float sepPosY = hullAABB.lowerBound.y - voxMax.y;
	bool hasNegYNeighbor = flags & b3_negYNeighbor;
	bool hasPosYNeighbor = flags & b3_posYNeighbor;
	float tolPosY = hasPosYNeighbor ? 0.0f : specDist;
	float tolNegY = hasNegYNeighbor ? 0.0f : specDist;
	if ( sepPosY > tolPosY || sepNegY > tolNegY )
		return true;

	float sepNegZ = voxMin.z - hullAABB.upperBound.z;
	float sepPosZ = hullAABB.lowerBound.z - voxMax.z;
	bool hasNegZNeighbor = flags & b3_negZNeighbor;
	bool hasPosZNeighbor = flags & b3_posZNeighbor;
	float tolPosZ = hasPosZNeighbor ? 0.0f : specDist;
	float tolNegZ = hasNegZNeighbor ? 0.0f : specDist;
	if ( sepPosZ > tolPosZ || sepNegZ > tolNegZ )
		return true;

	if ( !hasNegXNeighbor && sepNegX > *bestSep )
	{
		*bestSep = sepNegX;
		*bestNormal = (b3Vec3){ -1.0f, 0.0f, 0.0f };
		*normalAxis = 0;
	}
	if ( !hasPosXNeighbor && sepPosX > *bestSep )
	{
		*bestSep = sepPosX;
		*bestNormal = (b3Vec3){ 1.0f, 0.0f, 0.0f };
		*normalAxis = 0;
	}
	if ( !hasNegYNeighbor && sepNegY > *bestSep )
	{
		*bestSep = sepNegY;
		*bestNormal = (b3Vec3){ 0.0f, -1.0f, 0.0f };
		*normalAxis = 1;
	}
	if ( !hasPosYNeighbor && sepPosY > *bestSep )
	{
		*bestSep = sepPosY;
		*bestNormal = (b3Vec3){ 0.0f, 1.0f, 0.0f };
		*normalAxis = 1;
	}
	if ( !hasNegZNeighbor && sepNegZ > *bestSep )
	{
		*bestSep = sepNegZ;
		*bestNormal = (b3Vec3){ 0.0f, 0.0f, -1.0f };
		*normalAxis = 2;
	}
	if ( !hasPosZNeighbor && sepPosZ > *bestSep )
	{
		*bestSep = sepPosZ;
		*bestNormal = (b3Vec3){ 0.0f, 0.0f, 1.0f };
		*normalAxis = 2;
	}

	return false;
}

static inline bool hullFacesVsVoxSAT( b3Vec3 voxCenter, b3Vec3* supportDirs, b3Vec3* supportOffsets, uint32_t flags,
									  const b3Plane* hullPlanes, int count, float* bestSep, int* bestFaceIndex )
{
	for ( int i = 0; i < count; i++ )
	{
		b3Plane face = hullPlanes[i];

		// get OBB support
		b3Vec3 support = b3Add( voxCenter, supportOffsets[i] );

		// compute plane separation
		float sep = b3Dot( face.normal, support ) - face.offset;
		if ( sep > B3_SPECULATIVE_DISTANCE )
			return true;

		// check flags to see if this support face should be a candidate for MTV
		b3Vec3 supportDir = supportDirs[i];
		b3Vec3 absDir = b3Abs( supportDir );
		int supportAxis = absDir.x > absDir.y ? ( absDir.x > absDir.z ? 0 : 2 ) : ( absDir.y > absDir.z ? 1 : 2 );
		uint32_t supportMask = FLT(supportDir, supportAxis) >= 0.0f ? posAxisNeighbors[supportAxis] : negAxisNeighbors[supportAxis];
		if ( ( flags & supportMask ) == 0 && sep > *bestSep )
		{
			*bestSep = sep;
			*bestFaceIndex = i;
		}
	}

	return false;
}

static inline bool voxEdgesVsHullSAT();

// copy of getSupportWide from convex_manifold.c
#define B3_HULL_BIT_COUNT 7
static inline int b3GetSupportWide( b3Vec3 normal, const float* vx, const float* vy, const float* vz, int n, float bias )
{
	const b3FloatW nx = b3SplatW( normal.x );
	const b3FloatW ny = b3SplatW( normal.y );
	const b3FloatW nz = b3SplatW( normal.z );
	const b3FloatW biasV = b3SplatW( bias );

	// Start the minimum at a large value.
	b3FloatW minValue = b3SplatW( B3_HUGE );

	// Tail lanes hold vertex 0 with index bits >= vertexCount, so they never become the min value.
	for ( int i = 0; i < n; i += 4 )
	{
		b3FloatW x = b3LoadW( vx + i );
		b3FloatW y = b3LoadW( vy + i );
		b3FloatW z = b3LoadW( vz + i );
		b3FloatW d = b3AddW( b3MulW( nz, z ), b3AddW( b3MulW( ny, y ), b3MulW( nx, x ) ) );

		// This is always positive.
		b3FloatW value = b3SubW( biasV, d );
		b3FloatW augmentedValue = b3EmbedIndexW( value, i, B3_HULL_BIT_COUNT );
		minValue = b3MinW( minValue, augmentedValue );
	}

	// One horizontal min, the winning lane's value and index bits ride through.
	return b3MinIndexW( minValue, B3_HULL_BIT_COUNT );
}

static void collideVoxHull( VoxCollideContext* context, b3Transform bToA, b3Arena arena )
{
	b3Matrix3 bToAMat = b3MakeMatrixFromQuat( bToA.q );
	b3Matrix3 aToBMat = b3Transpose( bToAMat );

	const b3Voxels voxelsA = context->voxelsA;
	float invScale = 1.0f / voxelsA.scale;
	float specDist = B3_SPECULATIVE_DISTANCE * invScale;
	b3Vec3 hullCenter = b3MulSV( invScale, transformPointMat( bToAMat, bToA.p, context->hullB->center ) );

	// compute the query bounds for the voxel grid. This is the AABB of the hull expanded by the speculative distance.
	b3AABB hullAABB = b3ComputeHullAABB( context->hullB, bToA );
	hullAABB.lowerBound = b3MulSV( invScale, hullAABB.lowerBound );
	hullAABB.upperBound = b3MulSV( invScale, hullAABB.upperBound );
	b3AABB inflatedHullAABB = b3AABB_Inflate( hullAABB, specDist );
	b3AABB queryBounds = computeVoxelBounds( voxelsA.data, inflatedHullAABB.lowerBound, inflatedHullAABB.upperBound );

	// refresh the voxel cache
	refreshVoxCache( context->contact, voxelsA.data, queryBounds );

	// early exit if no voxels were in the query bounds
	if ( context->contact->voxelCache.count == 0 )
		return;

	// hull points in voxel space
	b3Vec3* hullPoints = b3Bump( &arena, context->hullB->vertexCount * sizeof( b3Vec3 ) );
	// hull planes in hull space
	const b3Plane* hullPlanes = b3GetHullPlanes( context->hullB );
	// hull faces
	const b3HullFace* hullFaces = b3GetHullFaces( context->hullB );
	// hull edges
	const b3HullHalfEdge* hullEdges = b3GetHullEdges( context->hullB );

	{ // transform hull points to voxel space
		const b3Vec3* pts = b3GetHullPoints( context->hullB );
		for ( int i = 0; i < context->hullB->vertexCount; i++ )
		{
			hullPoints[i] = b3MulSV( invScale, transformPointMat( bToAMat, bToA.p, pts[i] ) );
		}
	}

	// cache hull incedent face indices for each voxel face normal direction
	int hullIncFaceIndices[6] = { -1, -1, -1, -1, -1, -1 };

	// cache voxel supports per hull face
	b3Vec3* voxSupportDirs = b3Bump( &arena, context->hullB->faceCount * sizeof( b3Vec3 ) );
	b3Vec3* voxSupportOffsets = b3Bump( &arena, context->hullB->faceCount * sizeof( b3Vec3 ) );
	float voxHalfExtent = 0.5f * context->voxelsA.scale;
	for ( int i = 0; i < context->hullB->faceCount; i++ )
	{
		voxSupportDirs[i] = b3Neg( b3MulMV( bToAMat, hullPlanes[i].normal ) );
		b3Vec3 corner = { copysignf( voxHalfExtent, voxSupportDirs[i].x ), copysignf( voxHalfExtent, voxSupportDirs[i].y ),
						  copysignf( voxHalfExtent, voxSupportDirs[i].z ) };
		voxSupportOffsets[i] = b3MulMV( aToBMat, corner );
	}

	for ( int v = 0; v < context->contact->voxelCache.count; v++ )
	{
		uint32_t flags = context->contact->voxelCache.data[v].flags;
		b3Vec3 voxMin = context->contact->voxelCache.data[v].min;
		b3Vec3 voxCenter = b3Add( voxMin, b3Vec3Of( 0.5f ) );
		b3Vec3 cDir = b3Sub( hullCenter, voxCenter );

		// cull voxels that are only back faces relative to the hull center
		// (i.e., all exposed faces of the voxel are facing away from the hull)
		uint32_t dirMask = 0;
		dirMask |= cDir.x < 0 ? b3_negXNeighbor : b3_posXNeighbor;
		dirMask |= cDir.y < 0 ? b3_negYNeighbor : b3_posYNeighbor;
		dirMask |= cDir.z < 0 ? b3_negZNeighbor : b3_posZNeighbor;
		if ( ( flags & dirMask ) == dirMask )
			continue;

		b3Vec3 voxMax = b3Add( voxMin, b3Vec3Of( 1.0f ) );

		// voxel center in hull space
		b3Vec3 vcInB = invTransformPointMat( aToBMat, bToA.p, b3MulSV( voxelsA.scale, voxCenter ) );

		// test voxel face axes against hull, skip voxel if separated
		float bestSepA = -FLT_MAX;
		b3Vec3 bestNormalA = { 0 };
		int normalAxisA = -1;
		if ( voxFacesVsHullSAT( voxMin, voxMax, flags, hullAABB, specDist, &bestSepA, &bestNormalA, &normalAxisA ) )
			continue;

		if ( ( flags & b3_isFaceVoxel ) == b3_isFaceVoxel )
		{
			// face voxels have no edges or corners exposed, so we only care about penetration along face normals.
			// We don't care if there is edge or corner penetration, since the edges and corners of this voxel are non-structural.
			// This is safe, because a face voxel by definition has neighbor voxels, which will either be penetrated by
			// a hull vertex, or have an edge or corner that penetrates the hull. So, any voxel edge or voxel corner penetration
			// will be covered by a nearby voxel's structural features. This also avoids ghost collisions on voxel boundaries.
			// It could be argued that checking the other axes could allow for early outs, but when we build the contacts for
			// a face voxel, we only add hull vertices within the bounds of the exposed face. Worst case is bounded by the number
			// of vertices on the hull's incident face, and the checks for each vertex are exclusively less-than operations.
			// Compare that to checking all hull faces (AND edges), which is by definition more numerous than the number of
			// vertices of a single face, and involve dot products, cross products, and other heavy math. Face voxels will
			// typically be the most common type of voxel, and we cache the incident face per normal direction to amortize the
			// cost across voxels, so face voxels are drastically cheaper than a full generic SAT test per voxel.

			// if there was no exposed face plane with penetration, exit early
			if ( normalAxisA == -1 )
				continue;

			int altAxis0 = ( normalAxisA + 1 ) % 3;
			int altAxis1 = ( normalAxisA + 2 ) % 3;

			// get or set cached incident face index for the voxel face normal direction
			int cacheKey = ( normalAxisA << 1 ) | ( FLT( bestNormalA, normalAxisA ) < 0 ? 1 : 0 );
			if ( hullIncFaceIndices[cacheKey] == -1 )
			{
				int soaVertexCountB = ( context->hullB->vertexCount + 3 ) & ~3;
				const float* vxB = b3GetHullSoaVertices( context->hullB );
				const float* vyB = vxB + soaVertexCountB;
				const float* vzB = vyB + soaVertexCountB;

				b3Vec3 cB = b3AABB_Center( context->hullB->aabb );
				b3Vec3 hB = b3AABB_Extents( context->hullB->aabb );

				b3Vec3 normalInB = b3MulMV( aToBMat, bestNormalA );
				b3Vec3 direction = b3Neg( normalInB );
				float biasB = b3Dot( direction, cB ) + 1.0625f * b3Dot( b3Abs( direction ), hB );
				int supportIndex = b3GetSupportWide( b3Neg( normalInB ), vxB, vyB, vzB, soaVertexCountB, biasB );
				hullIncFaceIndices[cacheKey] = b3FindIncidentFace( context->hullB, normalInB, supportIndex );
			}

			// iterate face vertices to find candidate contact points
			int incFaceIndex = hullIncFaceIndices[cacheKey];
			const b3HullFace* incFace = &hullFaces[incFaceIndex];
			int edgeIndex = incFace->edge;
			do
			{
				const b3HullHalfEdge* edge = &hullEdges[edgeIndex];
				edgeIndex = edge->next;

				b3Vec3 pt = hullPoints[edge->origin];
				if ( FLT( pt, altAxis0 ) < FLT( voxMin, altAxis0 ) || FLT( pt, altAxis0 ) >= FLT( voxMax, altAxis0 ) ||
					 FLT( pt, altAxis1 ) < FLT( voxMin, altAxis1 ) || FLT( pt, altAxis1 ) >= FLT( voxMax, altAxis1 ) )
				{
					continue;
				}

				float vh = FLT( pt, normalAxisA );
				float vv = FLT( bestNormalA, normalAxisA ) < 0 ? FLT( voxMin, normalAxisA ) : FLT( voxMax, normalAxisA );
				float sep = FLT( bestNormalA, normalAxisA ) < 0 ? vv - vh : vh - vv;
				FLT( pt, normalAxisA ) = ( vh + vv ) * 0.5f;

				// create a contact point for this hull vertex inside the voxel
				VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
				cp->point = b3MulSV( voxelsA.scale, pt );
				cp->normal = bestNormalA;
				cp->separation = sep * voxelsA.scale;
			}
			while ( edgeIndex != incFace->edge );

			continue;
		}

		// check hull face normals against voxel vertices, skip voxel if separated
		// TODO: can we skip this for edge voxels, by the same logic we use on face voxels?
		float bestSepB = -FLT_MAX;
		int bestFaceIndexB = -1;
		if ( hullFacesVsVoxSAT( vcInB, voxSupportDirs, voxSupportOffsets, flags, hullPlanes, context->hullB->faceCount, &bestSepB,
								&bestFaceIndexB ) )
		{
			continue;
		}

		// TODO: edge-edge contacts
		// TODO: we can cache edge cross products per edge pair to amortize the cost across many voxels
		float bestSepE = -FLT_MAX;

		// scale bestSepB to be in same units as bestSepA/bestSepE
		bestSepB = bestSepB * voxelsA.scale;

		if ( bestSepA > bestSepB && bestSepA > bestSepE )
		{
			// alternate axes for the current face normal
			int altAxis0 = ( normalAxisA + 1 ) % 3;
			int altAxis1 = ( normalAxisA + 2 ) % 3;

			bool canClipNegAxis0 = !( flags & negAxisNeighbors[altAxis0] );
			bool canClipPosAxis0 = !( flags & posAxisNeighbors[altAxis0] );
			bool canClipNegAxis1 = !( flags & negAxisNeighbors[altAxis1] );
			bool canClipPosAxis1 = !( flags & posAxisNeighbors[altAxis1] );

			// get or set cached incident face index for the voxel face normal direction
			int cacheKey = ( normalAxisA << 1 ) | ( FLT( bestNormalA, normalAxisA ) < 0 ? 1 : 0 );
			if ( hullIncFaceIndices[cacheKey] == -1 )
			{
				int soaVertexCountB = ( context->hullB->vertexCount + 3 ) & ~3;
				const float* vxB = b3GetHullSoaVertices( context->hullB );
				const float* vyB = vxB + soaVertexCountB;
				const float* vzB = vyB + soaVertexCountB;

				b3Vec3 cB = b3AABB_Center( context->hullB->aabb );
				b3Vec3 hB = b3AABB_Extents( context->hullB->aabb );

				b3Vec3 normalInB = b3MulMV( aToBMat, bestNormalA );
				b3Vec3 direction = b3Neg( normalInB );
				float biasB = b3Dot( direction, cB ) + 1.0625f * b3Dot( b3Abs( direction ), hB );
				int supportIndex = b3GetSupportWide( b3Neg( normalInB ), vxB, vyB, vzB, soaVertexCountB, biasB );
				hullIncFaceIndices[cacheKey] = b3FindIncidentFace( context->hullB, normalInB, supportIndex );
			}

			b3Vec3 polyBufferA[B3_MAX_CLIP_POINTS];
			b3Vec3 polyBufferB[B3_MAX_CLIP_POINTS];

			// build face polygon
			b3Vec3* srcPoly = polyBufferA;
			int srcPolyCount = 0;

			int incFaceIndex = hullIncFaceIndices[cacheKey];
			const b3HullFace* incFace = &hullFaces[incFaceIndex];

			int edgeIndex = incFace->edge;
			do
			{
				const b3HullHalfEdge* edge = &hullEdges[edgeIndex];
				srcPoly[srcPolyCount++] = hullPoints[edge->origin];
				edgeIndex = edge->next;
			}
			while ( edgeIndex != incFace->edge );

			b3Vec3* dstPoly = polyBufferB;
			int dstPolyCount = 0;

			{ // clip polygon against clippable axes
				if ( canClipNegAxis0 )
				{
					b3Vec3 p0 = srcPoly[srcPolyCount - 1];
					float sep0 = FLT( voxMin, altAxis0 ) - FLT( p0, altAxis0 );
					for ( int i = 0; i < srcPolyCount; i++ )
					{
						b3Vec3 p1 = srcPoly[i];
						float sep1 = FLT( voxMin, altAxis0 ) - FLT( p1, altAxis0 );
						if ( sep0 <= 0.0f && sep1 <= 0.0f )
						{
							// both points are inside the negative axis 0 clipping plane, keep the current point
							dstPoly[dstPolyCount++] = p1;
						}
						else if ( sep0 <= 0.0f && sep1 > 0.0f )
						{
							// edge goes from inside to outside, keep intersection point
							float t = sep0 / ( sep0 - sep1 );
							dstPoly[dstPolyCount++] = b3Lerp( p0, p1, t );
						}
						else if ( sep0 > 0.0f && sep1 <= 0.0f )
						{
							// edge goes from outside to inside, keep intersection and current point
							float t = sep0 / ( sep0 - sep1 );
							dstPoly[dstPolyCount++] = b3Lerp( p0, p1, t );
							dstPoly[dstPolyCount++] = p1;
						}

						p0 = p1;
						sep0 = sep1;
					}

					B3_SWAP( srcPoly, dstPoly );
					srcPolyCount = dstPolyCount;
					dstPolyCount = 0;
				}
				if ( canClipNegAxis1 )
				{
					b3Vec3 p0 = srcPoly[srcPolyCount - 1];
					float sep0 = FLT( voxMin, altAxis1 ) - FLT( p0, altAxis1 );
					for ( int i = 0; i < srcPolyCount; i++ )
					{
						b3Vec3 p1 = srcPoly[i];
						float sep1 = FLT( voxMin, altAxis1 ) - FLT( p1, altAxis1 );
						if ( sep0 <= 0.0f && sep1 <= 0.0f )
						{
							// both points are inside the negative axis 1 clipping plane, keep the current point
							dstPoly[dstPolyCount++] = p1;
						}
						else if ( sep0 <= 0.0f && sep1 > 0.0f )
						{
							// edge goes from inside to outside, keep intersection point
							float t = sep0 / ( sep0 - sep1 );
							dstPoly[dstPolyCount++] = b3Lerp( p0, p1, t );
						}
						else if ( sep0 > 0.0f && sep1 <= 0.0f )
						{
							// edge goes from outside to inside, keep intersection and current point
							float t = sep0 / ( sep0 - sep1 );
							dstPoly[dstPolyCount++] = b3Lerp( p0, p1, t );
							dstPoly[dstPolyCount++] = p1;
						}

						p0 = p1;
						sep0 = sep1;
					}

					B3_SWAP( srcPoly, dstPoly );
					srcPolyCount = dstPolyCount;
					dstPolyCount = 0;
				}
				if ( canClipPosAxis0 )
				{
					b3Vec3 p0 = srcPoly[srcPolyCount - 1];
					float sep0 = FLT( p0, altAxis0 ) - FLT( voxMax, altAxis0 );
					for ( int i = 0; i < srcPolyCount; i++ )
					{
						b3Vec3 p1 = srcPoly[i];
						float sep1 = FLT( p1, altAxis0 ) - FLT( voxMax, altAxis0 );
						if ( sep0 <= 0.0f && sep1 <= 0.0f )
						{
							// both points are inside the positive axis 0 clipping plane, keep the current point
							dstPoly[dstPolyCount++] = srcPoly[i];
						}
						else if ( sep0 <= 0.0f && sep1 > 0.0f )
						{
							// edge goes from inside to outside, keep intersection point
							float t = sep0 / ( sep0 - sep1 );
							dstPoly[dstPolyCount++] = b3Lerp( p0, p1, t );
						}
						else if ( sep0 > 0.0f && sep1 <= 0.0f )
						{
							// edge goes from outside to inside, keep intersection and current point
							float t = sep0 / ( sep0 - sep1 );
							dstPoly[dstPolyCount++] = b3Lerp( p0, p1, t );
							dstPoly[dstPolyCount++] = p1;
						}

						p0 = p1;
						sep0 = sep1;
					}

					B3_SWAP( srcPoly, dstPoly );
					srcPolyCount = dstPolyCount;
					dstPolyCount = 0;
				}
				if ( canClipPosAxis1 )
				{
					b3Vec3 p0 = srcPoly[srcPolyCount - 1];
					float sep0 = FLT( p0, altAxis1 ) - FLT( voxMax, altAxis1 );
					for ( int i = 0; i < srcPolyCount; i++ )
					{
						b3Vec3 p1 = srcPoly[i];
						float sep1 = FLT( p1, altAxis1 ) - FLT( voxMax, altAxis1 );
						if ( sep0 <= 0.0f && sep1 <= 0.0f )
						{
							// both points are inside the positive axis 1 clipping plane, keep the current point
							dstPoly[dstPolyCount++] = p1;
						}
						else if ( sep0 <= 0.0f && sep1 > 0.0f )
						{
							// edge goes from inside to outside, keep intersection point
							float t = sep0 / ( sep0 - sep1 );
							dstPoly[dstPolyCount++] = b3Lerp( p0, p1, t );
						}
						else if ( sep0 > 0.0f && sep1 <= 0.0f )
						{
							// edge goes from outside to inside, keep intersection and current point
							float t = sep0 / ( sep0 - sep1 );
							dstPoly[dstPolyCount++] = b3Lerp( p0, p1, t );
							dstPoly[dstPolyCount++] = p1;
						}

						p0 = p1;
						sep0 = sep1;
					}

					B3_SWAP( srcPoly, dstPoly );
					srcPolyCount = dstPolyCount;
					dstPolyCount = 0;
				}
			}
			{ // cull points outside of unclippable axes
				if ( !canClipNegAxis0 )
				{
					int j = 0;
					for ( int i = 0; i < srcPolyCount; i++ )
					{
						if ( FLT( srcPoly[i], altAxis0 ) >= FLT( voxMin, altAxis0 ) )
						{
							srcPoly[j++] = srcPoly[i];
						}
					}
					srcPolyCount = j;
				}
				if ( !canClipNegAxis1 )
				{
					int j = 0;
					for ( int i = 0; i < srcPolyCount; i++ )
					{
						if ( FLT( srcPoly[i], altAxis1 ) >= FLT( voxMin, altAxis1 ) )
						{
							srcPoly[j++] = srcPoly[i];
						}
					}
					srcPolyCount = j;
				}
				if ( !canClipPosAxis0 )
				{
					int j = 0;
					for ( int i = 0; i < srcPolyCount; i++ )
					{
						if ( FLT( srcPoly[i], altAxis0 ) < FLT( voxMax, altAxis0 ) )
						{
							srcPoly[j++] = srcPoly[i];
						}
					}
					srcPolyCount = j;
				}
				if ( !canClipPosAxis1 )
				{
					int j = 0;
					for ( int i = 0; i < srcPolyCount; i++ )
					{
						if ( FLT( srcPoly[i], altAxis1 ) < FLT( voxMax, altAxis1 ) )
						{
							srcPoly[j++] = srcPoly[i];
						}
					}
					srcPolyCount = j;
				}
			}

			// add contacts for each point that survived clipping
			for ( int i = 0; i < srcPolyCount; i++ )
			{
				b3Vec3 pt = srcPoly[i];
				float vh = FLT( pt, normalAxisA );
				float vv = FLT( bestNormalA, normalAxisA ) < 0 ? FLT( voxMin, normalAxisA ) : FLT( voxMax, normalAxisA );
				float sep = FLT( bestNormalA, normalAxisA ) < 0 ? vv - vh : vh - vv;
				FLT( pt, normalAxisA ) = ( vh + vv ) * 0.5f;

				// create a contact point for this vertex inside the voxel
				VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
				cp->point = b3MulSV( voxelsA.scale, pt );
				cp->normal = bestNormalA;
				cp->separation = sep * voxelsA.scale;
			}
		}
		else if ( bestSepB > bestSepE )
		{
			const b3Vec3* hullpts = b3GetHullPoints( context->hullB );
			const b3HullFace* faceB = &hullFaces[bestFaceIndexB];
			const b3Plane* planeB = &hullPlanes[bestFaceIndexB];
			b3Vec3 dir = voxSupportDirs[bestFaceIndexB];
			b3Vec3 absDir = b3Abs( dir );

			// get ref face on voxel
			int normalAxis = absDir.x > absDir.y ? ( absDir.x > absDir.z ? 0 : 2 ) : ( absDir.y > absDir.z ? 1 : 2 );
			int altAxis0 = ( normalAxis + 1 ) % 3;
			int altAxis1 = ( normalAxis + 2 ) % 3;

			bool edgeCanCollide[5] = {
				!( flags & posAxisNeighbors[altAxis0] ),
				!( flags & negAxisNeighbors[altAxis1] ),
				!( flags & negAxisNeighbors[altAxis0] ),
				!( flags & posAxisNeighbors[altAxis1] ),
				true, // sentinel for edges created during clipping, which are not part of the original face
			};

			b3Vec3 normalOffset = b3MulSV( copysignf( voxHalfExtent, FLT( dir, normalAxis ) ), COLUMN( aToBMat, normalAxis ) );
			b3Vec3 altOffset0 = b3MulSV( voxHalfExtent, COLUMN( aToBMat, altAxis0 ) );
			b3Vec3 altOffset1 = b3MulSV( voxHalfExtent, COLUMN( aToBMat, altAxis1 ) );

			b3Vec3 polyBufferA[B3_MAX_CLIP_POINTS];
			b3Vec3 polyBufferB[B3_MAX_CLIP_POINTS];
			int edgeBufferA[B3_MAX_CLIP_POINTS];
			int edgeBufferB[B3_MAX_CLIP_POINTS];

			// create clip polygon for the reference face on the voxel
			b3Vec3 faceCenter = b3Add( vcInB, normalOffset );
			b3Vec3* srcPoly = polyBufferA;
			int* srcEdges = edgeBufferA;
			int srcPolyCount = 4;
			srcPoly[0] = b3Add( faceCenter, b3Add( altOffset0, altOffset1 ) );
			srcPoly[1] = b3Add( faceCenter, b3Add( altOffset0, b3Neg( altOffset1 ) ) );
			srcPoly[2] = b3Add( faceCenter, b3Add( b3Neg( altOffset0 ), b3Neg( altOffset1 ) ) );
			srcPoly[3] = b3Add( faceCenter, b3Add( b3Neg( altOffset0 ), altOffset1 ) );
			srcEdges[0] = 0;
			srcEdges[1] = 1;
			srcEdges[2] = 2;
			srcEdges[3] = 3;

			b3Vec3* dstPoly = polyBufferB;
			int* dstEdges = edgeBufferB;
			int dstPolyCount = 0;

			int edgeIndex = faceB->edge;
			do
			{
				// get clipping plane
				const b3HullHalfEdge* edge = hullEdges + edgeIndex;
				int nextEdgeIndex = edge->next;
				const b3HullHalfEdge* next = hullEdges + nextEdgeIndex;
				b3Vec3 vertex1 = hullpts[edge->origin];
				b3Vec3 vertex2 = hullpts[next->origin];
				b3Vec3 tangent = b3Normalize( b3Sub( vertex2, vertex1 ) );
				b3Vec3 binormal = b3Cross( tangent, planeB->normal );
				b3Plane clipPlane = b3MakePlaneFromNormalAndPoint( binormal, vertex1 );

				b3Vec3 p0 = srcPoly[srcPolyCount - 1];
				int e0 = srcEdges[srcPolyCount - 1];
				float sep0 = b3PlaneSeparation( clipPlane, p0 );
				for ( int i = 0; i < srcPolyCount; i++ )
				{
					b3Vec3 p1 = srcPoly[i];
					int e1 = srcEdges[i];

					float sep1 = b3PlaneSeparation( clipPlane, p1 );
					if ( sep0 <= 0.0f && sep1 <= 0.0f )
					{
						// both points are inside the clipping plane, keep the current point
						dstPoly[dstPolyCount] = p1;
						dstEdges[dstPolyCount] = e1;
						dstPolyCount++;
					}
					else if ( sep0 <= 0.0f && sep1 > 0.0f )
					{
						// edge goes from inside to outside, keep intersection point
						b3Vec3 intersection = b3Lerp( p0, p1, sep0 / ( sep0 - sep1 ) );
						dstPoly[dstPolyCount] = intersection;
						dstEdges[dstPolyCount] = 4; // dropping p1, so new edge
						dstPolyCount++;
					}
					else if ( sep0 > 0.0f && sep1 <= 0.0f )
					{
						// edge goes from outside to inside, keep intersection and current point
						b3Vec3 intersection = b3Lerp( p0, p1, sep0 / ( sep0 - sep1 ) );
						dstPoly[dstPolyCount] = intersection;
						dstEdges[dstPolyCount] = e0;
						dstPolyCount++;
						dstPoly[dstPolyCount] = p1;
						dstEdges[dstPolyCount] = e1;
						dstPolyCount++;
					}

					p0 = p1;
					e0 = e1;
					sep0 = sep1;
				}

				// swap buffers
				B3_SWAP( srcPoly, dstPoly );
				B3_SWAP( srcEdges, dstEdges );
				srcPolyCount = dstPolyCount;
				dstPolyCount = 0;

				edgeIndex = edge->next;
			}
			while ( edgeIndex != faceB->edge );

			// add contacts for each point that survived clipping
			for ( int i = 0, j = srcPolyCount - 1; i < srcPolyCount; j = i, i++ )
			{
				// if the point lies along a non-structural edge, cull it
				int incomingEdge = srcEdges[j];
				int outgoingEdge = srcEdges[i];
				if ( !edgeCanCollide[incomingEdge] || !edgeCanCollide[outgoingEdge] )
					continue;

				b3Vec3 point = srcPoly[i];
				float sep = b3PlaneSeparation( *planeB, point );
				b3Vec3 pt = transformPointMat( bToAMat, bToA.p, point );
				b3Vec3 normal = b3MulMV( bToAMat, planeB->normal );

				VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
				cp->point = pt;
				cp->normal = b3Neg( normal );
				cp->separation = sep;
			}
		}
		else
		{
			// TODO: handle edge-edge contact
		}
	}
}

static void collideVoxVox( VoxCollideContext* context, b3Transform bToA, b3Arena arena )
{
	printf( "collideVoxVox not implemented\n" );
}

typedef struct VoxCluster
{
	b3Vec3 normal;
	b3LocalManifoldPoint* points;
	int capacity;
	int count;
} VoxCluster;

bool b3ComputeVoxelManifolds( b3World* world, int workerIndex, b3Contact* contact, const b3Shape* shapeA, b3WorldTransform xfA,
							  const b3Shape* shapeB, b3WorldTransform xfB, b3Arena arena )
{
	B3_ASSERT( shapeA->type == b3_voxelShape );
	B3_ASSERT( shapeB->type == b3_voxelShape || shapeB->type == b3_sphereShape || shapeB->type == b3_capsuleShape ||
			   shapeB->type == b3_hullShape );
	B3_UNUSED( workerIndex );

	uint64_t ticks = b3GetTicks();

	VoxCollideContext context = { 0 };
	context.voxelsA = shapeA->voxels;
	context.contact = &contact->voxelContact;
	context.pointBuffer = b3Bump( &arena, POINT_BUFFER_CAPACITY * sizeof( VoxCandidatePoint ) );

	b3Transform transformBtoA = b3InvMulWorldTransforms( xfA, xfB );

	if ( shapeB->type == b3_sphereShape )
	{
		context.sphereB = &shapeB->sphere;
		collideVoxSphereW( &context, transformBtoA, arena );
	}
	else if ( shapeB->type == b3_capsuleShape )
	{
		context.capsuleB = &shapeB->capsule;
		collideVoxCapsule( &context, transformBtoA, arena );
	}
	else if ( shapeB->type == b3_hullShape )
	{
		context.hullB = shapeB->hull;
		collideVoxHull( &context, transformBtoA, arena );
	}
	else
	{
		B3_ASSERT( shapeB->type == b3_voxelShape );
		context.voxelsB = shapeB->voxels;
		collideVoxVox( &context, transformBtoA, arena );
	}

	if ( context.pointCount == 0 )
	{
		if ( contact->manifoldCount > 0 )
		{
			b3FreeManifolds( world, contact->manifolds, contact->manifoldCount );
			contact->manifolds = NULL;
			contact->manifoldCount = 0;
		}

		return false;
	}

	float collideMs = b3GetMilliseconds( ticks );

	// Cluster the manifold points by normal direction.
	const float clusterThreshold = 0.996f;
	VoxCluster* clusters = b3Bump( &arena, context.pointCount * sizeof( VoxCluster ) );
	int* clusterMemberships = b3Bump( &arena, context.pointCount * sizeof( int ) );
	int clusterCount = 0;
	for ( int i = 0; i < context.pointCount; i++ )
	{
		clusterMemberships[i] = B3_NULL_INDEX;
		VoxCandidatePoint* cp = context.pointBuffer + i;
		for ( int j = 0; j < clusterCount; j++ )
		{
			VoxCluster* cluster = clusters + j;
			if ( b3Dot( cp->normal, cluster->normal ) > clusterThreshold )
			{
				clusterMemberships[i] = j;
				cluster->capacity++;
				break;
			}
		}

		if ( clusterMemberships[i] != B3_NULL_INDEX )
			continue;

		VoxCluster* cluster = clusters + clusterCount;
		cluster->normal = cp->normal;
		clusterMemberships[i] = clusterCount;
		cluster->capacity = 1;
		cluster->count = 0;
		clusterCount++;
	}

	// Initialize the clusters
	b3LocalManifoldPoint* clusterPoints = b3Bump( &arena, context.pointCount * sizeof( b3LocalManifoldPoint ) );
	for ( int i = 0, j = 0; i < clusterCount; i++ )
	{
		VoxCluster* cluster = clusters + i;
		cluster->points = clusterPoints + j;
		j += cluster->capacity;
	}

	// Clone candidate points into the clusters
	for ( int i = 0; i < context.pointCount; i++ )
	{
		VoxCandidatePoint* cp = context.pointBuffer + i;
		int clusterIdx = clusterMemberships[i];
		VoxCluster* cluster = clusters + clusterIdx;
		b3LocalManifoldPoint* dstPt = cluster->points + cluster->count++;
		dstPt->point = cp->point;
		dstPt->separation = cp->separation;
	}

	// Reduce clusters
	for ( int i = 0; i < clusterCount; i++ )
	{
		VoxCluster* cluster = clusters + i;
		cluster->count = b3ReduceCluster( cluster->points, cluster->count, cluster->normal, arena );
	}

	// Make a temporary copy of previous manifolds
	int oldManifoldCount = contact->manifoldCount;
	b3Manifold* oldManifolds = NULL;
	if ( oldManifoldCount > 0 )
	{
		oldManifolds = b3Bump( &arena, oldManifoldCount * sizeof( b3Manifold ) );
		memcpy( oldManifolds, contact->manifolds, oldManifoldCount * sizeof( b3Manifold ) );
	}

	// Resize manifolds if needed
	if ( oldManifoldCount != clusterCount )
	{
		b3FreeManifolds( world, contact->manifolds, contact->manifoldCount );
		contact->manifolds = b3AllocateManifolds( world, clusterCount );
		contact->manifoldCount = (uint16_t)clusterCount;
	}
	else
	{
		// Mem zero manifolds
		memset( contact->manifolds, 0, contact->manifoldCount * sizeof( b3Manifold ) );
	}

	bool* consumed = NULL;
	if ( oldManifoldCount > 0 )
	{
		consumed = b3Bump( &arena, oldManifoldCount * sizeof( bool ) );
		memset( consumed, 0, oldManifoldCount * sizeof( bool ) );
	}

	b3Matrix3 matrixA = b3MakeMatrixFromQuat( xfA.q );

	const float normalMatchTolerance = 0.995f;
	for ( int i = 0; i < clusterCount; i++ )
	{
		VoxCluster* cluster = clusters + i;
		int pointCount = cluster->count;
		B3_ASSERT( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3Manifold* manifold = contact->manifolds + i;
		manifold->pointCount = pointCount;
		manifold->normal = b3MulMV( matrixA, cluster->normal );

		float bestDot = normalMatchTolerance;
		int bestIndex = B3_NULL_INDEX;

		for ( int j = 0; j < oldManifoldCount; j++ )
		{
			if ( consumed[j] == true )
			{
				continue;
			}

			float dot = b3Dot( oldManifolds[j].normal, manifold->normal );
			if ( dot > bestDot )
			{
				bestIndex = j;
				bestDot = dot;
			}
		}

		b3Manifold* matchedManifold = NULL;
		if ( bestIndex != B3_NULL_INDEX )
		{
			matchedManifold = oldManifolds + bestIndex;
			manifold->frictionImpulse = matchedManifold->frictionImpulse;
			manifold->rollingImpulse = matchedManifold->rollingImpulse;
			manifold->twistImpulse = matchedManifold->twistImpulse;
			consumed[bestIndex] = true;
		}

		for ( int j = 0; j < pointCount; j++ )
		{
			const b3LocalManifoldPoint* source = cluster->points + j;
			b3ManifoldPoint* target = manifold->points + j;
			// Contact points are computed in frame A
			target->anchorA = b3MulMV( matrixA, source->point );
			target->anchorB = b3Add( target->anchorA, b3SubPos( xfA.p, xfB.p ) );
			target->separation = source->separation;
			target->featureId = 1;

			// Preserve normal impulse if possible
			if ( matchedManifold != NULL )
			{
				int oldPointCount = matchedManifold->pointCount;
				for ( int k = 0; k < oldPointCount; k++ )
				{
					b3ManifoldPoint* oldPt = matchedManifold->points + k;

					if ( b3DistanceSquared( oldPt->anchorA, target->anchorA ) < POINT_RECYCLE_TOL_2 && oldPt->featureId != 0 )
					{
						target->normalImpulse = oldPt->normalImpulse;
						target->persisted = true;

						// claimed
						oldPt->featureId = 0;
						break;
					}
				}
			}
		}
	}

	const b3SurfaceMaterial* materialsA = b3GetShapeMaterials( shapeA );
	const b3SurfaceMaterial* materialB = b3GetShapeMaterials( shapeB );
	b3Vec3 tangentVelocityA = b3Vec3_zero;

	// Update friction and restitution if the voxels have materials
	// TODO: Implement per-voxel material support
	/* if ( shapeA->materialCount > 0 )
	{
		float friction = 0.0f;
		float restitution = 0.0f;
		float sampleCount = 0.0f;

		const uint8_t* materialIndices;
		if ( shapeA->type == b3_meshShape )
		{
			materialIndices = b3GetMeshMaterialIndices( shapeA->mesh.data );
		}
		else
		{
			materialIndices = b3GetHeightFieldMaterialIndices( shapeA->heightField );
		}

		for ( int i = 0; i < clusterCount; i++ )
		{
			b3Manifold* manifold = contact->manifolds + i;
			int pointCount = manifold->pointCount;
			for ( int j = 0; j < pointCount; j++ )
			{
				int triangleIndex = manifold->points[j].triangleIndex;
				int materialIndex;
				if ( shapeA->type == b3_meshShape )
				{
					materialIndex = materialIndices[triangleIndex];

					if ( materialMap != NULL )
					{
						materialIndex = materialMap[materialIndex];
					}
				}
				else
				{
					materialIndex = materialIndices[triangleIndex >> 1];
				}

				materialIndex = b3ClampInt( materialIndex, 0, shapeA->materialCount - 1 );
				b3SurfaceMaterial material = materialsA[materialIndex];
				friction += world->frictionCallback( material.friction, material.userMaterialId, materialB->friction,
													 materialB->userMaterialId );
				restitution += world->restitutionCallback( material.restitution, material.userMaterialId,
	materialB->restitution, materialB->userMaterialId );

				tangentVelocityA = b3Add( tangentVelocityA, material.tangentVelocity );

				sampleCount += 1.0f;
			}
		}

		if ( sampleCount > 0.0f )
		{
			float invCount = 1.0f / sampleCount;
			contact->friction = invCount * friction;
			contact->restitution = invCount * restitution;
			tangentVelocityA = b3MulSV( invCount, tangentVelocityA );
		}

		B3_ASSERT( b3IsValidFloat( contact->friction ) && contact->friction >= 0.0f );
		// B3_ASSERT( b3IsValidFloat( contact->restitution ) && contact->restitution >= 0.0f );
	}
	else */
	{
		// Keep these updated in case the values on the shapes are modified
		contact->friction = world->frictionCallback( materialsA[0].friction, materialsA[0].userMaterialId, materialB->friction,
													 materialB->userMaterialId );
		contact->restitution = world->restitutionCallback( materialsA[0].restitution, materialsA[0].userMaterialId,
														   materialB->restitution, materialB->userMaterialId );
		tangentVelocityA = materialsA[0].tangentVelocity;
	}

	tangentVelocityA = b3RotateVector( xfA.q, tangentVelocityA );

	float radiusB = 0.0f;
	if ( shapeB->type == b3_sphereShape )
	{
		radiusB = shapeB->sphere.radius;
	}
	else if ( shapeB->type == b3_capsuleShape )
	{
		radiusB = shapeB->capsule.radius;
	}
	else if ( shapeB->type == b3_hullShape )
	{
		radiusB = shapeB->hull->innerRadius;
	}

	contact->rollingResistance = materialB->rollingResistance * radiusB;

	b3Vec3 tangentVelocityB = b3RotateVector( xfB.q, materialB->tangentVelocity );
	contact->tangentVelocity = b3Sub( tangentVelocityA, tangentVelocityB );

	float ms = b3GetMilliseconds( ticks );
	b3Log( "voxel collision: collide took %f ms, post-collide took %f ms", collideMs, ms - collideMs );

	return true;
}
