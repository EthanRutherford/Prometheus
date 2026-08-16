// SPDX-FileCopyrightText: 2026 Ethan Rutherford
// SPDX-License-Identifier: MIT

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
static const b3FloatW oneW = B3_STATIC_FLOAT_W( 1.0f );
static const b3FloatW epsilonW = B3_STATIC_FLOAT_W( 1000.0f * FLT_MIN );
static const b3Vec3W zeroVW = { B3_STATIC_FLOAT_W( 0.0f ), B3_STATIC_FLOAT_W( 0.0f ), B3_STATIC_FLOAT_W( 0.0f ) };
static const b3Vec3W oneVW = { B3_STATIC_FLOAT_W( 1.0f ), B3_STATIC_FLOAT_W( 1.0f ), B3_STATIC_FLOAT_W( 1.0f ) };

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
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posXNeighborW, b3GreaterThanW( candidate.X, voxMax.X ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_negYNeighborW, b3LessThanW( candidate.Y, voxMin.Y ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posYNeighborW, b3GreaterThanW( candidate.Y, voxMax.Y ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_negZNeighborW, b3LessThanW( candidate.Z, voxMin.Z ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posZNeighborW, b3GreaterThanW( candidate.Z, voxMax.Z ) ) );
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

	b3VoxelCache* cache = b3Array_Emplace( ctx->contact->voxelCache );

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

	// clear the cache and gather new voxels
	contact->queryBounds = bounds;
	contact->voxelCache.count = 0;

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
	float maxDist2 = ( radius + specDist ) * ( radius + specDist );
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
				( (int*)&vox->flags )[lane] = 0;
				continue;
			}

			b3VoxelCache* cache = &context->contact->voxelCache.data[index];
			( (float*)&vox->min.X )[lane] = cache->min.x;
			( (float*)&vox->min.Y )[lane] = cache->min.y;
			( (float*)&vox->min.Z )[lane] = cache->min.z;
			( (int*)&vox->flags )[lane] = cache->flags;
		}
	}

	// create wide vectors for intersection parameters
	b3FloatW maxDist2W = b3SplatW( maxDist2 );
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
		if ( ( (int*)&wideVoxels[wi].accepted )[li] != 0 )
		{
			if ( i != acceptedCount )
			{
				int wj = acceptedCount / B3_SIMD_WIDTH;
				int lj = acceptedCount % B3_SIMD_WIDTH;
				( (float*)&wideVoxels[wj].min.X )[lj] = ( (float*)&wideVoxels[wi].min.X )[li];
				( (float*)&wideVoxels[wj].min.Y )[lj] = ( (float*)&wideVoxels[wi].min.Y )[li];
				( (float*)&wideVoxels[wj].min.Z )[lj] = ( (float*)&wideVoxels[wi].min.Z )[li];

				( (float*)&wideVoxels[wj].max.X )[lj] = ( (float*)&wideVoxels[wi].max.X )[li];
				( (float*)&wideVoxels[wj].max.Y )[lj] = ( (float*)&wideVoxels[wi].max.Y )[li];
				( (float*)&wideVoxels[wj].max.Z )[lj] = ( (float*)&wideVoxels[wi].max.Z )[li];

				( (int*)&wideVoxels[wj].flags )[lj] = ( (int*)&wideVoxels[wi].flags )[li];
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
			( (int*)&wideVoxels[wideCount - 1].flags )[lane] = 0;
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
		b3FloatW dist2 = b3DotW( d, d );
		vox->accepted = b3AndW( b3GreaterThanW( dist2, epsilonW ), b3LessThanW( dist2, maxDist2W ) );
		if ( !b3AnyTrueW( vox->accepted ) )
			continue;

		// compute the voxel normal, which is always axis-aligned. The normal points
		// towards the sphere center, so the axis with the largest component is the normal axis.
		b3Vec3W absD = b3AbsVW( d );
		b3Vec3W xgty = b3Vec3WOf( b3GreaterThanW( absD.X, absD.Y ) );
		b3Vec3W xgtz = b3Vec3WOf( b3GreaterThanW( absD.X, absD.Z ) );
		b3Vec3W ygtz = b3Vec3WOf( b3GreaterThanW( absD.Y, absD.Z ) );
		static const b3Vec3W xAxis = { B3_STATIC_MASK_W( 0xFFFFFFFF ), B3_STATIC_MASK_W( 0 ), B3_STATIC_MASK_W( 0 ) };
		static const b3Vec3W yAxis = { B3_STATIC_MASK_W( 0 ), B3_STATIC_MASK_W( 0xFFFFFFFF ), B3_STATIC_MASK_W( 0 ) };
		static const b3Vec3W zAxis = { B3_STATIC_MASK_W( 0 ), B3_STATIC_MASK_W( 0 ), B3_STATIC_MASK_W( 0xFFFFFFFF ) };
		b3Vec3W axisMask = b3SelectVW( xgty, b3SelectVW( xgtz, xAxis, zAxis ), b3SelectVW( ygtz, yAxis, zAxis ) );
		vox->normal = b3SelectVW( axisMask, b3SignVW( d ), zeroVW );

		// descale the point and compute separation
		vox->point = b3MulSVW( scale, closestPoint );
		vox->separation = b3MulW( scale, b3SubW( b3SqrtW( dist2 ), radiusW ) );
	}

	// Step 4: add a candidate point for all valid lanes
	for ( int i = 0; i < wideCount; i++ )
	{
		VoxelWide* vox = &wideVoxels[i];

		for ( int lane = 0; lane < B3_SIMD_WIDTH; lane++ )
		{
			if ( ( (int*)&vox->accepted )[lane] == 0 )
				continue;

			VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
			cp->point.x = ( (float*)&vox->point.X )[lane];
			cp->point.y = ( (float*)&vox->point.Y )[lane];
			cp->point.z = ( (float*)&vox->point.Z )[lane];

			cp->normal.x = ( (float*)&vox->normal.X )[lane];
			cp->normal.y = ( (float*)&vox->normal.Y )[lane];
			cp->normal.z = ( (float*)&vox->normal.Z )[lane];

			cp->separation = ( ( (float*)&vox->separation )[lane] );
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
	neighborMask |= candidate.x > voxMax.x ? b3_posXNeighbor : 0;
	neighborMask |= candidate.y < voxMin.y ? b3_negYNeighbor : 0;
	neighborMask |= candidate.y > voxMax.y ? b3_posYNeighbor : 0;
	neighborMask |= candidate.z < voxMin.z ? b3_negZNeighbor : 0;
	neighborMask |= candidate.z > voxMax.z ? b3_posZNeighbor : 0;
	return neighborMask;
}

static void collideVoxCapsule( VoxCollideContext* context, b3Transform bToA, b3Arena arena )
{
	// get the center, radius, and speculative distance of the capsule in voxel space/scale
	const b3Voxels voxelsA = context->voxelsA;
	float invScale = 1.0f / voxelsA.scale;
	float specDist = B3_SPECULATIVE_DISTANCE * invScale;
	float radius = context->capsuleB->radius * invScale;
	float maxDist2 = ( radius + specDist ) * ( radius + specDist );
	b3Vec3 center1 = b3MulSV( invScale, b3TransformPoint( bToA, context->capsuleB->center1 ) );
	b3Vec3 center2 = b3MulSV( invScale, b3TransformPoint( bToA, context->capsuleB->center2 ) );
	b3Vec3 dir = b3Sub( center2, center1 );
	b3Vec3 invDir = (b3Vec3){
		dir.x != 0 ? 1.0f / dir.x : 0.0f,
		dir.y != 0 ? 1.0f / dir.y : 0.0f,
		dir.z != 0 ? 1.0f / dir.z : 0.0f,
	};

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
		neighborMask |= min( center1.x, center2.x ) > voxMax.x ? b3_posXNeighbor : 0;
		neighborMask |= max( center1.y, center2.y ) < voxMin.y ? b3_negYNeighbor : 0;
		neighborMask |= min( center1.y, center2.y ) > voxMax.y ? b3_posYNeighbor : 0;
		neighborMask |= max( center1.z, center2.z ) < voxMin.z ? b3_negZNeighbor : 0;
		neighborMask |= min( center1.z, center2.z ) > voxMax.z ? b3_posZNeighbor : 0;
		if ( ( flags & neighborMask ) != 0 )
			continue;

		// find the closest points between the capsule segment and the voxel bounding box, using an
		// analytical solution. This is more performant than the general GJK solution.
		// capsules can result in up to two contact points with a cube, so we need to find the two closest points.
		float bestT[2];
		b3Vec3 bestPCaps[2];
		b3Vec3 bestPVox[2];
		b3Vec3 bestD[2];
		float bestDist2[2] = { FLT_MAX, FLT_MAX };
		int bestCount = 0;

		// capsule endpoints, plus the 6 faces of the voxel bounding box
		float ts[8] = {
			0.0f,
			1.0f,
			( voxMin.x - center1.x ) * invDir.x,
			( voxMax.x - center1.x ) * invDir.x,
			( voxMin.y - center1.y ) * invDir.y,
			( voxMax.y - center1.y ) * invDir.y,
			( voxMin.z - center1.z ) * invDir.z,
			( voxMax.z - center1.z ) * invDir.z,
		};

		for ( int j = 0; j < 8; j++ )
		{
			float t = b3ClampFloat( ts[j], 0.0f, 1.0f );
			b3Vec3 pCaps = b3Add( center1, b3MulSV( t, dir ) );
			b3Vec3 pVox = b3Clamp( pCaps, voxMin, voxMax );
			b3Vec3 d = b3Sub( pCaps, pVox );
			float dist2 = b3LengthSquared( d );
			if ( dist2 < 1000.0f * FLT_MIN || dist2 > maxDist2 )
				continue;

			if ( dist2 < bestDist2[0] )
			{
				bestT[1] = bestT[0];
				bestPCaps[1] = bestPCaps[0];
				bestPVox[1] = bestPVox[0];
				bestD[1] = bestD[0];
				bestDist2[1] = bestDist2[0];

				bestT[0] = t;
				bestPCaps[0] = pCaps;
				bestPVox[0] = pVox;
				bestD[0] = d;
				bestDist2[0] = dist2;
				bestCount = max( bestCount, 1 );
			}
			else if ( dist2 < bestDist2[1] )
			{
				bestT[1] = t;
				bestPCaps[1] = pCaps;
				bestPVox[1] = pVox;
				bestD[1] = d;
				bestDist2[1] = dist2;
				bestCount = 2;
			}
		}

		for ( int j = 0; j < bestCount; j++ )
		{
			// we can do the neighbor check again with the found contact point, potentially filtering out a few additional voxels
			// that were not filtered out by the first neighbor check. TODO: test to see if this is worth it.
			neighborMask = getNeighborMask( bestPCaps[j], voxMin, voxMax );
			if ( ( flags & neighborMask ) != 0 )
				continue;

			// compute the voxel normal, which is always axis-aligned. The normal points
			// towards the capsule segment closest point, so the axis with the largest component is the normal axis.
			b3Vec3 absD = b3Abs( bestD[j] );
			b3Vec3i axisMask = absD.x > absD.y ? ( absD.x > absD.z ? (b3Vec3i){ 1, 0, 0 } : (b3Vec3i){ 0, 0, 1 } )
											   : ( absD.y > absD.z ? (b3Vec3i){ 0, 1, 0 } : (b3Vec3i){ 0, 0, 1 } );
			b3Vec3 normal = b3Select( axisMask, b3Sign( bestD[j] ), b3Vec3_zero );

			// add a candidate point for the contact
			VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
			cp->point = b3MulSV( voxelsA.scale, bestPVox[j] );
			cp->normal = normal;
			cp->separation = ( sqrtf( bestDist2[j] ) - radius ) * voxelsA.scale;
		}
	}
}

static void collideVoxHull( VoxCollideContext* context, b3Transform bToA, b3Arena arena )
{
	printf( "collideVoxHull not implemented\n" );
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

static void initClusters( VoxCluster* clusters, b3LocalManifoldPoint* clusterPoints )
{
	// Initialize the cluster normals to the 6 axis-aligned directions
	// (Must match the clusterIndex function)
	clusters[0].normal.x = -1.0f;
	clusters[1].normal.x = 1.0f;
	clusters[2].normal.y = -1.0f;
	clusters[3].normal.y = 1.0f;
	clusters[4].normal.z = -1.0f;
	clusters[5].normal.z = 1.0f;

	for ( int i = 0; i < 6; i++ )
	{
		VoxCluster* cluster = clusters + i;
		cluster->points = clusterPoints;
		clusterPoints += cluster->capacity;
	}
}

static int clusterIndex( b3Vec3 normal )
{
	int x = (int)( normal.x > 0 );
	int y = (int)( normal.y > 0 ) + 2;
	int z = (int)( normal.z > 0 ) + 4;
	return ( ( normal.x != 0 ) ? x : 0 ) + ( ( normal.y != 0 ) ? y : 0 ) + ( ( normal.z != 0 ) ? z : 0 );
}

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
	// Because the voxel normals are axis-aligned, we only have 6 possible normal directions.
	// This allows for several assumptions that make this faster than the mesh clustering logic.
	VoxCluster clusters[6] = { 0 };
	int* clusterMemberships = b3Bump( &arena, context.pointCount * sizeof( int ) );
	for ( int i = 0; i < context.pointCount; i++ )
	{
		VoxCandidatePoint* cp = context.pointBuffer + i;
		int clusterIdx = clusterIndex( cp->normal );
		VoxCluster* cluster = clusters + clusterIdx;
		clusterMemberships[i] = clusterIdx;
		cluster->capacity++;
	}

	// Initialize the clusters
	b3LocalManifoldPoint* clusterPoints = b3Bump( &arena, context.pointCount * sizeof( b3LocalManifoldPoint ) );
	initClusters( clusters, clusterPoints );

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
	int clusterCount = 0;
	for ( int i = 0; i < 6; i++ )
	{
		VoxCluster* cluster = clusters + i;
		if ( cluster->count == 0 )
			continue;

		cluster->count = b3ReduceCluster( cluster->points, cluster->count, cluster->normal, arena );
		clusterCount++;

		// filter out any duplicate points in the cluster. Collisions that land on a voxel border
		// can be reported by both neighbors, so we need to remove duplicates to avoid jittering.
		for ( int j = 0; j < cluster->count; j++ )
		{
			for ( int k = j + 1; k < cluster->count; k++ )
			{
				b3LocalManifoldPoint* ptA = cluster->points + j;
				b3LocalManifoldPoint* ptB = cluster->points + k;
				if ( b3DistanceSquared( ptA->point, ptB->point ) < FLT_EPSILON * FLT_EPSILON )
				{
					cluster->points[k] = cluster->points[cluster->count - 1];
					cluster->count--;
					k--;
				}
			}
		}
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
	for ( int i = 0, m = 0; i < 6; i++ )
	{
		if ( clusters[i].count == 0 )
			continue;

		VoxCluster* cluster = clusters + i;
		int pointCount = cluster->count;
		B3_ASSERT( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3Manifold* manifold = contact->manifolds + m++;
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
				restitution += world->restitutionCallback( material.restitution, material.userMaterialId, materialB->restitution,
														   materialB->userMaterialId );

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
