#include "simd.h"
#include "voxel_manifold.h"

static b3Vec3W b3ClampVW( const b3Vec3W v, const b3Vec3W min, const b3Vec3W max )
{
	return (b3Vec3W){
		b3MaxW( min.X, b3MinW( max.X, v.X ) ),
		b3MaxW( min.Y, b3MinW( max.Y, v.Y ) ),
		b3MaxW( min.Z, b3MinW( max.Z, v.Z ) ),
	};
}

// builds a mask that can be used to detect if a voxel has a neighbor that is closer to the candidate point than itself.
// This is used to cull contact points early, knowing that there is at least one coplanar voxel that can generate a deeper
// contact point. This early culling helps reduce load on the later clustering algorithm, which can cull additional points.
static b3FloatW B3_WIDE( getNeighborMask )( const b3Vec3W candidate, const b3Vec3W voxMin, const b3Vec3W voxMax )
{
	b3FloatW zeroW = b3ZeroW();
	b3FloatW b3_negXNeighborW = b3SplatW( b3IntToFloat( b3_negXNeighbor ) );
	b3FloatW b3_posXNeighborW = b3SplatW( b3IntToFloat( b3_posXNeighbor ) );
	b3FloatW b3_negYNeighborW = b3SplatW( b3IntToFloat( b3_negYNeighbor ) );
	b3FloatW b3_posYNeighborW = b3SplatW( b3IntToFloat( b3_posYNeighbor ) );
	b3FloatW b3_negZNeighborW = b3SplatW( b3IntToFloat( b3_negZNeighbor ) );
	b3FloatW b3_posZNeighborW = b3SplatW( b3IntToFloat( b3_posZNeighbor ) );

	// this is effectively an AABB SAT test, resulting in a mask of the separating axes.
	// if a voxel has a neighbor along a separating axis, that neighbor is closer to the candidate point.
	// An extra bonus, this also filters out any contact points which would have a normal pointed into
	// a neighboring voxel, which is not a valid contact point for collision resolution.
	b3FloatW neighborMask = b3ZeroW();
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_negXNeighborW, b3LessThanW( candidate.X, voxMin.X ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posXNeighborW, b3GreaterOrEqualW( candidate.X, voxMax.X ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_negYNeighborW, b3LessThanW( candidate.Y, voxMin.Y ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posYNeighborW, b3GreaterOrEqualW( candidate.Y, voxMax.Y ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_negZNeighborW, b3LessThanW( candidate.Z, voxMin.Z ) ) );
	neighborMask = b3OrW( neighborMask, b3BlendW( zeroW, b3_posZNeighborW, b3GreaterOrEqualW( candidate.Z, voxMax.Z ) ) );
	return neighborMask;
}

void B3_WIDE( collideVoxSphere )( VoxCollideContext* context, b3Transform bToA, b3Arena arena )
{
	b3FloatW zeroW = b3ZeroW();
	b3FloatW halfW = b3SplatW( 0.5f );
	b3FloatW oneW = b3SplatW( 1.0f );
	b3FloatW epsilonW = b3SplatW( 1000.0f * FLT_MIN );
	b3Vec3W oneVW = { b3SplatW( 1.0f ), b3SplatW( 1.0f ), b3SplatW( 1.0f ) };

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
	refreshVoxCache( context->cacheA, voxelsA.data, queryBounds );

	// early exit if no voxels are in the query bounds
	if ( context->cacheA->count == 0 )
		return;

	// create and initialize the wide voxel array for SIMD processing
	int paddedCount = ( context->cacheA->count + B3_SIMD_WIDTH - 1 ) & ~( B3_SIMD_WIDTH - 1 );
	float* accepted = b3Bump( &arena, paddedCount * sizeof( float ) );

	// create wide vectors for intersection parameters
	b3FloatW maxDistSqrW = b3SplatW( maxDistSqr );
	b3FloatW scale = b3SplatW( voxelsA.scale );
	b3FloatW radiusW = b3SplatW( radius );
	b3Vec3W centerW = { b3SplatW( center.x ), b3SplatW( center.y ), b3SplatW( center.z ) };

	// Collide Step 1: filter gathered candidates using neighbor masks.
	for ( int i = 0; i < paddedCount; i += B3_SIMD_WIDTH )
	{
		b3Vec3W minW = b3LoadVW( &context->cacheA->vMinX[i], &context->cacheA->vMinY[i], &context->cacheA->vMinZ[i] );
		b3Vec3W maxW = b3AddVW( minW, oneVW );
		b3FloatW flagsW;
		memcpy( &flagsW, &context->cacheA->flags[i], sizeof( b3FloatW ) );

		// if there is a neighboring voxel which is closer to the sphere center than this voxel, then skip this one.
		// A neighboring voxel means we are part of an edge/surface, and we ideally only generate one contact point per
		// edge/surface. This reduces the number of contact points the manifold clustering algorithm needs to process.
		b3FloatW neighborMask = B3_WIDE( getNeighborMask )( centerW, minW, maxW );
		b3FloatW neighborResults = b3AndW( flagsW, neighborMask );
		b3StoreW( &accepted[i], b3EqualsW( neighborResults, zeroW ) );
	}

	// Step 2: compact the accepted candidates
	float* vMinX = b3Bump( &arena, paddedCount * sizeof( float ) );
	float* vMinY = b3Bump( &arena, paddedCount * sizeof( float ) );
	float* vMinZ = b3Bump( &arena, paddedCount * sizeof( float ) );
	int acceptedCount = 0;

	for ( int i = 0; i < context->cacheA->count; i++ )
	{
		if ( accepted[i] != 0 )
		{
			vMinX[acceptedCount] = context->cacheA->vMinX[i];
			vMinY[acceptedCount] = context->cacheA->vMinY[i];
			vMinZ[acceptedCount] = context->cacheA->vMinZ[i];
			acceptedCount++;
		}
	}

	// recompute the padded count post-compaction
	paddedCount = ( acceptedCount + B3_SIMD_WIDTH - 1 ) & ~( B3_SIMD_WIDTH - 1 );

	// Step 3: compute the closest point on each voxel to the sphere center, and compute the separation.
	float* pointX = b3Bump( &arena, paddedCount * sizeof( float ) );
	float* pointY = b3Bump( &arena, paddedCount * sizeof( float ) );
	float* pointZ = b3Bump( &arena, paddedCount * sizeof( float ) );
	float* normalX = b3Bump( &arena, paddedCount * sizeof( float ) );
	float* normalY = b3Bump( &arena, paddedCount * sizeof( float ) );
	float* normalZ = b3Bump( &arena, paddedCount * sizeof( float ) );
	float* separation = b3Bump( &arena, paddedCount * sizeof( float ) );

	for ( int i = 0; i < paddedCount; i += B3_SIMD_WIDTH )
	{
		b3Vec3W minW = b3LoadVW( &vMinX[i], &vMinY[i], &vMinZ[i] );
		b3Vec3W maxW = b3AddVW( minW, oneVW );

		// compute the closest point on the voxel bounds to the sphere center
		b3Vec3W closestPoint = b3ClampVW( centerW, minW, maxW );

		// compute the squared distance from the closest point to the sphere center
		b3Vec3W d = b3SubVW( centerW, closestPoint );
		b3FloatW distSqr = b3DotW( d, d );
		b3StoreW( &accepted[i], b3AndW( b3GreaterThanW( distSqr, epsilonW ), b3LessThanW( distSqr, maxDistSqrW ) ) );
		if ( !b3AnyTrueW( b3LoadW( &accepted[i] ) ) )
			continue;

		// compute normal and closest point on sphere.
		// contact point is midpoint between closest points
		b3FloatW dist = b3SqrtW( distSqr );
		b3StoreVW( &normalX[i], &normalY[i], &normalZ[i], b3MulSVW( b3DivW( oneW, dist ), d ) );
		b3Vec3W closestPointSphere = b3SubVW( centerW, b3MulSVW( radiusW, b3LoadVW( &normalX[i], &normalY[i], &normalZ[i] ) ) );

		// descale the point and compute separation
		b3StoreVW( &pointX[i], &pointY[i], &pointZ[i],
				   b3MulSVW( scale, b3MulSVW( halfW, b3AddVW( closestPoint, closestPointSphere ) ) ) );
		b3StoreW( &separation[i], b3MulW( scale, b3SubW( dist, radiusW ) ) );
	}

	// Step 4: add a candidate point for all valid lanes
	for ( int i = 0; i < acceptedCount; i++ )
	{
		if ( accepted[i] == 0 )
			continue;

		VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
		cp->point.x = pointX[i];
		cp->point.y = pointY[i];
		cp->point.z = pointZ[i];

		cp->normal.x = normalX[i];
		cp->normal.y = normalY[i];
		cp->normal.z = normalZ[i];

		cp->separation = separation[i];
	}
}
