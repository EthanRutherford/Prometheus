// SPDX-FileCopyrightText: 2026 Ethan Rutherford
// SPDX-License-Identifier: MIT

#include "contact.h"
#include "manifold.h"
#include "physics_world.h"
#include "reduce_cluster.h"
#include "shape.h"

#include "box3d/types.h"

#include <stdlib.h>

#define MANIFOLD_BUFFER_CAPACITY 256
#define POINT_BUFFER_CAPACITY MANIFOLD_BUFFER_CAPACITY * 32
#define POINT_RECYCLE_TOL_2 ( B3_LINEAR_SLOP * B3_LINEAR_SLOP )

// builds a mask that can be used to detect if a voxel has a neighbor that is closer to the candidate point than itself.
// This is used to cull contact points early, knowing that there is at least one coplanar voxel that can generate a deeper
// contact point. This early culling helps reduce load on the later clustering algorithm, which can cull additional points.
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

typedef struct VoxCollideContext
{
	b3LocalManifold* manifoldBuffer;
	int manifoldCount;
	b3LocalManifoldPoint* pointBuffer;
	int pointCount;

	const b3VoxelData* voxelsA;
	float maxDist2;
	float radius;
	union
	{
		// Sphere collision context
		struct
		{
			b3Vec3 center;
		};
		// Capsule collision context
		struct
		{
			b3Vec3 center1;
			b3Vec3 center2;
			b3Vec3 dir;
			b3Vec3 invDir;
		};
		// Hull collision context
		struct
		{
			int dummyB;
		};
		// Voxel collision context
		struct
		{
			const b3VoxelData* voxelsB;
		};
	};
} VoxCollideContext;

void voxSphereCallback( uint64_t code, uint32_t index, void* context )
{
	VoxCollideContext* ctx = (VoxCollideContext*)context;

	// abort if we run out of space in the manifold buffer
	if ( ctx->manifoldCount == MANIFOLD_BUFFER_CAPACITY )
		return;

	// skip fully occluded voxels. We want to calculate penetration based on the external surface,
	// so internal voxels would only add extra contact points that do not represent the true penetration
	// depth that needs to be resolved by the contact solver. Under typical circumstances, overlap with
	// internal voxels (without also overlapping surface voxels) should be prevented by collision resolution,
	// so skipping them here is a performance optimization that should not affect the correctness of the simulation.
	uint8_t flags = b3GetVoxelAttrs( ctx->voxelsA )[index].flags;
	if ( ( flags & b3_voxNeighborsMask ) == b3_voxOccludedMask )
		return;

	// decode the voxel bounding box from the Morton code
	b3Vec3 voxMin = { (float)b3DecodeVoxelX( code ), (float)b3DecodeVoxelY( code ), (float)b3DecodeVoxelZ( code ) };
	b3Vec3 voxMax = b3Add( voxMin, b3Vec3Of( 1.0f ) );

	// if there is a neighboring voxel which is closer to the sphere center than this voxel, then skip this one.
	// A neighboring voxel means we are part of an edge/surface, and we ideally only generate one contact point per edge/surface.
	// This reduces the number of contact points the manifold clustering algorithm needs to process.
	uint32_t neighborMask = getNeighborMask( ctx->center, voxMin, voxMax );
	if ( ( flags & neighborMask ) != 0 )
		return;

	// compute the closest point on the voxel bounds to the sphere center
	b3Vec3 closestPoint = b3Clamp( ctx->center, voxMin, voxMax );

	// compute the squared distance from the closest point to the sphere center
	b3Vec3 d = b3Sub( ctx->center, closestPoint );
	float dist2 = b3Dot( d, d );
	if ( dist2 < 1000.0f * FLT_MIN || dist2 > ctx->maxDist2 )
		return;

	// compute the voxel normal, which is always axis-aligned. The normal points
	// towards the sphere center, so the axis with the largest component is the normal axis.
	b3Vec3 absD = b3Abs( d );
	float maxAxis = max( absD.x, max( absD.y, absD.z ) );
	b3Vec3i axisMask = { absD.x == maxAxis, absD.y == maxAxis, absD.z == maxAxis };
	b3Vec3 normal = b3Select( axisMask, b3Sign( d ), b3Vec3_zero );

	// add a manifold and point for the contact
	b3LocalManifold* m = ctx->manifoldBuffer + ctx->manifoldCount++;
	m->normal = normal;
	m->points = ctx->pointBuffer + ctx->pointCount;
	m->pointCount = 1;

	b3LocalManifoldPoint* mp = ctx->pointBuffer + ctx->pointCount++;
	mp->point = closestPoint;
	mp->separation = sqrtf( dist2 ) - ctx->radius;
}

void collideVoxSphere( VoxCollideContext* context, b3Voxels voxelsA, const b3Sphere* sphereB, b3Transform bToA )
{
	// get the center, radius, and speculative distance of the sphere in voxel space/scale
	float invScale = 1.0f / voxelsA.scale;
	float specDist = B3_SPECULATIVE_DISTANCE * invScale;
	context->radius = sphereB->radius * invScale;
	context->maxDist2 = ( context->radius + specDist ) * ( context->radius + specDist );
	context->voxelsA = voxelsA.data;
	context->center = b3MulSV( invScale, b3TransformPoint( bToA, sphereB->center ) );

	// iterate the voxels that are within the bounding box
	b3Vec3 extent = b3Vec3Of( context->radius + specDist );
	b3AABB queryBounds = { b3Sub( context->center, extent ), b3Add( context->center, extent ) };
	b3QueryVoxels( voxelsA.data, queryBounds, voxSphereCallback, context );

	// descale the manifold points
	for ( int i = 0; i < context->pointCount; i++ )
	{
		b3LocalManifoldPoint* mp = context->pointBuffer + i;
		mp->point = b3MulSV( voxelsA.scale, mp->point );
		mp->separation = mp->separation * voxelsA.scale;
	}
}

void voxCapsuleCallback( uint64_t code, uint32_t index, void* context )
{
	VoxCollideContext* ctx = (VoxCollideContext*)context;

	// abort if we run out of space in the manifold buffer
	if ( ctx->manifoldCount == MANIFOLD_BUFFER_CAPACITY )
		return;

	// skip fully occluded voxels
	uint8_t flags = b3GetVoxelAttrs( ctx->voxelsA )[index].flags;
	if ( ( flags & b3_voxNeighborsMask ) == b3_voxOccludedMask )
		return;

	// decode the voxel bounding box from the Morton code
	b3Vec3 voxMin = { (float)b3DecodeVoxelX( code ), (float)b3DecodeVoxelY( code ), (float)b3DecodeVoxelZ( code ) };
	b3Vec3 voxMax = b3Add( voxMin, b3Vec3Of( 1.0f ) );

	// similar to the sphere case, we can skip voxels that are coplanar with nearer voxels
	// in this case, we're actually using the bounding box of the capsule segment, rather than a
	// single point, which is a conservative check that may miss some voxels that could be skipped.
	// However, this can prevent computing the closest point calculation below.
	uint32_t neighborMask = 0;
	neighborMask |= max( ctx->center1.x, ctx->center2.x ) < voxMin.x ? b3_negXNeighbor : 0;
	neighborMask |= min( ctx->center1.x, ctx->center2.x ) > voxMax.x ? b3_posXNeighbor : 0;
	neighborMask |= max( ctx->center1.y, ctx->center2.y ) < voxMin.y ? b3_negYNeighbor : 0;
	neighborMask |= min( ctx->center1.y, ctx->center2.y ) > voxMax.y ? b3_posYNeighbor : 0;
	neighborMask |= max( ctx->center1.z, ctx->center2.z ) < voxMin.z ? b3_negZNeighbor : 0;
	neighborMask |= min( ctx->center1.z, ctx->center2.z ) > voxMax.z ? b3_posZNeighbor : 0;
	if ( ( flags & neighborMask ) != 0 )
		return;

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
		( voxMin.x - ctx->center1.x ) * ctx->invDir.x,
		( voxMax.x - ctx->center1.x ) * ctx->invDir.x,
		( voxMin.y - ctx->center1.y ) * ctx->invDir.y,
		( voxMax.y - ctx->center1.y ) * ctx->invDir.y,
		( voxMin.z - ctx->center1.z ) * ctx->invDir.z,
		( voxMax.z - ctx->center1.z ) * ctx->invDir.z,
	};

	for ( int i = 0; i < 8; i++ )
	{
		float t = b3ClampFloat( ts[i], 0.0f, 1.0f );
		b3Vec3 pCaps = b3Add( ctx->center1, b3MulSV( t, ctx->dir ) );
		b3Vec3 pVox = b3Clamp( pCaps, voxMin, voxMax );
		b3Vec3 d = b3Sub( pCaps, pVox );
		float dist2 = b3LengthSquared( d );
		if ( dist2 < 1000.0f * FLT_MIN || dist2 > ctx->maxDist2 )
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

	for ( int i = 0; i < bestCount; i++ )
	{
		// we can do the neighbor check again with the found contact point, potentially filtering out a few additional voxels
		// that were not filtered out by the first neighbor check. TODO: test to see if this is worth it.
		neighborMask = getNeighborMask( bestPCaps[i], voxMin, voxMax );
		if ( ( flags & neighborMask ) != 0 )
			return;

		// compute the voxel normal, which is always axis-aligned. The normal points
		// towards the capsule segment closest point, so the axis with the largest component is the normal axis.
		b3Vec3 absD = b3Abs( bestD[i] );
		float maxAxis = max( absD.x, max( absD.y, absD.z ) );
		b3Vec3i axisMask = { absD.x == maxAxis, absD.y == maxAxis, absD.z == maxAxis };
		b3Vec3 normal = b3Select( axisMask, b3Sign( bestD[i] ), b3Vec3_zero );

		// add a manifold and point for the contact
		b3LocalManifold* m = ctx->manifoldBuffer + ctx->manifoldCount++;
		m->normal = normal;
		m->points = ctx->pointBuffer + ctx->pointCount;
		m->pointCount = 1;

		b3LocalManifoldPoint* mp = ctx->pointBuffer + ctx->pointCount++;
		mp->point = bestPVox[i];
		mp->separation = sqrtf( bestDist2[i] ) - ctx->radius;
	}
}

void collideVoxCapsule( VoxCollideContext* context, b3Voxels voxelsA, const b3Capsule* capsuleB, b3Transform bToA )
{
	// get the center, radius, and speculative distance of the capsule in voxel space/scale
	float invScale = 1.0f / voxelsA.scale;
	float specDist = B3_SPECULATIVE_DISTANCE * invScale;
	context->radius = capsuleB->radius * invScale;
	context->maxDist2 = ( context->radius + specDist ) * ( context->radius + specDist );
	context->voxelsA = voxelsA.data;
	context->center1 = b3MulSV( invScale, b3TransformPoint( bToA, capsuleB->center1 ) );
	context->center2 = b3MulSV( invScale, b3TransformPoint( bToA, capsuleB->center2 ) );
	context->dir = b3Sub( context->center2, context->center1 );
	context->invDir = (b3Vec3){
		context->dir.x != 0 ? 1.0f / context->dir.x : 0.0f,
		context->dir.y != 0 ? 1.0f / context->dir.y : 0.0f,
		context->dir.z != 0 ? 1.0f / context->dir.z : 0.0f,
	};

	// iterate the voxels that are within the bounding box
	b3Vec3 extent = b3Vec3Of( context->radius + specDist );
	b3Vec3 capsuleMin = b3Min( context->center1, context->center2 );
	b3Vec3 capsuleMax = b3Max( context->center1, context->center2 );
	b3AABB queryBounds = { b3Sub( capsuleMin, extent ), b3Add( capsuleMax, extent ) };
	b3QueryVoxels( voxelsA.data, queryBounds, voxCapsuleCallback, context );

	// descale the manifold points
	for ( int i = 0; i < context->pointCount; i++ )
	{
		b3LocalManifoldPoint* mp = context->pointBuffer + i;
		mp->point = b3MulSV( voxelsA.scale, mp->point );
		mp->separation = mp->separation * voxelsA.scale;
	}
}

void collideVoxHull( VoxCollideContext* context, b3Voxels voxelsA, const b3HullData* hullB, b3Transform bToA )
{
	printf( "collideVoxHull not implemented\n" );
}

void collideVoxVox( VoxCollideContext* context, b3Voxels voxelsA, b3Voxels voxelsB, b3Transform bToA )
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

void initClusters( VoxCluster* clusters, b3LocalManifoldPoint* clusterPoints )
{
	// Initialize the cluster normals to the 6 axis-aligned directions
	// (Must match the clusterIndex function)
	clusters[0].normal.x = -1.0f;
	clusters[1].normal.x = 1.0f;
	clusters[2].normal.y = -1.0f;
	clusters[3].normal.y = 1.0f;
	clusters[4].normal.z = -1.0f;
	clusters[5].normal.z = 1.0f;

	for ( int i = 0; i < 6; ++i )
	{
		VoxCluster* cluster = clusters + i;
		cluster->points = clusterPoints;
		clusterPoints += cluster->capacity;
	}
}

int clusterIndex( b3Vec3 normal )
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
	context.manifoldBuffer = b3Bump( &arena, MANIFOLD_BUFFER_CAPACITY * sizeof( b3LocalManifold ) );
	context.pointBuffer = b3Bump( &arena, POINT_BUFFER_CAPACITY * sizeof( b3LocalManifoldPoint ) );

	b3Transform transformBtoA = b3InvMulWorldTransforms( xfA, xfB );

	if ( shapeB->type == b3_sphereShape )
	{
		collideVoxSphere( &context, shapeA->voxels, &shapeB->sphere, transformBtoA );
	}
	else if ( shapeB->type == b3_capsuleShape )
	{
		collideVoxCapsule( &context, shapeA->voxels, &shapeB->capsule, transformBtoA );
	}
	else if ( shapeB->type == b3_hullShape )
	{
		collideVoxHull( &context, shapeA->voxels, shapeB->hull, transformBtoA );
	}
	else
	{
		B3_ASSERT( shapeB->type == b3_voxelShape );
		collideVoxVox( &context, shapeA->voxels, shapeB->voxels, transformBtoA );
	}

	if ( context.manifoldCount == 0 )
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
	int* clusterMemberships = b3Bump( &arena, context.manifoldCount * sizeof( int ) );
	int clusterPointCount = 0;
	for ( int i = 0; i < context.manifoldCount; ++i )
	{
		b3LocalManifold* lm = context.manifoldBuffer + i;
		int clusterIdx = clusterIndex( lm->normal );
		VoxCluster* cluster = clusters + clusterIdx;
		clusterMemberships[i] = clusterIdx;
		cluster->capacity += lm->pointCount;
		clusterPointCount += lm->pointCount;
	}

	// Initialize the clusters
	b3LocalManifoldPoint* clusterPoints = b3Bump( &arena, clusterPointCount * sizeof( b3LocalManifoldPoint ) );
	initClusters( clusters, clusterPoints );

	// Clone manifold points into the clusters
	for ( int i = 0; i < context.manifoldCount; ++i )
	{
		b3LocalManifold* lm = context.manifoldBuffer + i;
		int clusterIdx = clusterMemberships[i];
		VoxCluster* cluster = clusters + clusterIdx;

		for ( int j = 0; j < lm->pointCount; ++j )
		{
			b3LocalManifoldPoint* srcPt = lm->points + j;
			b3LocalManifoldPoint* dstPt = cluster->points + cluster->count++;
			dstPt->point = srcPt->point;
			dstPt->separation = srcPt->separation;
		}
	}

	// Reduce clusters
	int clusterCount = 0;
	for ( int i = 0; i < 6; ++i )
	{
		VoxCluster* cluster = clusters + i;
		if ( cluster->count == 0 )
			continue;

		cluster->count = b3ReduceCluster( cluster->points, cluster->count, cluster->normal, arena );
		clusterCount++;
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

		for ( int j = 0; j < oldManifoldCount; ++j )
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

		for ( int j = 0; j < pointCount; ++j )
		{
			const b3LocalManifoldPoint* source = cluster->points + j;
			b3ManifoldPoint* target = manifold->points + j;
			// Contact points are computed in frame A
			target->anchorA = b3MulMV( matrixA, source->point );
			target->anchorB = b3Add( target->anchorA, b3SubPos( xfA.p, xfB.p ) );
			target->separation = source->separation; // - restOffset;
			target->featureId = 1;

			// Preserve normal impulse if possible
			if ( matchedManifold != NULL )
			{
				int oldPointCount = matchedManifold->pointCount;
				for ( int k = 0; k < oldPointCount; ++k )
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

		for ( int i = 0; i < clusterCount; ++i )
		{
			b3Manifold* manifold = contact->manifolds + i;
			int pointCount = manifold->pointCount;
			for ( int j = 0; j < pointCount; ++j )
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
	if ( ms > 5.0f )
		b3Log( "slow voxel collision: collision took %f ms, post-collide took %f ms", collideMs, ms - collideMs );

	return true;
}
