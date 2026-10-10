#pragma once

#include "contact.h"

#include "box3d/types.h"

#include <stdbool.h>

typedef struct CacheRefreshContext
{
	const b3VoxelData* voxels;
	b3VoxelQueryCache* cache;
} CacheRefreshContext;

typedef struct VoxCandidatePoint
{
	b3Vec3 point;
	b3Vec3 normal;
	float separation;
} VoxCandidatePoint;

typedef struct VoxCollideContext
{
	b3VoxelQueryCache* cacheA;
	b3VoxelQueryCache* cacheB;

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

// Clip the query bounds to the voxel grid bounds, matching the behavior of b3QueryVoxels.
// This stabilizes the voxel contact cache when the query bounds shift by less than a voxel size.
inline b3AABB computeVoxelBounds( const b3VoxelData* voxels, b3Vec3 lower, b3Vec3 upper )
{
	b3Vec3 lowerBound = b3Max( b3Floor( lower ), voxels->bounds.lowerBound );
	b3Vec3 upperBound = b3Min( b3Ceil( upper ), voxels->bounds.upperBound );
	return (b3AABB){ lowerBound, upperBound };
}

bool refreshVoxCache( b3VoxelQueryCache* cache, const b3VoxelData* voxels, b3AABB bounds );

void collideVoxSphereW4( VoxCollideContext* context, b3Transform bToA, b3Arena arena );
void collideVoxCapsuleW4( VoxCollideContext* context, b3Transform bToA, b3Arena arena );
void collideVoxHullW4( VoxCollideContext* context, b3Transform bToA, b3Arena arena );
void collideVoxVoxW4( VoxCollideContext* context, b3Transform bToA, b3Arena arena );

void collideVoxSphereW8( VoxCollideContext* context, b3Transform bToA, b3Arena arena );
void collideVoxCapsuleW8( VoxCollideContext* context, b3Transform bToA, b3Arena arena );
void collideVoxHullW8( VoxCollideContext* context, b3Transform bToA, b3Arena arena );
void collideVoxVoxW8( VoxCollideContext* context, b3Transform bToA, b3Arena arena );
