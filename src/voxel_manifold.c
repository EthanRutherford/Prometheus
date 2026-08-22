// SPDX-FileCopyrightText: 2026 Ethan Rutherford
// SPDX-License-Identifier: MIT

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

static b3Vec3 invTransfromPointMat( b3Matrix3 invMat, b3Vec3 t, b3Vec3 p )
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

// process an edge or corner voxel to extract corners and edges
static void processVoxel( b3Vec3 voxMin, uint32_t flags, b3Vec3* corners, int* cornerCount, b3Vec3* edgePt0, b3Vec3* edgePt1,
						  b3Vec3* edgeNorm0, b3Vec3* edgeNorm1, int* edgeCount )
{
#define VOX_CORNER( X, Y, Z ) corners[( *cornerCount )++] = ( (b3Vec3){ voxMin.x + ( X ), voxMin.y + ( Y ), voxMin.z + ( Z ) } );
#define VOX_EDGE( X0, Y0, Z0, X1, Y1, Z1, NX0, NY0, NZ0, NX1, NY1, NZ1 )                                                         \
	edgePt0[( *edgeCount )] = ( (b3Vec3){ voxMin.x + ( X0 ), voxMin.y + ( Y0 ), voxMin.z + ( Z0 ) } );                           \
	edgePt1[( *edgeCount )] = ( (b3Vec3){ voxMin.x + ( X1 ), voxMin.y + ( Y1 ), voxMin.z + ( Z1 ) } );                           \
	edgeNorm0[( *edgeCount )] = ( (b3Vec3){ NX0, NY0, NZ0 } );                                                                   \
	edgeNorm1[( *edgeCount )] = ( (b3Vec3){ NX1, NY1, NZ1 } );                                                                   \
	( *edgeCount )++;

	/* clang-format off
		This function's switch statement body was generated using the following javascript code:
		function generate(posX, negX, posY, negY, posZ, negZ) {
			// case label
			const parts = [
				posX ? "b3_posXNeighbor" : "",
				negX ? "b3_negXNeighbor" : "",
				posY ? "b3_posYNeighbor" : "",
				negY ? "b3_negYNeighbor" : "",
				posZ ? "b3_posZNeighbor" : "",
				negZ ? "b3_negZNeighbor" : "",
			].filter(x => x);

			const label = `case ${parts.length ? parts.join(" | ") : "b3_noNeighbors"}:\n`;

			let body = "";

			// corners
			const LUTX = [negX, posX];
			const LUTY = [negY, posY];
			const LUTZ = [negZ, posZ];
			for (let x = 0; x <= 1; x++) {
				for (let y = 0; y <= 1; y++) {
					for (let z = 0; z <= 1; z++) {
						if (!LUTX[x] && !LUTY[y] && !LUTZ[z])
							body += `\tVOX_CORNER( ${x}, ${y}, ${z} )\n`;
					}
				}
			}

			// edges
			const LUTLUT = [[LUTY, LUTZ], [LUTX, LUTZ], [LUTX, LUTY]];
			for (let axis = 0; axis <= 2; axis++) {
				for (let v0 = 0; v0 <= 1; v0++) {
					for (let v1 = 0; v1 <= 1; v1++) {
						if (!LUTLUT[axis][0][v0] && !LUTLUT[axis][1][v1]) {
							let coord0 = [v0, v1];
							let coord1 = [v0, v1];
                            let norm0 = [v0 * 2 - 1, 0];
                            let norm1 = [0, v1 * 2 - 1];
							coord0.splice(axis, 0, 0);
							coord1.splice(axis, 0, 1);
                            norm0.splice(axis, 0, 0);
                            norm1.splice(axis, 0, 0);
                            
							body += "\tVOX_EDGE( " +
                                `${coord0[0]}, ` +
                                `${coord0[1]}, ` +
                                `${coord0[2]}, ` +
                                `${coord1[0]}, ` +
                                `${coord1[1]}, ` +
                                `${coord1[2]}, ` +
                                `${norm0[0]}, ` +
                                `${norm0[1]}, ` +
                                `${norm0[2]}, ` +
                                `${norm1[0]}, ` +
                                `${norm1[1]}, ` +
                                `${norm1[2]} )\n`;
						}
					}
				}
			}

			if (!body)
				return "";

			return label + body + "\tbreak;\n";
		}

		function genAll() {
			let str = "";
			for (let posX = 0; posX <= 1; posX++) {
				for (let negX = 0; negX <= 1; negX++) {
					for (let posY = 0; posY <= 1; posY++) {
						for (let negY = 0; negY <= 1; negY++) {
							for (let posZ = 0; posZ <= 1; posZ++) {
								for (let negZ = 0; negZ <= 1; negZ++) {
									str += generate(posX, negX, posY, negY, posZ, negZ);
								}
							}
						}
					}
				}
			}

			return str;
		}
	clang-format on */

	uint32_t voxelType = flags & b3_voxTypeMask;
	B3_ASSERT( voxelType == b3_isEdgeVoxel || voxelType == b3_isCornerVoxel );
	// corner voxels will usually only have one corner vertex, but may have
	// 2, 4, or 8 depending on the configuration of neighboring voxels.
	// A corner with 0 neighbors is a free floating cube, so all 8 corners are valid.
	// A corner voxel with 1 neighbor invalidates all corners on that face, leaving 4 valid corners.
	// A corner voxel with 2 neighbors forms an L shape, leaving 2 valid corners.
	// A corner voxel with 3 neighbors forms a proper corner, leaving only the single corner vertex.

	// edge voxels will usually have one edge, but could also have two or four depending on the configuration of
	// neighboring voxels. An edge always has at least 1 pair of opposing neighbors, and up to two remaining
	// neighbors. (3 remaining would mean a surface voxel, and all four would mean fully occluded)

	uint32_t neighborFlags = flags & b3_voxNeighborsMask;
	switch ( neighborFlags )
	{
		case b3_noNeighbors:
			VOX_CORNER( 0, 0, 0 )
			VOX_CORNER( 0, 0, 1 )
			VOX_CORNER( 0, 1, 0 )
			VOX_CORNER( 0, 1, 1 )
			VOX_CORNER( 1, 0, 0 )
			VOX_CORNER( 1, 0, 1 )
			VOX_CORNER( 1, 1, 0 )
			VOX_CORNER( 1, 1, 1 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negZNeighbor:
			VOX_CORNER( 0, 0, 1 )
			VOX_CORNER( 0, 1, 1 )
			VOX_CORNER( 1, 0, 1 )
			VOX_CORNER( 1, 1, 1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_posZNeighbor:
			VOX_CORNER( 0, 0, 0 )
			VOX_CORNER( 0, 1, 0 )
			VOX_CORNER( 1, 0, 0 )
			VOX_CORNER( 1, 1, 0 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negYNeighbor:
			VOX_CORNER( 0, 1, 0 )
			VOX_CORNER( 0, 1, 1 )
			VOX_CORNER( 1, 1, 0 )
			VOX_CORNER( 1, 1, 1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negYNeighbor | b3_negZNeighbor:
			VOX_CORNER( 0, 1, 1 )
			VOX_CORNER( 1, 1, 1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negYNeighbor | b3_posZNeighbor:
			VOX_CORNER( 0, 1, 0 )
			VOX_CORNER( 1, 1, 0 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negYNeighbor | b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_posYNeighbor:
			VOX_CORNER( 0, 0, 0 )
			VOX_CORNER( 0, 0, 1 )
			VOX_CORNER( 1, 0, 0 )
			VOX_CORNER( 1, 0, 1 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			break;
		case b3_posYNeighbor | b3_negZNeighbor:
			VOX_CORNER( 0, 0, 1 )
			VOX_CORNER( 1, 0, 1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			break;
		case b3_posYNeighbor | b3_posZNeighbor:
			VOX_CORNER( 0, 0, 0 )
			VOX_CORNER( 1, 0, 0 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			break;
		case b3_posYNeighbor | b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			break;
		case b3_posYNeighbor | b3_negYNeighbor:
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			break;
		case b3_posYNeighbor | b3_negYNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			break;
		case b3_posYNeighbor | b3_negYNeighbor | b3_posZNeighbor:
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			break;
		case b3_negXNeighbor:
			VOX_CORNER( 1, 0, 0 )
			VOX_CORNER( 1, 0, 1 )
			VOX_CORNER( 1, 1, 0 )
			VOX_CORNER( 1, 1, 1 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negXNeighbor | b3_negZNeighbor:
			VOX_CORNER( 1, 0, 1 )
			VOX_CORNER( 1, 1, 1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negXNeighbor | b3_posZNeighbor:
			VOX_CORNER( 1, 0, 0 )
			VOX_CORNER( 1, 1, 0 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negXNeighbor | b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negXNeighbor | b3_negYNeighbor:
			VOX_CORNER( 1, 1, 0 )
			VOX_CORNER( 1, 1, 1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negXNeighbor | b3_negYNeighbor | b3_negZNeighbor:
			VOX_CORNER( 1, 1, 1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negXNeighbor | b3_negYNeighbor | b3_posZNeighbor:
			VOX_CORNER( 1, 1, 0 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negXNeighbor | b3_negYNeighbor | b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0 )
			break;
		case b3_negXNeighbor | b3_posYNeighbor:
			VOX_CORNER( 1, 0, 0 )
			VOX_CORNER( 1, 0, 1 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			break;
		case b3_negXNeighbor | b3_posYNeighbor | b3_negZNeighbor:
			VOX_CORNER( 1, 0, 1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			break;
		case b3_negXNeighbor | b3_posYNeighbor | b3_posZNeighbor:
			VOX_CORNER( 1, 0, 0 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			break;
		case b3_negXNeighbor | b3_posYNeighbor | b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, -1, 0 )
			break;
		case b3_negXNeighbor | b3_posYNeighbor | b3_negYNeighbor:
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			break;
		case b3_negXNeighbor | b3_posYNeighbor | b3_negYNeighbor | b3_negZNeighbor:
			VOX_EDGE( 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1 )
			break;
		case b3_negXNeighbor | b3_posYNeighbor | b3_negYNeighbor | b3_posZNeighbor:
			VOX_EDGE( 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, -1 )
			break;
		case b3_posXNeighbor:
			VOX_CORNER( 0, 0, 0 )
			VOX_CORNER( 0, 0, 1 )
			VOX_CORNER( 0, 1, 0 )
			VOX_CORNER( 0, 1, 1 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			break;
		case b3_posXNeighbor | b3_negZNeighbor:
			VOX_CORNER( 0, 0, 1 )
			VOX_CORNER( 0, 1, 1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			break;
		case b3_posXNeighbor | b3_posZNeighbor:
			VOX_CORNER( 0, 0, 0 )
			VOX_CORNER( 0, 1, 0 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			break;
		case b3_posXNeighbor | b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			break;
		case b3_posXNeighbor | b3_negYNeighbor:
			VOX_CORNER( 0, 1, 0 )
			VOX_CORNER( 0, 1, 1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			break;
		case b3_posXNeighbor | b3_negYNeighbor | b3_negZNeighbor:
			VOX_CORNER( 0, 1, 1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			break;
		case b3_posXNeighbor | b3_negYNeighbor | b3_posZNeighbor:
			VOX_CORNER( 0, 1, 0 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			break;
		case b3_posXNeighbor | b3_negYNeighbor | b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0 )
			break;
		case b3_posXNeighbor | b3_posYNeighbor:
			VOX_CORNER( 0, 0, 0 )
			VOX_CORNER( 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			break;
		case b3_posXNeighbor | b3_posYNeighbor | b3_negZNeighbor:
			VOX_CORNER( 0, 0, 1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			break;
		case b3_posXNeighbor | b3_posYNeighbor | b3_posZNeighbor:
			VOX_CORNER( 0, 0, 0 )
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			break;
		case b3_posXNeighbor | b3_posYNeighbor | b3_posZNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0 )
			break;
		case b3_posXNeighbor | b3_posYNeighbor | b3_negYNeighbor:
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			break;
		case b3_posXNeighbor | b3_posYNeighbor | b3_negYNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 0, 1, 0, 1, 1, -1, 0, 0, 0, 0, 1 )
			break;
		case b3_posXNeighbor | b3_posYNeighbor | b3_negYNeighbor | b3_posZNeighbor:
			VOX_EDGE( 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor:
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor | b3_posZNeighbor:
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor | b3_negYNeighbor:
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor | b3_negYNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor | b3_negYNeighbor | b3_posZNeighbor:
			VOX_EDGE( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor | b3_posYNeighbor:
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor | b3_posYNeighbor | b3_negZNeighbor:
			VOX_EDGE( 0, 0, 1, 1, 0, 1, 0, -1, 0, 0, 0, 1 )
			break;
		case b3_posXNeighbor | b3_negXNeighbor | b3_posYNeighbor | b3_posZNeighbor:
			VOX_EDGE( 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1 )
			break;

		// a corner voxel should never have more than three neighboring voxels, and an edge voxel will
		// never have more than four, so any other case should be unreachable
		default:
			B3_ASSERT( 0 );
			break;
	}
#undef VOX_EDGE
#undef VOX_CORNER
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

		// compute normal and closest point on sphere.
		// contact point is midpoint between closest points
		b3FloatW dist = b3SqrtW( dist2 );
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
	// This will probably be used when we simd
	B3_UNUSED( arena );

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

			// compute normal and closest point on capsule.
			// contact point is midpoint between closest points
			float dist = sqrtf( bestDist2[j] );
			b3Vec3 normal = b3MulSV( 1.0f / dist, bestD[j] );
			b3Vec3 pVox = b3Sub( bestPVox[j], b3MulSV( radius, normal ) );
			b3Vec3 point = b3MulSV( 0.5f, b3Add( bestPCaps[j], pVox ) );

			// add a candidate point for the contact
			VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
			cp->point = b3MulSV( voxelsA.scale, point );
			cp->normal = normal;
			cp->separation = ( sqrtf( bestDist2[j] ) - radius ) * voxelsA.scale;
		}
	}
}

static void collideVoxHull( VoxCollideContext* context, b3Transform bToA, b3Arena arena )
{
	b3Matrix3 bToAMat = b3MakeMatrixFromQuat( bToA.q );
	b3Matrix3 AToBMat = b3Transpose( bToAMat );

	const b3Voxels voxelsA = context->voxelsA;
	float invScale = 1.0f / voxelsA.scale;
	float specDist = B3_SPECULATIVE_DISTANCE * invScale;
	b3Vec3 hullCenter = b3MulSV( invScale, transformPointMat( bToAMat, bToA.p, context->hullB->center ) );

	// compute the query bounds for the voxel grid. This is the AABB of the hull expanded by the speculative distance.
	b3AABB hullAABB = b3ComputeHullAABB( context->hullB, bToA );
	b3AABB queryBounds =
		computeVoxelBounds( voxelsA.data, b3Sub( b3MulSV( invScale, hullAABB.lowerBound ), b3Vec3Of( specDist ) ),
							b3Add( b3MulSV( invScale, hullAABB.upperBound ), b3Vec3Of( specDist ) ) );

	// refresh the voxel cache
	refreshVoxCache( context->contact, voxelsA.data, queryBounds );

	if ( context->contact->voxelCache.count == 0 )
		return;

	// hull corner vertices, in voxel space.
	b3Vec3* hullCorners = b3Bump( &arena, context->hullB->vertexCount * sizeof( b3Vec3 ) );
	int hullCornerCount = 0;
	// hull planes in hull space.
	const b3Plane* hullPlanes = b3GetHullPlanes( context->hullB );
	// hull edges, in hull space.
	const b3HullHalfEdge* hullEdges = b3GetHullEdges( context->hullB );
	// hull points, in hull space.
	const b3Vec3* hullPoints = b3GetHullPoints( context->hullB );

	{ // gather hull corners in voxel space
		b3AABB voxelsBounds = b3AABB_Inflate( voxelsA.data->bounds, specDist );
		for ( int i = 0; i < context->hullB->vertexCount; i++ )
		{
			b3Vec3 pt = b3MulSV( invScale, transformPointMat( bToAMat, bToA.p, hullPoints[i] ) );
			if ( b3AABB_Contains( voxelsBounds, (b3AABB){ pt, pt } ) )
			{
				hullCorners[hullCornerCount++] = pt;
			}
		}
	}

	// test hull vertices against voxels
	for ( int i = 0; i < hullCornerCount; i++ )
	{
		b3Vec3 pt = hullCorners[i];
		for ( int j = 0; j < context->contact->voxelCache.count; j++ )
		{
			b3Vec3 voxMin = context->contact->voxelCache.data[j].min;
			b3Vec3 voxMax = b3Add( voxMin, b3Vec3Of( 1.0f ) );
			uint32_t flags = context->contact->voxelCache.data[j].flags;

			// if there is a neighboring voxel which is closer to the hull vertex than this voxel, then skip this one.
			uint32_t neighborMask = getNeighborMask( pt, voxMin, voxMax );
			if ( ( flags & neighborMask ) != 0 )
				continue;

			// if the hull vertex is outside the voxel bounds, expanded by speculative distance, skip this one.
			b3AABB voxelBounds = b3AABB_Inflate( (b3AABB){ voxMin, voxMax }, specDist );
			if ( !b3AABB_Contains( voxelBounds, (b3AABB){ pt, pt } ) )
				continue;

			// use the neighbor flags to determine which faces are exposed, to use one as the normal.
			// If there's more than one, use the one that points most directly towards the hull center.
			b3Vec3 d = b3Sub( hullCenter, b3Add( voxMin, b3Vec3Of( 0.5f ) ) );

			// which axes, pointed to the hull center, is neighborless
			b3Vec3i axesMask = {
				flags & ( d.x > 0 ? b3_posXNeighbor : b3_negXNeighbor ) ? 0 : 1,
				flags & ( d.y > 0 ? b3_posYNeighbor : b3_negYNeighbor ) ? 0 : 1,
				flags & ( d.z > 0 ? b3_posZNeighbor : b3_negZNeighbor ) ? 0 : 1,
			};

			if ( axesMask.x == 0 && axesMask.y == 0 && axesMask.z == 0 )
				continue;

			b3Vec3 filtered = b3Select( axesMask, d, b3Vec3_zero );

			// compute the axis-aligned normal by finding the largest component of the filtered vector
			b3Vec3 absD = b3Abs( filtered );
			b3Vec3i axisMask = absD.x > absD.y ? ( absD.x > absD.z ? (b3Vec3i){ 1, 0, 0 } : (b3Vec3i){ 0, 0, 1 } )
											   : ( absD.y > absD.z ? (b3Vec3i){ 0, 1, 0 } : (b3Vec3i){ 0, 0, 1 } );
			b3Vec3 normal = b3Select( axisMask, b3Sign( filtered ), b3Vec3_zero );

			// add a candidate point for the contact
			VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
			cp->point = b3MulSV( voxelsA.scale, b3Clamp( pt, voxMin, voxMax ) );
			cp->normal = normal;
			cp->separation = b3Dot( normal, b3Sub( pt, b3Clamp( b3Add( voxMin, normal ), voxMin, voxMax ) ) ) * voxelsA.scale;
		}
	}

	// test corner and edge voxels against the hull
	for ( int v = 0; v < context->contact->voxelCache.count; v++ )
	{
		// skip face voxels (internal voxels were filtered out before entering the cache)
		b3VoxelCache* entry = context->contact->voxelCache.data + v;
		if ( ( entry->flags & b3_voxTypeMask ) == b3_isFaceVoxel )
			continue;

		// gather the corners and edges of the voxel
		b3Vec3 corners[8];
		b3Vec3 edges0[12];
		b3Vec3 edges1[12];
		b3Vec3 norms0[12];
		b3Vec3 norms1[12];
		int cornerCount = 0;
		int edgeCount = 0;
		processVoxel( entry->min, entry->flags, corners, &cornerCount, edges0, edges1, norms0, norms1, &edgeCount );

		// clip voxel corners against the hull, and add any that are inside as candidate points
		for ( int i = 0; i < cornerCount; i++ )
		{
			b3Vec3 pt = invTransfromPointMat( AToBMat, bToA.p, b3MulSV( voxelsA.scale, corners[i] ) );

			float bestSeparation = -FLT_MAX;
			b3Vec3 bestNormal = { 0 };
			int bestFace = -1;
			for ( int j = 0; j < context->hullB->faceCount; j++ )
			{
				b3Plane plane = hullPlanes[j];
				float sep = b3PlaneSeparation( plane, pt );
				if ( sep > B3_SPECULATIVE_DISTANCE )
				{
					bestFace = -1;
					break;
				}

				if ( sep > bestSeparation )
				{
					bestSeparation = sep;
					bestNormal = plane.normal;
					bestFace = j;
				}
			}

			// either no point was found or we found a separating axis
			if ( bestFace == -1 )
				continue;

			// add a candidate point for the contact (transform back into shape A space)
			VoxCandidatePoint* cp = context->pointBuffer + context->pointCount++;
			cp->point = transformPointMat( bToAMat, bToA.p, pt );
			cp->normal = b3Neg( b3MulMV( bToAMat, bestNormal ) );
			cp->separation = bestSeparation;
		}

		// clip voxel edges against hull edges, and add any that are inside as candidate points
		float squaredTolerance = 0.005f * 0.005f;
		for ( int i = 0; i < edgeCount; i++ )
		{
			b3Vec3 vp0 = invTransfromPointMat( AToBMat, bToA.p, b3MulSV( voxelsA.scale, edges0[i] ) );
			b3Vec3 vp1 = invTransfromPointMat( AToBMat, bToA.p, b3MulSV( voxelsA.scale, edges1[i] ) );
			b3Vec3 vn0 = b3MulMV( AToBMat, norms0[i] );
			b3Vec3 vn1 = b3MulMV( AToBMat, norms1[i] );
			b3Vec3 ve = b3Sub( vp1, vp0 );

			for ( int j = 0; j < context->hullB->edgeCount; j += 2 )
			{
				const b3HullHalfEdge* edge = hullEdges + j;
				const b3HullHalfEdge* twin = hullEdges + j + 1;
				B3_ASSERT( edge->twin == j + 1 && twin->twin == j );

				b3Vec3 hp0 = hullPoints[edge->origin];
				b3Vec3 he = b3Sub( hullPoints[twin->origin], hp0 );

				b3Vec3 hn0 = hullPlanes[edge->face].normal;
				b3Vec3 hn1 = hullPlanes[twin->face].normal;

				// TODO
			}
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
