// SPDX-FileCopyrightText: 2026 Ethan Rutherford
// SPDX-License-Identifier: MIT

#include "bits.h"
#include "math_internal.h"
#include "shape.h"

#include "box3d/base.h"
#include "box3d/collision.h"
#include "box3d/constants.h"

#include <stdlib.h>
#include <string.h>

static uint32_t structSize = sizeof( b3VoxelData );
static uint32_t nodeSize = sizeof( b3VoxelNode );
static uint32_t attrSize = sizeof( b3VoxelAttrs );

// This is an artifically high limit on tree height, used for fixed-size arrays in the voxel creation code.
// Each level of the tree is a 4x4x4 voxel volume, so a tree height of 12 would represent a 4^12 = 16 million voxel wide volume,
// which is obviously beyond practical limits for a voxel shape. It's oversized so that we have some extra space in the stack to
// avoid some branching logic in the voxel creation code. Most voxel shapes are likely to be within a tree height of around 5,
// which is 1024 voxels wide. With Teardown-scale 10cm voxels, that's 102.4 meters across, roughly the size of a football field
// (either kind). 4^12 would be 1.6 million meters, which is roughly 1000 miles, or nearly 4 times the length of the grand canyon.
// But I would walk 500 miles, and I would walk 500 more...
#define B3_MAX_TREE_HEIGHT 12

#define sign( x ) ( ( x > 0 ) - ( x < 0 ) )

/// Voxel coordinate encoding and decoding

// These masks interleave the bits of the integer x, y, and z coordinates of a voxel into a single 64-bit integer.
// The ordering resembles a morton code, but rather than interleaving them zyxzyx, they're interleaved zzyyxx, which
// orders the children of a node in a more traditional x + 4y + 16z order. This is not only easier to reason about for the
// average dev, but also allows for some useful bit shifting and masking operations when processing voxels en masse.
#define B3_VOXEL_X_MASK 0x30C30C30C30C30C3ULL
#define B3_VOXEL_Y_MASK 0xC30C30C30C30C30CULL
#define B3_VOXEL_Z_MASK 0x0C30C30C30C30C30ULL

uint64_t b3EncodeVoxel( uint32_t x, uint32_t y, uint32_t z )
{
	return _pdep_u64( x, B3_VOXEL_X_MASK ) | _pdep_u64( y, B3_VOXEL_Y_MASK ) | _pdep_u64( z, B3_VOXEL_Z_MASK );
}
uint32_t b3DecodeVoxelX( uint64_t code )
{
	return (uint32_t)_pext_u64( code, B3_VOXEL_X_MASK );
}
uint32_t b3DecodeVoxelY( uint64_t code )
{
	return (uint32_t)_pext_u64( code, B3_VOXEL_Y_MASK );
}
uint32_t b3DecodeVoxelZ( uint64_t code )
{
	return (uint32_t)_pext_u64( code, B3_VOXEL_Z_MASK );
}

/// Voxel helper functions

B3_INLINE bool hasChild( uint64_t occupancy, uint32_t bitIndex )
{
	return ( occupancy & ( 1ULL << bitIndex ) ) != 0;
}

B3_INLINE uint32_t getChildIndex( uint64_t occupancy, uint32_t bitIndex )
{
	return popcount_64( occupancy & ( ( 1ULL << bitIndex ) - 1 ) );
}

B3_INLINE uint32_t getChildBitIndex( b3Vec3 pos, int height )
{
	int offset = height << 1;
	b3Vec3i offsetPos = b3And( b3ShiftRight( b3ToVec3i( pos ), offset ), b3Vec3iOf( 0x3 ) );
	return offsetPos.x | ( offsetPos.y << 2 ) | ( offsetPos.z << 4 );
}

/// Voxel shape creation and destruction

B3_INLINE int compareVoxels( const void* a, const void* b )
{
	const b3VoxelDef* arg1 = (const b3VoxelDef*)a;
	const b3VoxelDef* arg2 = (const b3VoxelDef*)b;
	return ( arg1->encoded > arg2->encoded ) - ( arg1->encoded < arg2->encoded );
}

b3VoxelData* b3CreateVoxels( const b3VoxelsDef* def )
{
	// sort the voxel coordinates for efficient BVH construction, if not already sorted
	if ( !def->isPresorted )
		qsort( def->voxels, def->voxelCount, sizeof( b3VoxelDef ), compareVoxels );

	// STAGE 1: iterate through the voxel list and count the number of bytes required to store the BVH.
	// The number of nodes is equal to the number of unique parent nodes in the 64-tree hierarchy.
	// After the nodes, we also need one byte per voxel to store the material index, so the total byte count is
	// structSize + (nodeCount * nodeSize) + (def->voxelCount * attrSize).

	// Voxel coordinates encode a 64-tree hierarchy, where each 6 bits of the encoded coordinate represents a child node in the
	// tree. Since the voxel list is sorted, the last voxel in the list will have the largest encoded coordinate. The tree height
	// required to encompass all the voxels is therefore log base 64 of the largest voxel coordinate, which we can compute with
	// bit intrinsics by computing the highest set bit (log base 2) and dividing by 6 (log base 64).

	// Another way to consider this is that the nodes each represent a 4x4x4 voxel volume, so the tree height needed to encompass
	// the entire volume is log base 4 of the largest axis of the total bounds. The naive way to compute this would be to first
	// find the largest x, y, and z coordinates, then compute the log base 4 of the largest axis. However, since the bits of the
	// x, y, and z coordinates are interleaved in the encoded voxel coordinate, we can find the largest axis for any given voxel
	// by simply finding the highest set bit. Furthermore, since the voxels are sorted in ascending numeric order, the last voxel
	// in the list will naturally be the one with the highest set bit, and therefore the largest axis. Divide the bit index by
	// 3, since there are 3 axes, and then divide by 2 to convert from log base 2 to log base 4, and we have the tree height.
	// This is equivalent to dividing by 6, which is the number of bits per level in the 64-tree hierarchy.

	uint32_t treeHeight = msb_64( def->voxels[def->voxelCount - 1].encoded ) / 6;

	// node count is initialized to the tree height (plus one for the root node),
	// this represents the number of nodes between the root and the first voxel.
	// also track number of nodes per layer, to initialize offsets later.
	int nodeCount = treeHeight + 1;
	int countByLayer[B3_MAX_TREE_HEIGHT] = { 0 };
	for ( int i = 0; i < nodeCount; i++ )
		countByLayer[i] = 1;

	uint64_t lastCode = def->voxels[0].encoded;
	for ( int i = 1; i < def->voxelCount; i++ )
	{
		uint64_t code = def->voxels[i].encoded;

		// count the number of unique parent nodes between the last voxel and the current voxel.
		// the xor results in a mask containing the highest differing bits, which (divided by 6)
		// gives the number of new ancestor nodes between this voxel and the common ancestor.
		int newCount = msb_64( lastCode ^ code ) / 6;
		nodeCount += newCount;
		lastCode = code;

		// increment the count of nodes per layer affected
		for ( int layer = 0; layer < newCount; layer++ )
			countByLayer[layer]++;
	}

	// STAGE 2: allocate the voxel data and fill in the header information.
	// with the node count computed, we can now compute the total byte count and allocate the voxel data.

	size_t byteCount = b3AlignUp8( structSize );
	size_t nodeOffset = byteCount;
	byteCount += b3AlignUp8( nodeCount * nodeSize );
	size_t voxelsOffset = byteCount;
	byteCount += b3AlignUp8( def->voxelCount * attrSize );

	b3VoxelData* voxels = b3Alloc( byteCount );

	// zero initialize for determinism
	memset( voxels, 0, byteCount );

	voxels->version = B3_VOXEL_VERSION;
	voxels->byteCount = (uint32_t)byteCount;
	voxels->bounds = (b3AABB){ b3Vec3Of( INFINITY ), b3Vec3Of( -INFINITY ) };
	voxels->treeHeight = treeHeight;
	voxels->nodeOffset = (uint32_t)nodeOffset;
	voxels->nodeCount = (uint32_t)nodeCount;
	voxels->voxelOffset = (uint32_t)voxelsOffset;
	voxels->voxelCount = (uint32_t)def->voxelCount;

	// STAGE 3: iterate through the voxel list and fill in the BVH nodes and material indices.
	// while we iterate through the voxel list, we can also compute the local AABB of the voxel volume,
	// which is the minimum and maximum voxel coordinates in each axis.

	// set up stack for iterative traversal of the voxel list and BVH construction.
	typedef struct StackFrame
	{
		// current node in this stack frame
		b3VoxelNode* node;
		// current offset of the next child node to be added on this layer
		uint32_t layerOffset;
	} StackFrame;

	StackFrame stack[B3_MAX_TREE_HEIGHT];
	memset( stack, 0, sizeof( stack ) );
	for ( int i = treeHeight; i > 0; i-- )
		stack[i].layerOffset = stack[i + 1].layerOffset + countByLayer[i];

	uint8_t* nodeStart = (uint8_t*)voxels + nodeOffset;
	uint8_t* attrStart = (uint8_t*)voxels + voxelsOffset;

	uint32_t height = treeHeight;
	stack[height].node = (b3VoxelNode*)( nodeStart );
	stack[height].node->childrenOffset = treeHeight == 0 ? 0 : 1 * nodeSize;
	for ( int i = 0; i < def->voxelCount; )
	{
		// walk down tree until leaf node, creating intermediate nodes along the way.
		while ( height > 0 )
		{
			b3VoxelNode* node = stack[height].node;
			uint64_t code = def->voxels[i].encoded;

			// compute the child index and set occupancy bit
			int childIndex = ( code >> ( height * 6 ) ) & 0x3F;
			node->occupancy |= ( 1ULL << childIndex );

			// create and initialize child node
			b3VoxelNode* childNode = (b3VoxelNode*)( nodeStart ) + stack[height].layerOffset;
			stack[height].layerOffset++;

			// push child node onto stack and continue down the tree
			stack[--height].node = childNode;
			childNode->childrenOffset = height == 0 ? i * attrSize : stack[height].layerOffset * nodeSize;
		}

		// at leaf node; add voxels until we reach a voxel with a different parent node
		while ( height == 0 )
		{
			uint64_t code = def->voxels[i].encoded;
			uint32_t bitIndex = code & 0x3F;

			stack[height].node->occupancy |= 1ULL << bitIndex;
			b3VoxelAttrs* attrs = (b3VoxelAttrs*)( attrStart ) + i;
			attrs->matIndex = def->voxels[i].matIndex;

			// update bounds
			b3Vec3 voxelPos = { (float)b3DecodeVoxelX( code ), (float)b3DecodeVoxelY( code ), (float)b3DecodeVoxelZ( code ) };
			voxels->bounds.lowerBound = b3Min( voxels->bounds.lowerBound, voxelPos );
			voxels->bounds.upperBound = b3Max( voxels->bounds.upperBound, b3Add( voxelPos, b3Vec3Of( 1.0f ) ) );

			// compute the height of the common ancestor between this voxel and the next voxel. if the height is greater than
			// zero, we jump back up the tree to the common ancestor and continue from there.
			height = ++i < def->voxelCount ? msb_64( code ^ def->voxels[i].encoded ) / 6 : treeHeight + 1;
		}
	}

	voxels->hash = 0;
	voxels->hash = b3NonZeroHash( b3Hash( B3_HASH_INIT, (uint8_t*)voxels, voxels->byteCount ) );

	return voxels;
}

void b3DestroyVoxels( b3VoxelData* voxels )
{
	b3Free( voxels, voxels->byteCount );
}

/// Voxel shape property computation

typedef struct MassAppenderContext
{
	const b3Voxels* shape;
	float baseDensity;
	b3SurfaceMaterial* materials;
	b3MassData massData;
	b3MassData* voxelMasses;
} MassAppenderContext;

void MassAppender( uint64_t code, uint32_t index, void* context )
{
	MassAppenderContext* ma = (MassAppenderContext*)context;
	b3MassData* voxelMass = ma->voxelMasses + index;

	float x = ( (float)b3DecodeVoxelX( code ) + 0.5f ) * ma->shape->scale;
	float y = ( (float)b3DecodeVoxelY( code ) + 0.5f ) * ma->shape->scale;
	float z = ( (float)b3DecodeVoxelZ( code ) + 0.5f ) * ma->shape->scale;
	voxelMass->center = (b3Vec3){ x, y, z };

	b3VoxelAttrs* attrs = (b3VoxelAttrs*)( (uint8_t*)ma->shape->data + ma->shape->data->voxelOffset ) + index;
	float density = ma->materials == NULL ? ma->baseDensity : ma->materials[attrs->matIndex].density;
	voxelMass->mass = density * ma->shape->scale * ma->shape->scale * ma->shape->scale;

	float ixx = voxelMass->mass * ma->shape->scale * ma->shape->scale / 6.0f;
	voxelMass->inertia = b3MakeDiagonalMatrix( ixx, ixx, ixx );

	ma->massData.mass += voxelMass->mass;
	ma->massData.center = b3MulAdd( ma->massData.center, voxelMass->mass, voxelMass->center );
}

b3MassData b3ComputeVoxelMass( const b3Voxels* shape, float baseDensity, b3SurfaceMaterial* materials )
{
	// compute the mass properties by iterating through voxels, computing the properties
	// per voxel, and then merging those properties into a final result.
	MassAppenderContext context = { shape, baseDensity, materials, { 0 }, { 0 } };
	context.voxelMasses = b3Alloc( shape->data->voxelCount * sizeof( b3MassData ) );
	memset( context.voxelMasses, 0, shape->data->voxelCount * sizeof( b3MassData ) );
	b3IterateVoxels( shape->data, MassAppender, &context );

	float invMass = 1.0f / context.massData.mass;
	context.massData.center = b3MulSV( invMass, context.massData.center );

	for ( uint32_t i = 0; i < shape->data->voxelCount; i++ )
	{
		b3MassData* voxelMass = context.voxelMasses + i;
		b3Vec3 offset = b3Sub( context.massData.center, voxelMass->center );
		b3Matrix3 inertia = b3AddMM( voxelMass->inertia, b3Steiner( voxelMass->mass, offset ) );
		context.massData.inertia = b3AddMM( context.massData.inertia, inertia );
	}

	b3Free( context.voxelMasses, shape->data->voxelCount * sizeof( b3MassData ) );

	return context.massData;
}

b3AABB b3ComputeVoxelAABB( const b3Voxels* shape, b3Transform transform )
{
	// theoretically we could compute a tighter AABB by iterating through the voxels, but even for
	// hulls and meshes box3D uses the following method instead. Narrow-phase collision, or ray casting,
	// will test intersection against the oriented bounding box before advancing to the actual intersection
	// logic, so the tradeoff most likely favors the faster AABB computation over the more accurate one.
	b3AABB bounds = { b3MulSV( shape->scale, shape->data->bounds.lowerBound ),
					  b3MulSV( shape->scale, shape->data->bounds.upperBound ) };
	return b3AABB_Transform( transform, bounds );
}

/// Voxel shape intersection tests

bool b3OverlapVoxels( const b3Voxels* shape, b3Transform shapeTransform, const b3ShapeProxy* proxy )
{
	// TODO: implement
	printf( "b3OverlapVoxels: not implemented\n" );
	return false;
}

float rayBoxIntersect( b3Vec3 origin, b3Vec3 invDir, b3AABB bounds )
{
	b3Vec3 t1 = b3Mul( b3Sub( bounds.lowerBound, origin ), invDir );
	b3Vec3 t2 = b3Mul( b3Sub( bounds.upperBound, origin ), invDir );
	b3Vec3 tminDir = b3Min( t1, t2 );
	b3Vec3 tmaxDir = b3Max( t1, t2 );

	float tmin = max( tminDir.x, max( tminDir.y, tminDir.z ) );
	float tmax = min( tmaxDir.x, min( tmaxDir.y, tmaxDir.z ) );

	return ( tmax >= 0 && tmax >= tmin ) ? max( tmin, 0.0f ) : INFINITY;
}

float rayMarchVoxels( b3CastOutput* output, const b3VoxelData* shape, b3Vec3 origin, b3Vec3 dir, float t )
{
	b3Vec3i flipMask = { dir.x > 0, dir.y > 0, dir.z > 0 };
	uint32_t mirrorMask = ( flipMask.x ? 0x3 << 0 : 0 ) | ( flipMask.y ? 0x3 << 2 : 0 ) | ( flipMask.z ? 0x3 << 4 : 0 );
	b3Vec3 treeScale = b3Vec3Of( (float)( 1 << ( 2 * ( shape->treeHeight + 1 ) ) ) );

	origin = b3Select( flipMask, b3Sub( treeScale, origin ), origin );
	b3Vec3 exitBounds = b3Select( flipMask, b3Sub( treeScale, shape->bounds.upperBound ), shape->bounds.lowerBound );
	b3Vec3 upBounds = b3Select( flipMask, b3Sub( treeScale, shape->bounds.lowerBound ), shape->bounds.upperBound );

	b3Vec3 pos = b3Clamp( origin, exitBounds, b3Sub( upBounds, b3Vec3Of( 0.0001f ) ) );
	b3Vec3 absDir = b3Abs( dir );
	b3Vec3 invDir = { -1.0f / absDir.x, -1.0f / absDir.y, -1.0f / absDir.z };

	b3Vec3 sideDist = b3Sub( b3Floor( b3Sub( pos, b3Vec3Of( 1.0f ) ) ), pos );

	b3VoxelNode* nodeStack[B3_MAX_TREE_HEIGHT];
	memset( nodeStack, 0, sizeof( nodeStack ) );
	nodeStack[shape->treeHeight] = (b3VoxelNode*)( (uint8_t*)shape + shape->nodeOffset );

	float dist = 0.0;
	int height = shape->treeHeight;
	for ( int i = 0; i < 256; i++ )
	{
		// Descend tree until we find leaf, or the node is empty at the current position
		uint32_t childBitIndex = getChildBitIndex( pos, height ) ^ mirrorMask;
		while ( height > 0 && hasChild( nodeStack[height]->occupancy, childBitIndex ) )
		{
			uint32_t childIndex = getChildIndex( nodeStack[height]->occupancy, childBitIndex );
			uint32_t nodeOffset = nodeStack[height]->childrenOffset + childIndex * nodeSize;
			nodeStack[--height] = (b3VoxelNode*)( (uint8_t*)shape + shape->nodeOffset + nodeOffset );

			childBitIndex = getChildBitIndex( pos, height ) ^ mirrorMask;
		}
		if ( height == 0 && hasChild( nodeStack[height]->occupancy, childBitIndex ) )
		{
			// get the voxel index
			uint32_t childIndex = getChildIndex( nodeStack[height]->occupancy, childBitIndex );
			uint32_t voxelOffset = nodeStack[height]->childrenOffset + childIndex * attrSize;
			b3VoxelAttrs* attrs = (b3VoxelAttrs*)( (uint8_t*)shape + shape->voxelOffset + voxelOffset );

			// set final hit properties
			float tmax = min( min( sideDist.x, sideDist.y ), sideDist.z );
			output->point = b3Select( flipMask, b3Sub( treeScale, pos ), pos );
			output->normal = (b3Vec3){
				tmax >= sideDist.x ? -sign( dir.x ) : 0.0f,
				tmax >= sideDist.y ? -sign( dir.y ) : 0.0f,
				tmax >= sideDist.z ? -sign( dir.z ) : 0.0f,
			};
			output->materialIndex = attrs->matIndex;
			output->hit = true;

			return dist;
		}

		// if the current 2x2x2 subregion is empty, we can coalesce the region,
		// and step ahead by double the distance.
		uint32_t subCubeShift = ( childBitIndex & 0x2A );
		bool isEmpty = ( ( nodeStack[height]->occupancy >> subCubeShift ) & 0x00330033 ) == 0;
		uint32_t offset = height * 2 + ( isEmpty ? 1 : 0 );

		// Compute DDA distance to cell boundary, and step forward
		int scale = 1 << offset;
		b3Vec3i cellMin = b3And( b3ToVec3i( pos ), b3Vec3iOf( 0xFFFFFFFF << offset ) );
		sideDist = b3Mul( b3Sub( b3FromVec3i( cellMin ), origin ), invDir );
		dist = min( min( sideDist.x, sideDist.y ), sideDist.z );

		b3Vec3i sideMask = { sideDist.x != dist, sideDist.y != dist, sideDist.z != dist };
		b3Vec3 neighborMax = b3FromVec3i( b3Addi( cellMin, b3MulSVi( scale, sideMask ) ) );
		pos = b3Min( b3Sub( origin, b3MulSV( dist, absDir ) ), b3Sub( neighborMax, b3Vec3Of( 0.0001f ) ) );

		// stop if we exit the object bounds
		if ( pos.x < exitBounds.x || pos.y < exitBounds.y || pos.z < exitBounds.z || dist > t )
			break;

		// find the most significant bit that doesn't match between the new pos
		// and the old pos. This is the depth of the common ancestor node we need to return to
		b3Vec3i diffPos = b3Xor( b3ToVec3i( pos ), cellMin );
		int diffHeight = msb_32( diffPos.x | diffPos.y | diffPos.z ) / 2;

		height = max( height, diffHeight );
	}

	return -1.0f;
}

b3CastOutput b3RayCastVoxels( const b3Voxels* shape, const b3RayCastInput* input )
{
	B3_ASSERT( b3IsValidRay( input ) );
	b3CastOutput output = { 0 };

	// convert the ray cast input into the local scale of the voxel grid
	b3Vec3 localOrigin = b3MulSV( 1.0f / shape->scale, input->origin );
	b3Vec3 localTranslation = b3MulSV( 1.0f / shape->scale, input->translation );

	float length;
	b3Vec3 dir = b3GetLengthAndNormalize( &length, localTranslation );
	if ( length == 0.0f )
	{
		// zero length ray, perform point query
		b3VoxelNode* node = (b3VoxelNode*)( (uint8_t*)shape->data + shape->data->nodeOffset );
		uint64_t code = b3EncodeVoxel( (uint32_t)localOrigin.x, (uint32_t)localOrigin.y, (uint32_t)localOrigin.z );
		for ( int i = shape->data->treeHeight; i > 0; i-- )
		{
			uint32_t bitIndex = ( code >> ( i * 6 ) ) & 0x3F;
			if ( !( node->occupancy & ( 1ULL << bitIndex ) ) )
				return output; // no intersection

			uint32_t childIndex = popcount_64( node->occupancy & ( ( 1ULL << bitIndex ) - 1 ) );
			node = (b3VoxelNode*)( (uint8_t*)shape->data + shape->data->nodeOffset + node->childrenOffset ) + childIndex;
		}

		uint32_t bitIndex = code & 0x3F;
		if ( !( node->occupancy & ( 1ULL << bitIndex ) ) )
			return output; // no intersection

		uint32_t childIndex = popcount_64( node->occupancy & ( ( 1ULL << bitIndex ) - 1 ) );
		b3VoxelAttrs* attrs = (b3VoxelAttrs*)( (uint8_t*)shape->data + shape->data->voxelOffset ) + childIndex;

		output.point = input->origin;
		output.materialIndex = attrs->matIndex;
		output.hit = true;
		return output;
	}

	float localMaxRayDist = b3Length( localTranslation );
	float maxDist = input->maxFraction * localMaxRayDist;
	b3Vec3 invDir = (b3Vec3){ 1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z };

	// intersect the ray with the voxel grid's local AABB to determine the entry point for the ray marcher.
	// if the ray misses, or the hit is beyond the maximum distance, early out with no hit.
	float boxDist = rayBoxIntersect( localOrigin, invDir, shape->data->bounds );
	if ( boxDist > maxDist )
		return output;

	b3Vec3 entryPoint = b3Add( localOrigin, b3MulSV( boxDist, dir ) );
	float marchDist = rayMarchVoxels( &output, shape->data, entryPoint, dir, maxDist - boxDist );
	if ( !output.hit )
		return output;

	output.fraction = ( boxDist + marchDist ) / localMaxRayDist;
	output.point = b3MulSV( shape->scale, output.point );
	return output;
}

b3CastOutput b3ShapeCastVoxels( const b3Voxels* shape, const b3ShapeCastInput* input )
{
	b3CastOutput output = { 0 };

	// TODO: implement
	printf( "b3ShapeCastVoxels: not implemented\n" );

	return output;
}

int b3CollideMoverAndVoxels( b3PlaneResult* results, int capacity, const b3Voxels* shape, const b3Capsule* mover )
{
	// TODO: implement
	printf( "b3CollideMoverAndVoxels: not implemented\n" );
	return 0;
}

// Voxel iterator

void b3IterateVoxels( const b3VoxelData* voxels, b3VoxelIteratorFcn* fcn, void* context )
{
	typedef struct StackFrame
	{
		const b3VoxelNode* node;
		uint64_t occupancy;
		uint64_t code;
	} StackFrame;

	StackFrame stack[B3_MAX_TREE_HEIGHT];
	memset( stack, 0, sizeof( stack ) );

	uint8_t* nodeStart = (uint8_t*)voxels + voxels->nodeOffset;

	uint32_t height = voxels->treeHeight;
	stack[height].node = (const b3VoxelNode*)( (const uint8_t*)voxels + voxels->nodeOffset );
	stack[height].occupancy = stack[height].node->occupancy;
	while ( height <= voxels->treeHeight )
	{
		if ( stack[height].occupancy == 0 )
		{
			height++;
			continue;
		}

		uint32_t bitIndex = countr_zero_64( stack[height].occupancy );
		stack[height].occupancy &= ~( 1ULL << bitIndex );
		uint64_t code = ( stack[height].code << 6 ) | bitIndex;
		uint32_t childrenOffset = stack[height].node->childrenOffset;
		uint32_t childIndex = popcount_64( stack[height].node->occupancy & ( ( 1ULL << bitIndex ) - 1 ) );

		if ( height == 0 )
		{
			uint32_t index = ( childrenOffset / attrSize ) + childIndex;
			fcn( code, index, context );
		}
		else
		{
			const b3VoxelNode* childNode =
				(const b3VoxelNode*)( nodeStart + childrenOffset + childIndex * sizeof( b3VoxelNode ) );
			stack[--height].node = childNode;
			stack[height].occupancy = childNode->occupancy;
			stack[height].code = ( stack[height + 1].code << 6 ) | bitIndex;
		}
	}
}
