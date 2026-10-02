#pragma once

#include "manifold.h"

#include "box3d/collision.h"

bool b3BuildFaceAContact( b3LocalManifold* manifold, int capacity, const b3HullData* hullA, const b3HullData* hullB,
						  b3Transform transformBtoA, b3SeparatingAxis query, b3SATCache* cache );

bool b3BuildFaceBContact( b3LocalManifold* manifold, int capacity, const b3HullData* hullA, const b3HullData* hullB,
						  b3Transform transformBtoA, b3SeparatingAxis query, b3SATCache* cache );

bool b3BuildEdgeContact( b3LocalManifold* manifold, const b3HullData* hullA, const b3HullData* hullB,
						 b3Transform transformBtoA, b3SeparatingAxis query, b3SATCache* cache );

// Declare SIMD functions

b3AxisQuery b3ComputeSeparatingAxisW4( const b3HullData* hullA, const b3HullData* hullB, b3Transform xfB, bool earlyReturn );
b3AxisQuery b3ComputeSeparatingAxisW8( const b3HullData* hullA, const b3HullData* hullB, b3Transform xfB, bool earlyReturn );

void b3CollideHullsW4( b3LocalManifold* manifold, int capacity, const b3HullData* hullA, const b3HullData* hullB,
					   b3Transform transformBtoA, b3SATCache* cache );
void b3CollideHullsW8( b3LocalManifold* manifold, int capacity, const b3HullData* hullA, const b3HullData* hullB,
					   b3Transform transformBtoA, b3SATCache* cache );
