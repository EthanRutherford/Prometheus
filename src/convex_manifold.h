#pragma once

#include "manifold.h"

#include "box3d/collision.h"

b3AxisQuery b3ComputeSeparatingAxisW4( const b3HullData* hullA, const b3HullData* hullB, b3Transform xfB, bool earlyReturn );
b3AxisQuery b3ComputeSeparatingAxisW8( const b3HullData* hullA, const b3HullData* hullB, b3Transform xfB, bool earlyReturn );
