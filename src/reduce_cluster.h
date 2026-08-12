#pragma once

#include "arena_allocator.h"

#include "box3d/types.h"

int b3ReduceCluster( b3LocalManifoldPoint* points, int count1, b3Vec3 normal, b3Arena arena );
