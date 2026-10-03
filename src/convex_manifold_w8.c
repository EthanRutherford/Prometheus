// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#define B3_SIMD_WIDTH 8

#include "core.h"

#if defined( B3_SIMD_AVX2 )

#include <immintrin.h>

B3_AVX2_BEGIN

#include "convex_manifold_wide.inl"

B3_AVX2_END

#endif
