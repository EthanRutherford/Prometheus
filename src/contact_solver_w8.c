// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "core.h"

#if defined( B3_SIMD_AVX2 )
#define B3_SIMD_WIDTH 8
#include "contact_solver_wide.inl"
#endif
