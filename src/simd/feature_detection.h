/* From
https://github.com/simdjson/simdjson/blob/master/src/internal/isadetection.h
Highly modified.

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

// on x86 systems we have two options for SIMD: SSE2 and AVX2
// Haswell+ CPUs, which were first introduced in 2013, support AVX2 instructions.
// The majority of CPUs on the market today should have AVX2 support, but there are still
// some older CPUs which only support SSE2 instructions. For this reason, we use runtime hardware
// feature detection to choose the appropriate SIMD instruction set at startup. This is only run
// on x86 systems, and only when avx2 is not specifically requested by compiler flags.
// NOTE: this file assumes it is only included on B3_SIMD_X86_DYNAMIC_DISPATCH builds, and does
// not re-check the SIMD feature flags/macro definitions.

#include <stdbool.h>
#include <stdint.h>
#if defined( _MSC_VER )
#include <intrin.h>
#elif ( defined( HAVE_GCC_GET_CPUID ) && defined( USE_GCC_GET_CPUID ) ) || defined( __FILC__ )
#include <cpuid.h>
#endif
#if defined( __linux__ )
#include <sys/auxv.h>
#endif
#if defined( _WIN32 ) && !defined( _WINDOWS_ )
// We avoid including <windows.h> (macro pollution); this matches the
// declaration in the Windows SDK (BOOL WINAPI IsProcessorFeaturePresent(DWORD)).
extern int __stdcall IsProcessorFeaturePresent( unsigned long ProcessorFeature );
#endif
#ifdef __FILC__
#include <stdfil.h>
#endif

static const uint32_t b3_cpuid_avx2_bit = 1 << 5;
static const uint64_t b3_cpuid_avx256_saved = ( (uint64_t)1 ) << 2;
static const uint32_t b3_cpuid_osxsave = ( ( (uint32_t)1 ) << 26 ) | ( ( (uint32_t)1 ) << 27 );

static inline void b3_cpuid( uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx )
{
#if defined( _MSC_VER )
	int cpu_info[4];
	__cpuidex( cpu_info, *eax, *ecx );
	*eax = cpu_info[0];
	*ebx = cpu_info[1];
	*ecx = cpu_info[2];
	*edx = cpu_info[3];
#elif ( defined( HAVE_GCC_GET_CPUID ) && defined( USE_GCC_GET_CPUID ) ) || defined( __FILC__ )
	uint32_t level = *eax;
	__get_cpuid( level, eax, ebx, ecx, edx );
#else
	uint32_t a = *eax, b, c = *ecx, d;
	asm volatile( "cpuid\n\t" : "+a"( a ), "=b"( b ), "+c"( c ), "=d"( d ) );
	*eax = a;
	*ebx = b;
	*ecx = c;
	*edx = d;
#endif
}

static inline uint64_t b3_xgetbv()
{
#if defined( _MSC_VER )
	return _xgetbv( 0 );
#elif defined( __FILC__ )
	return zxgetbv();
#else
	uint32_t xcr0_lo, xcr0_hi;
	asm volatile( "xgetbv\n\t" : "=a"( xcr0_lo ), "=d"( xcr0_hi ) : "c"( 0 ) );
	return xcr0_lo | ( ( (uint64_t)xcr0_hi ) << 32 );
#endif
}

static inline bool b3_supportsAVX2()
{
	uint32_t eax, ebx, ecx, edx;

	// EBX for EAX=0x1
	eax = 0x1;
	ecx = 0x0;
	b3_cpuid( &eax, &ebx, &ecx, &edx );

	if ( ( ecx & b3_cpuid_osxsave ) != b3_cpuid_osxsave )
	{
		return false;
	}

	// xgetbv for checking if the OS saves registers
	uint64_t xcr0 = b3_xgetbv();

	if ( ( xcr0 & b3_cpuid_avx256_saved ) == 0 )
	{
		return false;
	}

	// ECX for EAX=0x7
	eax = 0x7;
	ecx = 0x0;
	b3_cpuid( &eax, &ebx, &ecx, &edx );
	return ( ebx & b3_cpuid_avx2_bit ) != 0;
}
