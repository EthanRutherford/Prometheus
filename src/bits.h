#include <immintrin.h>
#include <stdint.h>

#if defined( __GNUC__ ) || defined( __clang__ )

static inline int countr_zero_32( uint32_t x )
{
	return x ? __builtin_ctz( x ) : 32;
}

static inline int countr_zero_64( uint64_t x )
{
	return x ? __builtin_ctzll( x ) : 64;
}

static inline int countl_zero_32( uint32_t x )
{
	return x ? __builtin_clz( x ) : 32;
}
static inline int countl_zero_64( uint64_t x )
{
	return x ? __builtin_clzll( x ) : 64;
}

static inline int popcount_32( uint32_t x )
{
	return __builtin_popcount( x );
}
static inline int popcount_64( uint64_t x )
{
	return __builtin_popcountll( x );
}

#elif defined( _MSC_VER )

#include <intrin.h>

static inline int countr_zero_32( uint32_t x )
{
	unsigned long index;
	return _BitScanForward( &index, x ) ? index : 32;
}

static inline int countr_zero_64( uint64_t x )
{
	unsigned long index;
	return _BitScanForward64( &index, x ) ? index : 64;
}

static inline int countl_zero_32( uint32_t x )
{
	unsigned long index;
	return _BitScanReverse( &index, x ) ? ( 31 - index ) : 32;
}

static inline int countl_zero_64( uint64_t x )
{
	unsigned long index;
	return _BitScanReverse64( &index, x ) ? ( 63 - index ) : 64;
}

static inline int popcount_32( uint32_t x )
{
	return __popcnt( x );
}

static inline int popcount_64( uint64_t x )
{
	return (int)__popcnt64( x );
}

#else
#error "Unsupported compiler"
#endif

static inline int lsb_32( uint32_t x )
{
	return countr_zero_32( x );
}

static inline int lsb_64( uint64_t x )
{
	return countr_zero_64( x );
}

static inline int msb_32( uint32_t x )
{
	return 31 - countl_zero_32( x );
}

static inline int msb_64( uint64_t x )
{
	return 63 - countl_zero_64( x );
}
