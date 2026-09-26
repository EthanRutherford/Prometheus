#include "body.h"
#include "contact_solver_w.h"
#include "platform.h"
#include "simd.h"

#if defined( B3_SIMD_HAS_WIDTH_8 )
#define B3_SIMD_WIDTH 8

// Wide vec2
typedef struct b3Vec2W8
{
	b3FloatW8 x, y;
} b3Vec2W8;

// Wide vec3
typedef struct b3Vec3W8
{
	b3FloatW8 X, Y, Z;
} b3Vec3W8;

// Wide quaternion
typedef struct b3QuatW8
{
	b3Vec3W8 V;
	b3FloatW8 S;
} b3QuatW8;

// Wide symmetric matrix2
typedef struct b3SymMatrix2W8
{
	b3FloatW8 cxx, cxy, cyy;
} b3SymMatrix2W8;

// Wide symmetric matrix3
typedef struct b3SymMatrix3W8
{
	b3FloatW8 cxx, cxy, cxz, cyy, cyz, czz;
} b3SymMatrix3W8;

typedef struct b3Matrix3W8
{
	b3Vec3W8 cx, cy, cz;
} b3Matrix3W8;

// s * a
static inline b3Vec3W8 b3MulSVW8( b3FloatW8 s, b3Vec3W8 a )
{
	return (b3Vec3W8){ b3MulW8( s, a.X ), b3MulW8( s, a.Y ), b3MulW8( s, a.Z ) };
}

// a - s * b
static inline b3Vec3W8 b3MulSubSVW8( b3Vec3W8 a, b3FloatW8 s, b3Vec3W8 b )
{
	return (b3Vec3W8){ b3SubW8( a.X, b3MulW8( s, b.X ) ), b3SubW8( a.Y, b3MulW8( s, b.Y ) ), b3SubW8( a.Z, b3MulW8( s, b.Z ) ) };
}

// a + s * b
static inline b3Vec3W8 b3MulAddSVW8( b3Vec3W8 a, b3FloatW8 s, b3Vec3W8 b )
{
	return (b3Vec3W8){ b3AddW8( a.X, b3MulW8( s, b.X ) ), b3AddW8( a.Y, b3MulW8( s, b.Y ) ), b3AddW8( a.Z, b3MulW8( s, b.Z ) ) };
}

// a + b
static inline b3Vec2W8 b3AddV2W8( b3Vec2W8 a, b3Vec2W8 b )
{
	return (b3Vec2W8){
		b3AddW8( a.x, b.x ),
		b3AddW8( a.y, b.y ),
	};
}

// a - b
static inline b3Vec3W8 b3SubVW8( b3Vec3W8 a, b3Vec3W8 b )
{
	return (b3Vec3W8){
		b3SubW8( a.X, b.X ),
		b3SubW8( a.Y, b.Y ),
		b3SubW8( a.Z, b.Z ),
	};
}

// a + b
static inline b3Vec3W8 b3AddVW8( b3Vec3W8 a, b3Vec3W8 b )
{
	return (b3Vec3W8){
		b3AddW8( a.X, b.X ),
		b3AddW8( a.Y, b.Y ),
		b3AddW8( a.Z, b.Z ),
	};
}

// m * a
static inline b3Vec2W8 b3MulMV2W8( b3SymMatrix2W8 m, b3Vec2W8 a )
{
	b3Vec2W8 b = {
		b3AddW8( b3MulW8( m.cxx, a.x ), b3MulW8( m.cxy, a.y ) ),
		b3AddW8( b3MulW8( m.cxy, a.x ), b3MulW8( m.cyy, a.y ) ),
	};

	return b;
}

// m * a
static inline b3Vec3W8 b3MulMVW8( b3SymMatrix3W8 m, b3Vec3W8 a )
{
	b3Vec3W8 b = {
		b3AddW8( b3MulW8( m.cxx, a.X ), b3AddW8( b3MulW8( m.cxy, a.Y ), b3MulW8( m.cxz, a.Z ) ) ),
		b3AddW8( b3MulW8( m.cxy, a.X ), b3AddW8( b3MulW8( m.cyy, a.Y ), b3MulW8( m.cyz, a.Z ) ) ),
		b3AddW8( b3MulW8( m.cxz, a.X ), b3AddW8( b3MulW8( m.cyz, a.Y ), b3MulW8( m.czz, a.Z ) ) ),
	};

	return b;
}

// a - m * b
static inline b3Vec3W8 b3MulSubMVW8( b3Vec3W8 a, b3SymMatrix3W8 m, b3Vec3W8 b )
{
	b3Vec3W8 c = {
		b3AddW8( b3MulW8( m.cxx, b.X ), b3AddW8( b3MulW8( m.cxy, b.Y ), b3MulW8( m.cxz, b.Z ) ) ),
		b3AddW8( b3MulW8( m.cxy, b.X ), b3AddW8( b3MulW8( m.cyy, b.Y ), b3MulW8( m.cyz, b.Z ) ) ),
		b3AddW8( b3MulW8( m.cxz, b.X ), b3AddW8( b3MulW8( m.cyz, b.Y ), b3MulW8( m.czz, b.Z ) ) ),
	};

	return (b3Vec3W8){ b3SubW8( a.X, c.X ), b3SubW8( a.Y, c.Y ), b3SubW8( a.Z, c.Z ) };
}

// a + m * b
static inline b3Vec3W8 b3MulAddMVW8( b3Vec3W8 a, b3SymMatrix3W8 m, b3Vec3W8 b )
{
	b3Vec3W8 c = {
		b3AddW8( b3MulW8( m.cxx, b.X ), b3AddW8( b3MulW8( m.cxy, b.Y ), b3MulW8( m.cxz, b.Z ) ) ),
		b3AddW8( b3MulW8( m.cxy, b.X ), b3AddW8( b3MulW8( m.cyy, b.Y ), b3MulW8( m.cyz, b.Z ) ) ),
		b3AddW8( b3MulW8( m.cxz, b.X ), b3AddW8( b3MulW8( m.cyz, b.Y ), b3MulW8( m.czz, b.Z ) ) ),
	};

	return (b3Vec3W8){ b3AddW8( a.X, c.X ), b3AddW8( a.Y, c.Y ), b3AddW8( a.Z, c.Z ) };
}

static inline b3FloatW8 b3DotW8( b3Vec3W8 a, b3Vec3W8 b )
{
	return b3AddW8( b3AddW8( b3MulW8( a.X, b.X ), b3MulW8( a.Y, b.Y ) ), b3MulW8( a.Z, b.Z ) );
}

static inline b3Vec3W8 b3CrossW8( b3Vec3W8 a, b3Vec3W8 b )
{
	b3Vec3W8 c;
	c.X = b3SubW8( b3MulW8( a.Y, b.Z ), b3MulW8( a.Z, b.Y ) );
	c.Y = b3SubW8( b3MulW8( a.Z, b.X ), b3MulW8( a.X, b.Z ) );
	c.Z = b3SubW8( b3MulW8( a.X, b.Y ), b3MulW8( a.Y, b.X ) );
	return c;
}

static inline b3Matrix3W8 b3MakeMatrixFromQuatW8( b3QuatW8 q )
{
	b3FloatW8 x2 = b3AddW8( q.V.X, q.V.X );
	b3FloatW8 y2 = b3AddW8( q.V.Y, q.V.Y );
	b3FloatW8 z2 = b3AddW8( q.V.Z, q.V.Z );
	b3FloatW8 xx2 = b3MulW8( q.V.X, x2 );
	b3FloatW8 yy2 = b3MulW8( q.V.Y, y2 );
	b3FloatW8 zz2 = b3MulW8( q.V.Z, z2 );
	b3FloatW8 xy2 = b3MulW8( q.V.X, y2 );
	b3FloatW8 xz2 = b3MulW8( q.V.X, z2 );
	b3FloatW8 yz2 = b3MulW8( q.V.Y, z2 );
	b3FloatW8 xw2 = b3MulW8( q.S, x2 );
	b3FloatW8 yw2 = b3MulW8( q.S, y2 );
	b3FloatW8 zw2 = b3MulW8( q.S, z2 );
	b3FloatW8 one = b3SplatW8( 1.0f );

	b3Matrix3W8 m;
	m.cx.X = b3SubW8( one, b3AddW8( yy2, zz2 ) );
	m.cx.Y = b3AddW8( xy2, zw2 );
	m.cx.Z = b3SubW8( xz2, yw2 );
	m.cy.X = b3SubW8( xy2, zw2 );
	m.cy.Y = b3SubW8( one, b3AddW8( xx2, zz2 ) );
	m.cy.Z = b3AddW8( yz2, xw2 );
	m.cz.X = b3AddW8( xz2, yw2 );
	m.cz.Y = b3SubW8( yz2, xw2 );
	m.cz.Z = b3SubW8( one, b3AddW8( xx2, yy2 ) );
	return m;
}

static inline b3Vec3W8 b3MulM3VW8( b3Matrix3W8 m, b3Vec3W8 a )
{
	b3Vec3W8 b = {
		b3AddW8( b3MulW8( m.cx.X, a.X ), b3AddW8( b3MulW8( m.cy.X, a.Y ), b3MulW8( m.cz.X, a.Z ) ) ),
		b3AddW8( b3MulW8( m.cx.Y, a.X ), b3AddW8( b3MulW8( m.cy.Y, a.Y ), b3MulW8( m.cz.Y, a.Z ) ) ),
		b3AddW8( b3MulW8( m.cx.Z, a.X ), b3AddW8( b3MulW8( m.cy.Z, a.Y ), b3MulW8( m.cz.Z, a.Z ) ) ),
	};

	return b;
}

static inline b3Vec3W8 b3InvRotateVectorW8( b3QuatW8 q, b3Vec3W8 a )
{
	b3Vec3W8 t = b3CrossW8( q.V, a );
	t = (b3Vec3W8){ b3AddW8( t.X, t.X ), b3AddW8( t.Y, t.Y ), b3AddW8( t.Z, t.Z ) };
	b3Vec3W8 u = b3CrossW8( q.V, t );
	return (b3Vec3W8){
		b3AddW8( b3SubW8( a.X, b3MulW8( q.S, t.X ) ), u.X ),
		b3AddW8( b3SubW8( a.Y, b3MulW8( q.S, t.Y ) ), u.Y ),
		b3AddW8( b3SubW8( a.Z, b3MulW8( q.S, t.Z ) ), u.Z ),
	};
}

// Soft contact constraints with sub-stepping support
// Uses fixed anchors for Jacobians for better behavior on rolling shapes (circles & capsules)
// http://mmacklin.com/smallsteps.pdf
// https://box2d.org/files/ErinCatto_SoftConstraints_GDC2011.pdf

typedef struct b3ContactConstraintPointW8
{
	b3Vec3W8 anchorAs, anchorBs;
	b3FloatW8 baseSeparations;
	b3FloatW8 normalImpulses;
	b3FloatW8 totalNormalImpulses;
	b3FloatW8 normalMasses;
	b3FloatW8 leverArms;
	b3FloatW8 relativeVelocities;
	b3FloatW8 restitutionImpulses;
} b3ContactConstraintPointW8;

// Solves eight points
typedef struct b3ContactConstraintW8
{
	// These are base 1
	int indexA[B3_SIMD_WIDTH];
	int indexB[B3_SIMD_WIDTH];

	int pointCounts[B3_SIMD_WIDTH];

	b3FloatW8 invMassA, invMassB;
	b3SymMatrix3W8 invIA, invIB;
	b3Vec3W8 normal;

	// todo test computing the tangents on the fly, at least tangent2
	b3Vec3W8 tangent1;
	b3Vec3W8 tangent2;

	// Friction centers
	b3Vec3W8 centerA, centerB;
	b3FloatW8 twistMass;
	b3FloatW8 twistImpulse;
	b3SymMatrix2W8 tangentMass;
	b3Vec2W8 frictionImpulse;
	b3SymMatrix3W8 rollingMass;
	b3Vec3W8 rollingImpulse;
	b3FloatW8 friction;
	b3FloatW8 rollingResistance;
	b3FloatW8 tangentVelocity1;
	b3FloatW8 tangentVelocity2;
	b3FloatW8 restitution;

	b3Manifold* manifolds[B3_SIMD_WIDTH];

	// todo store the maximum point count per wide constraint
	// to make this work I need zero initialization which is too
	// expensive for all the wide constraint data. Instead
	// the graph color should store the point count as a compact secondary
	// transient array with zero initialization.
	b3ContactConstraintPointW8 points[B3_MAX_MANIFOLD_POINTS];

} b3ContactConstraintW8;

int b3GetWideContactConstraintByteCountW8( void )
{
	return sizeof( b3ContactConstraintW8 );
}

// wide version of b3BodyState
typedef struct b3BodyStateW8
{
	b3Vec3W8 v;
	b3Vec3W8 w;
	b3Vec3W8 dp;
	b3QuatW8 dq;
} b3BodyStateW8;

_Static_assert( sizeof( b3BodyState ) == 64, "body state layout" );
_Static_assert( offsetof( b3BodyState, linearVelocity ) == 0 && offsetof( b3BodyState, angularVelocity ) == 16 &&
					offsetof( b3BodyState, deltaPosition ) == 32 && offsetof( b3BodyState, deltaRotation ) == 48,
				"body state layout" );

B3_FORCE_INLINE b3BodyStateW8 b3GatherBodiesW8( const b3BodyState* B3_RESTRICT states, const int* B3_RESTRICT indices )
{
	const float* identity = (const float*)&b3_identityBodyState;

	// Indices are 0 for null
	const float* p1 = indices[0] == 0 ? identity : (const float*)( states + indices[0] - 1 );
	const float* p2 = indices[1] == 0 ? identity : (const float*)( states + indices[1] - 1 );
	const float* p3 = indices[2] == 0 ? identity : (const float*)( states + indices[2] - 1 );
	const float* p4 = indices[3] == 0 ? identity : (const float*)( states + indices[3] - 1 );
	const float* p5 = indices[4] == 0 ? identity : (const float*)( states + indices[4] - 1 );
	const float* p6 = indices[5] == 0 ? identity : (const float*)( states + indices[5] - 1 );
	const float* p7 = indices[6] == 0 ? identity : (const float*)( states + indices[6] - 1 );
	const float* p8 = indices[7] == 0 ? identity : (const float*)( states + indices[7] - 1 );

	b3BodyStateW8 s;
	b3FloatW8 pad;

	b3TransposeW8( b3LoadW8( p1 ), b3LoadW8( p2 ), b3LoadW8( p3 ), b3LoadW8( p4 ), b3LoadW8( p5 ), b3LoadW8( p6 ), b3LoadW8( p7 ),
				   b3LoadW8( p8 ), &s.v.X, &s.v.Y, &s.v.Z, &pad, &s.w.X, &s.w.Y, &s.w.Z, &pad );
	b3TransposeW8( b3LoadW8( p1 + 8 ), b3LoadW8( p2 + 8 ), b3LoadW8( p3 + 8 ), b3LoadW8( p4 + 8 ), b3LoadW8( p5 + 8 ),
				   b3LoadW8( p6 + 8 ), b3LoadW8( p7 + 8 ), b3LoadW8( p8 + 8 ), &s.dp.X, &s.dp.Y, &s.dp.Z, &pad, &s.dq.V.X,
				   &s.dq.V.Y, &s.dq.V.Z, &s.dq.S );

	return s;
}

B3_FORCE_INLINE void b3StoreBodyVelocityW8( b3BodyState* B3_RESTRICT states, int index, b3FloatW8 vw )
{
	// Indices are 0 for null
	if ( index == 0 )
	{
		return;
	}

	b3BodyState* state = states + index - 1;
	uint32_t flags = state->flags;
	if ( ( flags & b3_dynamicFlag ) == 0 )
	{
		return;
	}

	b3StoreW8( (float*)state, vw );
}

// This writes only the velocities back to the solver bodies
B3_FORCE_INLINE void b3ScatterBodiesW8( b3BodyState* B3_RESTRICT states, const int* B3_RESTRICT indices,
										const b3BodyStateW8* B3_RESTRICT simdBody )
{
	// I don't use any dummy body in the body array because this will lead to multithreaded sharing and the
	// associated cache flushing.
	b3FloatW8 zero = b3ZeroW8();
	b3FloatW8 vw1, vw2, vw3, vw4, vw5, vw6, vw7, vw8;
	b3TransposeW8( simdBody->v.X, simdBody->v.Y, simdBody->v.Z, zero, simdBody->w.X, simdBody->w.Y, simdBody->w.Z, zero, &vw1,
				   &vw2, &vw3, &vw4, &vw5, &vw6, &vw7, &vw8 );

	b3StoreBodyVelocityW8( states, indices[0], vw1 );
	b3StoreBodyVelocityW8( states, indices[1], vw2 );
	b3StoreBodyVelocityW8( states, indices[2], vw3 );
	b3StoreBodyVelocityW8( states, indices[3], vw4 );
	b3StoreBodyVelocityW8( states, indices[4], vw5 );
	b3StoreBodyVelocityW8( states, indices[5], vw6 );
	b3StoreBodyVelocityW8( states, indices[6], vw7 );
	b3StoreBodyVelocityW8( states, indices[7], vw8 );
}

_Static_assert( offsetof( b3ManifoldPoint, anchorA ) == 0, "manifold point layout" );
_Static_assert( offsetof( b3ManifoldPoint, anchorB ) == 12, "manifold point layout" );
_Static_assert( offsetof( b3ManifoldPoint, separation ) == 24, "manifold point layout" );
_Static_assert( offsetof( b3ManifoldPoint, normalImpulse ) == 28, "manifold point layout" );
_Static_assert( offsetof( b3Manifold, twistImpulse ) == offsetof( b3Manifold, normal ) + 12, "manifold layout" );
_Static_assert( offsetof( b3Manifold, frictionImpulse ) == offsetof( b3Manifold, normal ) + 16, "manifold layout" );
_Static_assert( offsetof( b3Manifold, rollingImpulse ) == offsetof( b3Manifold, normal ) + 28, "manifold layout" );
_Static_assert( offsetof( b3Manifold, pointCount ) == offsetof( b3Manifold, normal ) + 40, "manifold layout" );
_Static_assert( offsetof( b3Matrix3, cy ) == 12 && offsetof( b3Matrix3, cz ) == 24 && sizeof( b3Matrix3 ) == 36,
				"matrix layout" );

static const b3Contact b3_zeroContact = { 0 };
static const b3Manifold b3_zeroManifold = { 0 };
static const b3BodySim b3_zeroBodySim = { 0 };

#define B3_GATHER_LANES_W8( wide, lanes, field )                                                                                 \
	wide = b3SetW8( lanes[0]->field, lanes[1]->field, lanes[2]->field, lanes[3]->field, lanes[4]->field, lanes[5]->field,        \
					lanes[6]->field, lanes[7]->field )

static inline b3SymMatrix3W8 b3GatherInvInertiaW8( const b3BodySim* const* simLanes )
{
	const float* i0 = &simLanes[0]->invInertiaWorld.cx.x;
	const float* i1 = &simLanes[1]->invInertiaWorld.cx.x;
	const float* i2 = &simLanes[2]->invInertiaWorld.cx.x;
	const float* i3 = &simLanes[3]->invInertiaWorld.cx.x;
	const float* i4 = &simLanes[4]->invInertiaWorld.cx.x;
	const float* i5 = &simLanes[5]->invInertiaWorld.cx.x;
	const float* i6 = &simLanes[6]->invInertiaWorld.cx.x;
	const float* i7 = &simLanes[7]->invInertiaWorld.cx.x;

	b3SymMatrix3W8 m;
	b3FloatW8 unused;
	b3TransposeW8( b3LoadW8( i0 ), b3LoadW8( i1 ), b3LoadW8( i2 ), b3LoadW8( i3 ), b3LoadW8( i4 ), b3LoadW8( i5 ), b3LoadW8( i6 ), b3LoadW8( i7 ),
				   &m.cxx, &m.cxy, &m.cxz, &unused, &m.cyy, &m.cyz, &unused, &unused );
	m.czz = b3SetW8( i0[8], i1[8], i2[8], i3[8], i4[8], i5[8], i6[8], i7[8] );
	return m;
}

static inline b3SymMatrix3W8 b3AddSymW8( b3SymMatrix3W8 a, b3SymMatrix3W8 b )
{
	return (b3SymMatrix3W8){
		b3AddW8( a.cxx, b.cxx ), b3AddW8( a.cxy, b.cxy ), b3AddW8( a.cxz, b.cxz ),
		b3AddW8( a.cyy, b.cyy ), b3AddW8( a.cyz, b.cyz ), b3AddW8( a.czz, b.czz ),
	};
}

static inline b3SymMatrix3W8 b3InvertSymW8( b3SymMatrix3W8 m )
{
	b3FloatW8 cxx = b3SubW8( b3MulW8( m.cyy, m.czz ), b3MulW8( m.cyz, m.cyz ) );
	b3FloatW8 cxy = b3SubW8( b3MulW8( m.cxz, m.cyz ), b3MulW8( m.cxy, m.czz ) );
	b3FloatW8 cxz = b3SubW8( b3MulW8( m.cxy, m.cyz ), b3MulW8( m.cxz, m.cyy ) );
	b3FloatW8 cyy = b3SubW8( b3MulW8( m.cxx, m.czz ), b3MulW8( m.cxz, m.cxz ) );
	b3FloatW8 cyz = b3SubW8( b3MulW8( m.cxy, m.cxz ), b3MulW8( m.cxx, m.cyz ) );
	b3FloatW8 czz = b3SubW8( b3MulW8( m.cxx, m.cyy ), b3MulW8( m.cxy, m.cxy ) );

	b3FloatW8 det = b3AddW8( b3MulW8( m.cxx, cxx ), b3AddW8( b3MulW8( m.cxy, cxy ), b3MulW8( m.cxz, cxz ) ) );
	b3FloatW8 valid = b3GreaterThanW8( b3AbsW8( det ), b3SplatW8( 1000.0f * FLT_MIN ) );
	b3FloatW8 invDet = b3BlendW8( b3ZeroW8(), b3DivW8( b3SplatW8( 1.0f ), det ), valid );

	return (b3SymMatrix3W8){
		b3MulW8( invDet, cxx ), b3MulW8( invDet, cxy ), b3MulW8( invDet, cxz ),
		b3MulW8( invDet, cyy ), b3MulW8( invDet, cyz ), b3MulW8( invDet, czz ),
	};
}

static inline b3Vec3W8 b3PerpW8( b3Vec3W8 a )
{
	b3FloatW8 zero = b3ZeroW8();
	b3FloatW8 half = b3SplatW8( 0.5f );
	b3FloatW8 mask = b3OrW8( b3LessThanW8( a.X, b3NegW8( half ) ), b3GreaterThanW8( a.X, half ) );

	b3Vec3W8 p;
	p.X = b3BlendW8( zero, a.Y, mask );
	p.Y = b3BlendW8( a.Z, b3NegW8( a.X ), mask );
	p.Z = b3BlendW8( b3NegW8( a.Y ), zero, mask );

	b3FloatW8 lengthSquared = b3DotW8( p, p );
	b3FloatW8 valid = b3GreaterThanW8( lengthSquared, b3SplatW8( 1000.0f * FLT_MIN ) );
	b3FloatW8 s = b3BlendW8( zero, b3DivW8( b3SplatW8( 1.0f ), b3SqrtW8( lengthSquared ) ), valid );
	return b3MulSVW8( s, p );
}

void b3PrepareContacts_ConvexW8( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( prepare_contact, "Prepare Contact", b3_colorYellow, true );
	b3World* world = context->world;
	b3BodySim* sims = context->sims;
	b3BodyState* states = context->states;
#if B3_ENABLE_VALIDATION
	b3Body* bodies = world->bodies.data;
#endif
	b3WidePrepareSpan* spans = context->widePrepareSpans;
	b3ContactConstraintW8* wideBase = context->wideConstraints;

	b3FloatW8 zeroW = b3ZeroW8();
	b3FloatW8 oneW = b3SplatW8( 1.0f );
	b3FloatW8 twoW = b3SplatW8( 2.0f );
	b3FloatW8 warmStartScale = world->enableWarmStarting ? oneW : zeroW;
	b3FloatW8 invTau = b3SplatW8( 1.0f / B3_SPECULATIVE_DISTANCE );
	b3FloatW8 minFrictionWeight = b3SplatW8( B3_MIN_FRICTION_WEIGHT );
	b3FloatW8 minDet = b3SplatW8( 1000.0f * FLT_MIN );
	bool anyRestitution = false;

	int wideIndex = block.startIndex;
	int endWideIndex = block.startIndex + block.count;

	// Find color for start index. Linear search but fast.
	int colorIndex = 0;
	while ( spans[colorIndex + 1].start <= wideIndex )
	{
		colorIndex += 1;
	}

	// Loop over block
	while ( wideIndex < endWideIndex )
	{
		int colorWideStart = spans[colorIndex].start;
		int colorWideEndIndex = b3MinInt( spans[colorIndex + 1].start, endWideIndex );
		int colorContactCount = spans[colorIndex].count;
		int* contactIds = spans[colorIndex].contacts;

		// Loop over color
		for ( ; wideIndex < colorWideEndIndex; ++wideIndex )
		{
			b3ContactConstraintW8* c = wideBase + wideIndex;
			int localWideIndex = wideIndex - colorWideStart;

			const b3Contact* contactLanes[B3_SIMD_WIDTH];
			const b3Manifold* manifoldLanes[B3_SIMD_WIDTH];
			const b3BodySim* simLanesA[B3_SIMD_WIDTH];
			const b3BodySim* simLanesB[B3_SIMD_WIDTH];
			int hitEventLanes = 0;

			for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
			{
				int contactIndex = B3_SIMD_WIDTH * localWideIndex + lane;
				if ( contactIndex < colorContactCount )
				{
					b3Contact* contact = b3Array_Get( world->contacts, contactIds[contactIndex] );
					B3_ASSERT( contact->manifoldCount == 1 );
					b3Manifold* manifold = contact->manifolds;

					int indexA = b3DecodeAwakeIndex( contact->encodedBodySimA );
					int indexB = b3DecodeAwakeIndex( contact->encodedBodySimB );

#if B3_ENABLE_VALIDATION
					b3Body* bodyA = bodies + contact->edges[0].bodyId;
					b3Body* bodyB = bodies + contact->edges[1].bodyId;
					B3_ASSERT( contact->encodedBodySimA == b3EncodeBodySimIndex( bodyA ) );
					B3_ASSERT( contact->encodedBodySimB == b3EncodeBodySimIndex( bodyB ) );
#endif

					c->indexA[lane] = indexA + 1;
					c->indexB[lane] = indexB + 1;
					c->pointCounts[lane] = manifold->pointCount;
					c->manifolds[lane] = manifold;

					contactLanes[lane] = contact;
					manifoldLanes[lane] = manifold;
					simLanesA[lane] = indexA == B3_NULL_INDEX ? &b3_zeroBodySim : sims + indexA;
					simLanesB[lane] = indexB == B3_NULL_INDEX ? &b3_zeroBodySim : sims + indexB;
					hitEventLanes |= ( contact->flags & b3_simEnableHitEvent ) != 0 ? 1 << lane : 0;
				}
				else
				{
					c->indexA[lane] = 0;
					c->indexB[lane] = 0;
					c->pointCounts[lane] = 0;
					c->manifolds[lane] = NULL;

					contactLanes[lane] = &b3_zeroContact;
					manifoldLanes[lane] = &b3_zeroManifold;
					simLanesA[lane] = &b3_zeroBodySim;
					simLanesB[lane] = &b3_zeroBodySim;
				}
			}

			b3FloatW8 mA, mB;
			B3_GATHER_LANES_W8( mA, simLanesA, invMass );
			B3_GATHER_LANES_W8( mB, simLanesB, invMass );
			b3SymMatrix3W8 iA = b3GatherInvInertiaW8( simLanesA );
			b3SymMatrix3W8 iB = b3GatherInvInertiaW8( simLanesB );
			c->invMassA = mA;
			c->invMassB = mB;
			c->invIA = iA;
			c->invIB = iB;

			b3Vec3W8 tangentVelocity;
			B3_GATHER_LANES_W8( c->friction, contactLanes, friction );
			B3_GATHER_LANES_W8( c->rollingResistance, contactLanes, rollingResistance );
			B3_GATHER_LANES_W8( c->restitution, contactLanes, restitution );
			B3_GATHER_LANES_W8( tangentVelocity.X, contactLanes, tangentVelocity.x );
			B3_GATHER_LANES_W8( tangentVelocity.Y, contactLanes, tangentVelocity.y );
			B3_GATHER_LANES_W8( tangentVelocity.Z, contactLanes, tangentVelocity.z );

			b3Vec3W8 normal, frictionImpulse, rollingImpulse;
			b3FloatW8 twistImpulse;
			{
				const float* m0 = &manifoldLanes[0]->normal.x;
				const float* m1 = &manifoldLanes[1]->normal.x;
				const float* m2 = &manifoldLanes[2]->normal.x;
				const float* m3 = &manifoldLanes[3]->normal.x;
				const float* m4 = &manifoldLanes[4]->normal.x;
				const float* m5 = &manifoldLanes[5]->normal.x;
				const float* m6 = &manifoldLanes[6]->normal.x;
				const float* m7 = &manifoldLanes[7]->normal.x;
				b3TransposeW8( b3LoadW8( m0 ), b3LoadW8( m1 ), b3LoadW8( m2 ), b3LoadW8( m3 ), b3LoadW8( m4 ), b3LoadW8( m5 ),
							   b3LoadW8( m6 ), b3LoadW8( m7 ), &normal.X, &normal.Y, &normal.Z, &twistImpulse, &frictionImpulse.X,
							   &frictionImpulse.Y, &frictionImpulse.Z, &rollingImpulse.X );
				rollingImpulse.Y = b3SetW8( m0[8], m1[8], m2[8], m3[8], m4[8], m5[8], m6[8], m7[8] );
				rollingImpulse.Z = b3SetW8( m0[9], m1[9], m2[9], m3[9], m4[9], m5[9], m6[9], m7[9] );
			}

			b3Vec3W8 tangent1 = b3PerpW8( normal );
			b3Vec3W8 tangent2 = b3CrossW8( tangent1, normal );
			c->normal = normal;
			c->tangent1 = tangent1;
			c->tangent2 = tangent2;
			c->tangentVelocity1 = b3DotW8( tangentVelocity, tangent1 );
			c->tangentVelocity2 = b3DotW8( tangentVelocity, tangent2 );

			b3FloatW8 rollingMask = b3GreaterThanW8( c->rollingResistance, zeroW );
			c->twistImpulse = b3MulW8( warmStartScale, twistImpulse );
			c->rollingImpulse.X = b3BlendW8( zeroW, b3MulW8( warmStartScale, rollingImpulse.X ), rollingMask );
			c->rollingImpulse.Y = b3BlendW8( zeroW, b3MulW8( warmStartScale, rollingImpulse.Y ), rollingMask );
			c->rollingImpulse.Z = b3BlendW8( zeroW, b3MulW8( warmStartScale, rollingImpulse.Z ), rollingMask );
			c->frictionImpulse.x = b3MulW8( warmStartScale, b3DotW8( frictionImpulse, tangent1 ) );
			c->frictionImpulse.y = b3MulW8( warmStartScale, b3DotW8( frictionImpulse, tangent2 ) );

			b3FloatW8 pointCountW =
				b3SetW8( (float)c->pointCounts[0], (float)c->pointCounts[1], (float)c->pointCounts[2], (float)c->pointCounts[3],
						 (float)c->pointCounts[4], (float)c->pointCounts[5], (float)c->pointCounts[6], (float)c->pointCounts[7] );

			b3Vec3W8 centerA = { zeroW, zeroW, zeroW };
			b3Vec3W8 centerB = { zeroW, zeroW, zeroW };
			b3FloatW8 totalFrictionWeight = zeroW;

			for ( int pointIndex = 0; pointIndex < B3_MAX_MANIFOLD_POINTS; ++pointIndex )
			{
				b3ContactConstraintPointW8* cp = c->points + pointIndex;
				b3FloatW8 pointMask = b3GreaterThanW8( pointCountW, b3SplatW8( (float)pointIndex ) );

				const float* p0 = (const float*)( manifoldLanes[0]->points + pointIndex );
				const float* p1 = (const float*)( manifoldLanes[1]->points + pointIndex );
				const float* p2 = (const float*)( manifoldLanes[2]->points + pointIndex );
				const float* p3 = (const float*)( manifoldLanes[3]->points + pointIndex );
				const float* p4 = (const float*)( manifoldLanes[4]->points + pointIndex );
				const float* p5 = (const float*)( manifoldLanes[5]->points + pointIndex );
				const float* p6 = (const float*)( manifoldLanes[6]->points + pointIndex );
				const float* p7 = (const float*)( manifoldLanes[7]->points + pointIndex );

				b3Vec3W8 rA, rB;
				b3FloatW8 separation, normalImpulse;
				b3TransposeW8( b3LoadW8( p0 ), b3LoadW8( p1 ), b3LoadW8( p2 ), b3LoadW8( p3 ), b3LoadW8( p4 ), b3LoadW8( p5 ),
							   b3LoadW8( p6 ), b3LoadW8( p7 ), &rA.X, &rA.Y, &rA.Z, &rB.X, &rB.Y, &rB.Z, &separation,
							   &normalImpulse );

				rA.X = b3BlendW8( zeroW, rA.X, pointMask );
				rA.Y = b3BlendW8( zeroW, rA.Y, pointMask );
				rA.Z = b3BlendW8( zeroW, rA.Z, pointMask );
				rB.X = b3BlendW8( zeroW, rB.X, pointMask );
				rB.Y = b3BlendW8( zeroW, rB.Y, pointMask );
				rB.Z = b3BlendW8( zeroW, rB.Z, pointMask );
				separation = b3BlendW8( zeroW, separation, pointMask );
				normalImpulse = b3BlendW8( zeroW, normalImpulse, pointMask );

				// C0 friction center decay. Needed to prevent spinning top drift (GyroscopicPrecession sample).
				// See details in b3PrepareContacts_Mesh. This code should stay in sync.
				b3FloatW8 weight = b3MinW8( b3MaxW8( b3SubW8( twoW, b3MulW8( separation, invTau ) ), minFrictionWeight ), oneW );
				weight = b3BlendW8( zeroW, weight, pointMask );
				centerA = b3MulAddSVW8( centerA, weight, rA );
				centerB = b3MulAddSVW8( centerB, weight, rB );
				totalFrictionWeight = b3AddW8( totalFrictionWeight, weight );

				cp->anchorAs = rA;
				cp->anchorBs = rB;
				cp->baseSeparations = b3SubW8( separation, b3DotW8( b3SubVW8( rB, rA ), normal ) );
				cp->normalImpulses = b3MulW8( warmStartScale, normalImpulse );
				cp->totalNormalImpulses = zeroW;
				cp->relativeVelocities = zeroW;
				cp->restitutionImpulses = zeroW;

				b3Vec3W8 rnA = b3CrossW8( rA, normal );
				b3Vec3W8 rnB = b3CrossW8( rB, normal );
				b3FloatW8 kNormal = b3AddW8( mA, mB );
				kNormal = b3AddW8( kNormal, b3DotW8( rnA, b3MulMVW8( iA, rnA ) ) );
				kNormal = b3AddW8( kNormal, b3DotW8( rnB, b3MulMVW8( iB, rnB ) ) );
				b3FloatW8 valid = b3AndW8( b3GreaterThanW8( kNormal, zeroW ), pointMask );
				cp->normalMasses = b3BlendW8( zeroW, b3DivW8( oneW, kNormal ), valid );
			}

			b3FloatW8 invWeight =
				b3BlendW8( zeroW, b3DivW8( oneW, totalFrictionWeight ), b3GreaterThanW8( totalFrictionWeight, zeroW ) );
			centerA = b3MulSVW8( invWeight, centerA );
			centerB = b3MulSVW8( invWeight, centerB );
			c->centerA = centerA;
			c->centerB = centerB;

			for ( int pointIndex = 0; pointIndex < B3_MAX_MANIFOLD_POINTS; ++pointIndex )
			{
				b3ContactConstraintPointW8* cp = c->points + pointIndex;
				b3FloatW8 pointMask = b3GreaterThanW8( pointCountW, b3SplatW8( (float)pointIndex ) );
				b3Vec3W8 d = b3SubVW8( cp->anchorAs, centerA );
				cp->leverArms = b3BlendW8( zeroW, b3SqrtW8( b3DotW8( d, d ) ), pointMask );
			}

			{
				b3Vec3W8 rtA1 = b3CrossW8( centerA, tangent1 );
				b3Vec3W8 rtA2 = b3CrossW8( centerA, tangent2 );
				b3Vec3W8 rtB1 = b3CrossW8( centerB, tangent1 );
				b3Vec3W8 rtB2 = b3CrossW8( centerB, tangent2 );
				b3Vec3W8 iArtA1 = b3MulMVW8( iA, rtA1 );
				b3Vec3W8 iArtA2 = b3MulMVW8( iA, rtA2 );
				b3Vec3W8 iBrtB1 = b3MulMVW8( iB, rtB1 );
				b3Vec3W8 iBrtB2 = b3MulMVW8( iB, rtB2 );

				b3FloatW8 kxx = b3AddW8( b3AddW8( b3AddW8( mA, mB ), b3DotW8( rtA1, iArtA1 ) ), b3DotW8( rtB1, iBrtB1 ) );
				b3FloatW8 kyy = b3AddW8( b3AddW8( b3AddW8( mA, mB ), b3DotW8( rtA2, iArtA2 ) ), b3DotW8( rtB2, iBrtB2 ) );
				b3FloatW8 kxy = b3AddW8( b3DotW8( rtA1, iArtA2 ), b3DotW8( rtB1, iBrtB2 ) );

				b3FloatW8 det = b3SubW8( b3MulW8( kxx, kyy ), b3MulW8( kxy, kxy ) );
				b3FloatW8 valid = b3GreaterThanW8( b3AbsW8( det ), minDet );
				b3FloatW8 invDet = b3BlendW8( zeroW, b3DivW8( oneW, det ), valid );
				c->tangentMass.cxx = b3MulW8( invDet, kyy );
				c->tangentMass.cxy = b3NegW8( b3MulW8( invDet, kxy ) );
				c->tangentMass.cyy = b3MulW8( invDet, kxx );
			}

			b3SymMatrix3W8 invIAB = b3AddSymW8( iA, iB );

			{
				b3FloatW8 kTwist = b3DotW8( normal, b3MulMVW8( invIAB, normal ) );
				c->twistMass = b3BlendW8( zeroW, b3DivW8( oneW, kTwist ), b3GreaterThanW8( kTwist, zeroW ) );
			}

			if ( b3AllZeroW8( c->rollingResistance ) == false )
			{
				c->rollingMass = b3InvertSymW8( invIAB );
			}
			else
			{
				c->rollingMass = (b3SymMatrix3W8){ zeroW, zeroW, zeroW, zeroW, zeroW, zeroW };
			}

			// Only sample contact point normal velocity if needed.
			bool haveRestitution = b3AnyTrueW8( b3GreaterThanW8( c->restitution, zeroW ) );
			anyRestitution = anyRestitution || haveRestitution;

			if ( hitEventLanes != 0 || haveRestitution )
			{
				b3BodyStateW8 bA = b3GatherBodiesW8( states, c->indexA );
				b3BodyStateW8 bB = b3GatherBodiesW8( states, c->indexB );

				for ( int pointIndex = 0; pointIndex < B3_MAX_MANIFOLD_POINTS; ++pointIndex )
				{
					b3ContactConstraintPointW8* cp = c->points + pointIndex;

					b3Vec3W8 vrA = b3AddVW8( bA.v, b3CrossW8( bA.w, cp->anchorAs ) );
					b3Vec3W8 vrB = b3AddVW8( bB.v, b3CrossW8( bB.w, cp->anchorBs ) );
					b3FloatW8 vn = b3DotW8( normal, b3SubVW8( vrB, vrA ) );
					cp->relativeVelocities = vn;

					if ( hitEventLanes != 0 )
					{
						float normalVelocities[B3_SIMD_WIDTH];
						b3StoreW8( normalVelocities, vn );

						for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
						{
							if ( ( hitEventLanes & ( 1 << lane ) ) != 0 && pointIndex < c->pointCounts[lane] )
							{
								c->manifolds[lane]->points[pointIndex].normalVelocity = normalVelocities[lane];
							}
						}
					}
				}
			}
		}

		// Advance to next color
		colorIndex += 1;
	}

	if ( anyRestitution )
	{
		b3AtomicStoreInt( &context->anyRestitution, 1 );
	}

	b3TracyCZoneEnd( prepare_contact );
}

void b3WarmStartContacts_ConvexW8( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( warm_start_contact, "Warm Start", b3_colorGreen, true );

	b3BodyState* states = context->states;
	b3ContactConstraintW8* constraints = context->graph->colors[block.colorIndex].wideConstraints;

	for ( int i = block.startIndex; i < block.startIndex + block.count; ++i )
	{
		b3ContactConstraintW8* c = constraints + i;
		b3BodyStateW8 bA = b3GatherBodiesW8( states, c->indexA );
		b3BodyStateW8 bB = b3GatherBodiesW8( states, c->indexB );

		int pointCount1 = b3MaxInt( c->pointCounts[0], c->pointCounts[1] );
		int pointCount2 = b3MaxInt( c->pointCounts[2], c->pointCounts[3] );
		int pointCount3 = b3MaxInt( c->pointCounts[4], c->pointCounts[5] );
		int pointCount4 = b3MaxInt( c->pointCounts[6], c->pointCounts[7] );
		int pointCount = b3MaxInt( b3MaxInt( pointCount1, pointCount2 ), b3MaxInt( pointCount3, pointCount4 ) );
		B3_VALIDATE( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3FloatW8 zeroW = b3ZeroW8();
		b3FloatW8 totalNormalImpulse = zeroW;
		b3Vec3W8 momentA = { zeroW, zeroW, zeroW };
		b3Vec3W8 momentB = { zeroW, zeroW, zeroW };

		for ( int j = 0; j < pointCount; ++j )
		{
			b3ContactConstraintPointW8* cp = c->points + j;
			b3FloatW8 normalImpulse = cp->normalImpulses;
			totalNormalImpulse = b3AddW8( totalNormalImpulse, normalImpulse );
			momentA = b3MulAddSVW8( momentA, normalImpulse, cp->anchorAs );
			momentB = b3MulAddSVW8( momentB, normalImpulse, cp->anchorBs );
		}

		b3Vec3W8 normal = c->normal;
		b3Vec3W8 frictionImpulse = b3MulSVW8( c->frictionImpulse.x, c->tangent1 );
		frictionImpulse = b3MulAddSVW8( frictionImpulse, c->frictionImpulse.y, c->tangent2 );

		b3Vec3W8 linearImpulse = b3MulAddSVW8( frictionImpulse, totalNormalImpulse, normal );

		b3Vec3W8 twistImpulse = b3MulSVW8( c->twistImpulse, normal );
		b3Vec3W8 angularImpulseA =
			b3AddVW8( b3AddVW8( b3CrossW8( momentA, normal ), b3CrossW8( c->centerA, frictionImpulse ) ), twistImpulse );
		b3Vec3W8 angularImpulseB =
			b3AddVW8( b3AddVW8( b3CrossW8( momentB, normal ), b3CrossW8( c->centerB, frictionImpulse ) ), twistImpulse );

		if ( b3AllZeroW8( c->rollingResistance ) == false )
		{
			angularImpulseA = b3AddVW8( angularImpulseA, c->rollingImpulse );
			angularImpulseB = b3AddVW8( angularImpulseB, c->rollingImpulse );
		}

		bA.w = b3MulSubMVW8( bA.w, c->invIA, angularImpulseA );
		bA.v = b3MulSubSVW8( bA.v, c->invMassA, linearImpulse );
		bB.w = b3MulAddMVW8( bB.w, c->invIB, angularImpulseB );
		bB.v = b3MulAddSVW8( bB.v, c->invMassB, linearImpulse );

		b3ScatterBodiesW8( states, c->indexA, &bA );
		b3ScatterBodiesW8( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( warm_start_contact );
}

// Solve the non-penetration constraints with the soft bias. No friction and no restitution.
void b3PushContacts_ConvexW8( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( push_contact, "Push Contact", b3_colorAliceBlue, true );

	b3BodyState* states = context->states;
	b3ContactConstraintW8* constraints = context->graph->colors[block.colorIndex].wideConstraints;
	b3FloatW8 inv_h = b3SplatW8( context->inv_h );
	b3FloatW8 contactSpeed = b3SplatW8( -context->world->contactSpeed );
	b3FloatW8 oneW = b3SplatW8( 1.0f );

	// Stiffer for static contacts to avoid bodies getting pushed through the ground. Selected per
	// lane from the null body index instead of stored per constraint.
	b3FloatW8 dynamicBiasRate = b3SplatW8( context->contactSoftness.massScale * context->contactSoftness.biasRate );
	b3FloatW8 dynamicMassScale = b3SplatW8( context->contactSoftness.massScale );
	b3FloatW8 dynamicImpulseScale = b3SplatW8( context->contactSoftness.impulseScale );
	b3FloatW8 staticBiasRate = b3SplatW8( context->staticSoftness.massScale * context->staticSoftness.biasRate );
	b3FloatW8 staticMassScale = b3SplatW8( context->staticSoftness.massScale );
	b3FloatW8 staticImpulseScale = b3SplatW8( context->staticSoftness.impulseScale );

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3ContactConstraintW8* c = constraints + wideIndex;

		int pointCount1 = b3MaxInt( c->pointCounts[0], c->pointCounts[1] );
		int pointCount2 = b3MaxInt( c->pointCounts[2], c->pointCounts[3] );
		int pointCount3 = b3MaxInt( c->pointCounts[4], c->pointCounts[5] );
		int pointCount4 = b3MaxInt( c->pointCounts[6], c->pointCounts[7] );
		int pointCount = b3MaxInt( b3MaxInt( pointCount1, pointCount2 ), b3MaxInt( pointCount3, pointCount4 ) );
		B3_VALIDATE( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3BodyStateW8 bA = b3GatherBodiesW8( states, c->indexA );
		b3BodyStateW8 bB = b3GatherBodiesW8( states, c->indexB );

		b3FloatW8 softMask = b3SoftMaskW8( c->indexA, c->indexB );
		b3FloatW8 biasRate = b3BlendW8( dynamicBiasRate, staticBiasRate, softMask );
		b3FloatW8 massScale = b3BlendW8( dynamicMassScale, staticMassScale, softMask );
		b3FloatW8 impulseScale = b3BlendW8( dynamicImpulseScale, staticImpulseScale, softMask );

		b3Vec3W8 dp = b3SubVW8( bB.dp, bA.dp );

		// Convert to normals to local space to reduce transform math.
		b3FloatW8 normalSeparation = b3DotW8( c->normal, dp );
		b3Vec3W8 normalA = b3InvRotateVectorW8( bA.dq, c->normal );
		b3Vec3W8 normalB = b3InvRotateVectorW8( bB.dq, c->normal );

		for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
		{
			b3ContactConstraintPointW8* cp = c->points + pointIndex;

			// Fixed anchor points for applying impulses
			b3Vec3W8 rA = cp->anchorAs;
			b3Vec3W8 rB = cp->anchorBs;

			b3FloatW8 s = b3AddW8( b3AddW8( normalSeparation, b3SubW8( b3DotW8( normalB, rB ), b3DotW8( normalA, rA ) ) ),
								   cp->baseSeparations );

			// Apply speculative bias if separation is greater than zero, otherwise apply soft constraint bias
			b3FloatW8 separated = b3GreaterThanW8( s, b3ZeroW8() );

			// Speculative bias - positive
			b3FloatW8 specBias = b3MulW8( s, inv_h );

			// Overlap bias - negative
			b3FloatW8 overlapBias = b3MaxW8( b3MulW8( biasRate, s ), contactSpeed );
			b3FloatW8 velocityBias = b3BlendW8( overlapBias, specBias, separated );

			b3FloatW8 pointMassScale = b3BlendW8( massScale, oneW, separated );
			b3FloatW8 pointImpulseScale = b3BlendW8( impulseScale, b3ZeroW8(), separated );

			// Relative velocity at contact
			b3Vec3W8 vrA = b3AddVW8( bA.v, b3CrossW8( bA.w, rA ) );
			b3Vec3W8 vrB = b3AddVW8( bB.v, b3CrossW8( bB.w, rB ) );
			b3FloatW8 vn = b3DotW8( b3SubVW8( vrB, vrA ), c->normal );

			// Compute normal impulse
			b3FloatW8 negImpulse = b3AddW8( b3MulW8( cp->normalMasses, b3AddW8( b3MulW8( pointMassScale, vn ), velocityBias ) ),
											b3MulW8( pointImpulseScale, cp->normalImpulses ) );

			// Clamp the accumulated impulse
			b3FloatW8 newImpulse = b3MaxW8( b3SubW8( cp->normalImpulses, negImpulse ), b3ZeroW8() );
			b3FloatW8 deltaImpulse = b3SubW8( newImpulse, cp->normalImpulses );
			cp->normalImpulses = newImpulse;

			// Apply contact impulse
			b3Vec3W8 P = b3MulSVW8( deltaImpulse, c->normal );
			bA.w = b3MulSubMVW8( bA.w, c->invIA, b3CrossW8( rA, P ) );
			bA.v = b3MulSubSVW8( bA.v, c->invMassA, P );
			bB.w = b3MulAddMVW8( bB.w, c->invIB, b3CrossW8( rB, P ) );
			bB.v = b3MulAddSVW8( bB.v, c->invMassB, P );
		}

		b3ScatterBodiesW8( states, c->indexA, &bA );
		b3ScatterBodiesW8( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( push_contact );
}

// Solve the normal constraint, friction, and rolling resistance.
void b3SolveContacts_ConvexW8( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( solve_contact, "Solve Contact", b3_colorAliceBlue, true );

	b3BodyState* states = context->states;
	b3ContactConstraintW8* constraints = context->graph->colors[block.colorIndex].wideConstraints;
	b3FloatW8 inv_h = b3SplatW8( context->inv_h );
	b3FloatW8 oneW = b3SplatW8( 1.0f );
	b3FloatW8 epsilonW = b3SplatW8( FLT_EPSILON );

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3ContactConstraintW8* c = constraints + wideIndex;

		int pointCount1 = b3MaxInt( c->pointCounts[0], c->pointCounts[1] );
		int pointCount2 = b3MaxInt( c->pointCounts[2], c->pointCounts[3] );
		int pointCount3 = b3MaxInt( c->pointCounts[4], c->pointCounts[5] );
		int pointCount4 = b3MaxInt( c->pointCounts[6], c->pointCounts[7] );
		int pointCount = b3MaxInt( b3MaxInt( pointCount1, pointCount2 ), b3MaxInt( pointCount3, pointCount4 ) );
		B3_VALIDATE( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3BodyStateW8 bA = b3GatherBodiesW8( states, c->indexA );
		b3BodyStateW8 bB = b3GatherBodiesW8( states, c->indexB );

		b3Vec3W8 dp = b3SubVW8( bB.dp, bA.dp );
		b3FloatW8 normalSeparation = b3DotW8( c->normal, dp );
		b3Vec3W8 normalA = b3InvRotateVectorW8( bA.dq, c->normal );
		b3Vec3W8 normalB = b3InvRotateVectorW8( bB.dq, c->normal );

		b3FloatW8 totalNormalImpulse = b3ZeroW8();
		b3FloatW8 totalTwistLimit = b3ZeroW8();

		for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
		{
			b3ContactConstraintPointW8* cp = c->points + pointIndex;

			// Fixed anchor points for applying impulses
			b3Vec3W8 rA = cp->anchorAs;
			b3Vec3W8 rB = cp->anchorBs;

			b3FloatW8 s = b3AddW8( b3AddW8( normalSeparation, b3SubW8( b3DotW8( normalB, rB ), b3DotW8( normalA, rA ) ) ),
								   cp->baseSeparations );

			// Speculative bias, positive if separated and zero if overlapped
			b3FloatW8 velocityBias = b3MaxW8( b3ZeroW8(), b3MulW8( s, inv_h ) );

			// Relative velocity at contact
			b3Vec3W8 vrA = b3AddVW8( bA.v, b3CrossW8( bA.w, rA ) );
			b3Vec3W8 vrB = b3AddVW8( bB.v, b3CrossW8( bB.w, rB ) );
			b3FloatW8 vn = b3DotW8( b3SubVW8( vrB, vrA ), c->normal );

			// Compute normal impulse
			b3FloatW8 negImpulse = b3MulW8( cp->normalMasses, b3AddW8( vn, velocityBias ) );

			// Clamp the accumulated impulse
			b3FloatW8 newImpulse = b3MaxW8( b3SubW8( cp->normalImpulses, negImpulse ), b3ZeroW8() );
			b3FloatW8 deltaImpulse = b3SubW8( newImpulse, cp->normalImpulses );
			cp->normalImpulses = newImpulse;
			cp->totalNormalImpulses = b3AddW8( cp->totalNormalImpulses, newImpulse );
			totalNormalImpulse = b3AddW8( totalNormalImpulse, newImpulse );
			totalTwistLimit = b3AddW8( totalTwistLimit, b3MulW8( cp->leverArms, newImpulse ) );

			// Apply contact impulse
			b3Vec3W8 P = b3MulSVW8( deltaImpulse, c->normal );
			bA.w = b3MulSubMVW8( bA.w, c->invIA, b3CrossW8( rA, P ) );
			bA.v = b3MulSubSVW8( bA.v, c->invMassA, P );
			bB.w = b3MulAddMVW8( bB.w, c->invIB, b3CrossW8( rB, P ) );
			bB.v = b3MulAddSVW8( bB.v, c->invMassB, P );
		}

		// Rolling resistance
		if ( b3AllZeroW8( c->rollingResistance ) == false )
		{
			// flip A/B order to negate
			b3Vec3W8 deltaImpulse = b3MulMVW8( c->rollingMass, b3SubVW8( bA.w, bB.w ) );
			b3Vec3W8 oldImpulse = c->rollingImpulse;
			c->rollingImpulse = b3AddVW8( oldImpulse, deltaImpulse );

			b3FloatW8 maxImpulse = b3MulW8( c->rollingResistance, totalNormalImpulse );
			b3FloatW8 lengthSquared = b3DotW8( c->rollingImpulse, c->rollingImpulse );

			// if ( magSqr > maxLambda * maxLambda + FLT_EPSILON )
			//{
			//	c->rollingImpulse *= maxLambda / sqrtf( magSqr );
			// }

			b3FloatW8 mask = b3GreaterThanW8( lengthSquared, b3MulAddW8( epsilonW, maxImpulse, maxImpulse ) );

			// No approximate _mm_rsqrt_ps here to maintain cross-platform determinism
			b3FloatW8 normalize = b3DivW8( maxImpulse, b3AddW8( b3SqrtW8( lengthSquared ), epsilonW ) );
			b3FloatW8 scale = b3BlendW8( oneW, normalize, mask );

			// Ensure zero rolling resistance yields no impulse
			b3FloatW8 rollingMask = b3GreaterThanW8( c->rollingResistance, b3ZeroW8() );
			scale = b3BlendW8( b3ZeroW8(), scale, rollingMask );

			c->rollingImpulse = b3MulSVW8( scale, c->rollingImpulse );

			deltaImpulse = b3SubVW8( c->rollingImpulse, oldImpulse );

			bA.w = b3MulSubMVW8( bA.w, c->invIA, deltaImpulse );
			bB.w = b3MulAddMVW8( bB.w, c->invIB, deltaImpulse );
		}

		// Central twist friction
		{
			b3FloatW8 twistSpeed = b3DotW8( c->normal, b3SubVW8( bB.w, bA.w ) );
			b3FloatW8 maxLambda = b3MulW8( c->friction, totalTwistLimit );
			b3FloatW8 deltaImpulse = b3NegW8( b3MulW8( c->twistMass, twistSpeed ) );
			b3FloatW8 oldImpulse = c->twistImpulse;
			c->twistImpulse = b3SymClampW8( b3AddW8( oldImpulse, deltaImpulse ), maxLambda );
			deltaImpulse = b3SubW8( c->twistImpulse, oldImpulse );

			b3Vec3W8 L = b3MulSVW8( deltaImpulse, c->normal );
			bA.w = b3MulSubMVW8( bA.w, c->invIA, L );
			bB.w = b3MulAddMVW8( bB.w, c->invIB, L );
		}

		// Central friction
		{
			b3Vec3W8 tangent1 = c->tangent1;
			b3Vec3W8 tangent2 = c->tangent2;

			// Fixed anchor points for applying impulses
			b3Vec3W8 rA = c->centerA;
			b3Vec3W8 rB = c->centerB;

			// Relative tangent velocity at contact
			b3Vec3W8 vrA = b3AddVW8( bA.v, b3CrossW8( bA.w, rA ) );
			b3Vec3W8 vrB = b3AddVW8( bB.v, b3CrossW8( bB.w, rB ) );
			b3Vec3W8 vr = b3SubVW8( vrB, vrA );
			b3Vec2W8 vt = {
				b3SubW8( b3DotW8( vr, tangent1 ), c->tangentVelocity1 ),
				b3SubW8( b3DotW8( vr, tangent2 ), c->tangentVelocity2 ),
			};

			// Incremental tangent impulse
			b3Vec2W8 deltaImpulse = b3MulMV2W8( c->tangentMass, vt );
			deltaImpulse = (b3Vec2W8){ b3NegW8( deltaImpulse.x ), b3NegW8( deltaImpulse.y ) };
			b3Vec2W8 newImpulse = b3AddV2W8( c->frictionImpulse, deltaImpulse );

			b3FloatW8 friction = c->friction;
			b3FloatW8 maxImpulse = b3MulW8( friction, totalNormalImpulse );

			// Clamp the accumulated impulse
			b3FloatW8 lengthSquared = b3AddW8( b3MulW8( newImpulse.x, newImpulse.x ), b3MulW8( newImpulse.y, newImpulse.y ) );

			// Max impulse can be zero
			b3FloatW8 mask = b3GreaterThanW8( lengthSquared, b3MulW8( maxImpulse, maxImpulse ) );

			// No approximate _mm_rsqrt_ps here to maintain cross-platform determinism. Add epsilon to avoid divide by
			// zero.
			b3FloatW8 normalize = b3DivW8( maxImpulse, b3AddW8( b3SqrtW8( lengthSquared ), epsilonW ) );
			b3FloatW8 scale = b3BlendW8( oneW, normalize, mask );
			newImpulse = (b3Vec2W8){
				b3MulW8( scale, newImpulse.x ),
				b3MulW8( scale, newImpulse.y ),
			};

			deltaImpulse = (b3Vec2W8){
				b3SubW8( newImpulse.x, c->frictionImpulse.x ),
				b3SubW8( newImpulse.y, c->frictionImpulse.y ),
			};

			c->frictionImpulse = newImpulse;

			// Apply delta impulse
			b3Vec3W8 P = b3AddVW8( b3MulSVW8( deltaImpulse.x, tangent1 ), b3MulSVW8( deltaImpulse.y, tangent2 ) );
			bA.w = b3MulSubMVW8( bA.w, c->invIA, b3CrossW8( rA, P ) );
			bA.v = b3MulSubSVW8( bA.v, c->invMassA, P );
			bB.w = b3MulAddMVW8( bB.w, c->invIB, b3CrossW8( rB, P ) );
			bB.v = b3MulAddSVW8( bB.v, c->invMassB, P );
		}

		b3ScatterBodiesW8( states, c->indexA, &bA );
		b3ScatterBodiesW8( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( solve_contact );
}

void b3ApplyRestitution_ConvexW8( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( restitution, "Restitution", b3_colorDodgerBlue, true );

	b3BodyState* states = context->states;
	b3ContactConstraintW8* constraints = context->graph->colors[block.colorIndex].wideConstraints;
	b3FloatW8 inv_h = b3SplatW8( context->inv_h );
	b3FloatW8 negRestitutionThreshold = b3SplatW8( -context->world->restitutionThreshold );
	b3FloatW8 zeroW = b3ZeroW8();
	bool propagate = context->world->enableRestitutionPropagation;

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3ContactConstraintW8* c = constraints + wideIndex;
		if ( propagate == false && b3AllZeroW8( c->restitution ) )
		{
			continue;
		}

		int pointCount1 = b3MaxInt( c->pointCounts[0], c->pointCounts[1] );
		int pointCount2 = b3MaxInt( c->pointCounts[2], c->pointCounts[3] );
		int pointCount3 = b3MaxInt( c->pointCounts[4], c->pointCounts[5] );
		int pointCount4 = b3MaxInt( c->pointCounts[6], c->pointCounts[7] );
		int pointCount = b3MaxInt( b3MaxInt( pointCount1, pointCount2 ), b3MaxInt( pointCount3, pointCount4 ) );
		B3_VALIDATE( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3BodyStateW8 bA = b3GatherBodiesW8( states, c->indexA );
		b3BodyStateW8 bB = b3GatherBodiesW8( states, c->indexB );

		b3FloatW8 restitutionMask = b3GreaterThanW8( c->restitution, zeroW );
		b3Vec3W8 dp = b3SubVW8( bB.dp, bA.dp );
		b3Matrix3W8 dqA = b3MakeMatrixFromQuatW8( bA.dq );
		b3Matrix3W8 dqB = b3MakeMatrixFromQuatW8( bB.dq );

		for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
		{
			b3ContactConstraintPointW8* cp = c->points + pointIndex;

			b3Vec3W8 rA = cp->anchorAs;
			b3Vec3W8 rB = cp->anchorBs;

			b3FloatW8 normalMass = propagate ? cp->normalMasses : b3BlendW8( zeroW, cp->normalMasses, restitutionMask );

			b3FloatW8 compressionImpulse = b3SubW8( cp->totalNormalImpulses, cp->restitutionImpulses );
			b3FloatW8 armed =
				b3AndW8( b3AndW8( restitutionMask, b3LessThanW8( cp->relativeVelocities, negRestitutionThreshold ) ),
						 b3GreaterThanW8( compressionImpulse, zeroW ) );

			b3Vec3W8 rsA = b3MulM3VW8( dqA, rA );
			b3Vec3W8 rsB = b3MulM3VW8( dqB, rB );
			b3Vec3W8 ds = b3AddVW8( dp, b3SubVW8( rsB, rsA ) );
			b3FloatW8 s = b3AddW8( b3DotW8( c->normal, ds ), cp->baseSeparations );

			b3FloatW8 specBias = b3MaxW8( zeroW, b3MulW8( s, inv_h ) );
			b3FloatW8 velocityBias = b3BlendW8( specBias, b3MulW8( c->restitution, cp->relativeVelocities ), armed );

			b3Vec3W8 vrA = b3AddVW8( bA.v, b3CrossW8( bA.w, rA ) );
			b3Vec3W8 vrB = b3AddVW8( bB.v, b3CrossW8( bB.w, rB ) );
			b3FloatW8 vn = b3DotW8( b3SubVW8( vrB, vrA ), c->normal );

			b3FloatW8 negImpulse = b3MulW8( normalMass, b3AddW8( vn, velocityBias ) );

			b3FloatW8 newImpulse = b3MaxW8( b3SubW8( cp->normalImpulses, negImpulse ), zeroW );
			b3FloatW8 impulse = b3SubW8( newImpulse, cp->normalImpulses );

			b3FloatW8 approachImpulse =
				b3MinW8( b3MaxW8( b3NegW8( b3MulW8( normalMass, vn ) ), zeroW ), b3MaxW8( impulse, zeroW ) );
			b3FloatW8 allowance =
				b3SubW8( b3MulW8( c->restitution, b3AddW8( compressionImpulse, approachImpulse ) ), cp->restitutionImpulses );
			b3FloatW8 maxImpulse = b3AddW8( approachImpulse, b3MaxW8( allowance, zeroW ) );
			impulse = b3BlendW8( impulse, b3MinW8( impulse, maxImpulse ), armed );

			cp->normalImpulses = b3AddW8( cp->normalImpulses, impulse );
			cp->restitutionImpulses = b3AddW8( cp->restitutionImpulses, b3SubW8( impulse, approachImpulse ) );
			cp->totalNormalImpulses = b3AddW8( cp->totalNormalImpulses, impulse );

			b3Vec3W8 P = b3MulSVW8( impulse, c->normal );
			bA.w = b3MulSubMVW8( bA.w, c->invIA, b3CrossW8( rA, P ) );
			bA.v = b3MulSubSVW8( bA.v, c->invMassA, P );
			bB.w = b3MulAddMVW8( bB.w, c->invIB, b3CrossW8( rB, P ) );
			bB.v = b3MulAddSVW8( bB.v, c->invMassB, P );
		}

		b3ScatterBodiesW8( states, c->indexA, &bA );
		b3ScatterBodiesW8( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( restitution );
}

// Store impulses by contact constraint
void b3StoreImpulses_ConvexW8( b3SolverBlock block, b3StepContext* context, int workerIndex )
{
	b3TracyCZoneNC( store_impulses, "Store", b3_colorFireBrick, true );

	b3World* world = context->world;
	b3WidePrepareSpan* spans = context->widePrepareSpans;
	const b3ContactConstraintW8* wideBase = context->wideConstraints;
	b3TaskContext* taskContext = world->taskContexts.data + workerIndex;
	b3BitSet* hitEventBitSet = &taskContext->hitEventBitSet;
	bool hasHitEvents = taskContext->hasHitEvents;
	float negHitThreshold = -world->hitEventThreshold;

	int wideIndex = block.startIndex;
	int endWideIndex = block.startIndex + block.count;

	// Find color for start index
	int colorIndex = 0;
	while ( spans[colorIndex + 1].start <= wideIndex )
	{
		colorIndex += 1;
	}

	while ( wideIndex < endWideIndex )
	{
		int colorWideStart = spans[colorIndex].start;
		int colorWideEndIndex = b3MinInt( spans[colorIndex + 1].start, endWideIndex );
		int colorContactCount = spans[colorIndex].count;
		int* contactIds = spans[colorIndex].contacts;

		for ( ; wideIndex < colorWideEndIndex; ++wideIndex )
		{
			const b3ContactConstraintW8* c = wideBase + wideIndex;
			const float* frictionImpulse1 = (float*)&c->frictionImpulse.x;
			const float* frictionImpulse2 = (float*)&c->frictionImpulse.y;
			const float* tangent1X = (float*)&c->tangent1.X;
			const float* tangent1Y = (float*)&c->tangent1.Y;
			const float* tangent1Z = (float*)&c->tangent1.Z;
			const float* tangent2X = (float*)&c->tangent2.X;
			const float* tangent2Y = (float*)&c->tangent2.Y;
			const float* tangent2Z = (float*)&c->tangent2.Z;
			const float* twistImpulse = (float*)&c->twistImpulse;
			const float* rollingImpulseX = (float*)&c->rollingImpulse.X;
			const float* rollingImpulseY = (float*)&c->rollingImpulse.Y;
			const float* rollingImpulseZ = (float*)&c->rollingImpulse.Z;

			int localWideIndex = wideIndex - colorWideStart;

			for ( int lane = 0; lane < B3_SIMD_WIDTH; ++lane )
			{
				int contactIndex = B3_SIMD_WIDTH * localWideIndex + lane;
				if ( contactIndex >= colorContactCount )
				{
					break;
				}

				b3Manifold* m = c->manifolds[lane];
				if ( m == NULL )
				{
					continue;
				}

				float f1 = frictionImpulse1[lane];
				float f2 = frictionImpulse2[lane];
				m->frictionImpulse = (b3Vec3){
					f1 * tangent1X[lane] + f2 * tangent2X[lane],
					f1 * tangent1Y[lane] + f2 * tangent2Y[lane],
					f1 * tangent1Z[lane] + f2 * tangent2Z[lane],
				};
				m->twistImpulse = twistImpulse[lane];
				m->rollingImpulse = (b3Vec3){
					rollingImpulseX[lane],
					rollingImpulseY[lane],
					rollingImpulseZ[lane],
				};

				int pointCount = m->pointCount;
				for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
				{
					const b3ContactConstraintPointW8* cp = c->points + pointIndex;
					const float* normalImpulse = (float*)&cp->normalImpulses;
					const float* totalNormalImpulse = (float*)&cp->totalNormalImpulses;

					b3ManifoldPoint* mp = m->points + pointIndex;
					mp->normalImpulse = normalImpulse[lane];
					mp->totalNormalImpulse = totalNormalImpulse[lane];
				}

				int contactId = contactIds[contactIndex];
				b3Contact* contact = b3Array_Get( world->contacts, contactId );
				if ( ( contact->flags & b3_simEnableHitEvent ) != 0 )
				{
					for ( int k = 0; k < pointCount; ++k )
					{
						b3ManifoldPoint* mp = m->points + k;

						// Need to check total impulse because the point may be speculative and not colliding
						if ( mp->normalVelocity < negHitThreshold && mp->totalNormalImpulse > 0.0f )
						{
							b3SetBit( hitEventBitSet, contact->contactId );
							hasHitEvents = true;
							break;
						}
					}
				}
			}
		}

		colorIndex += 1;
	}

	taskContext->hasHitEvents = hasHitEvents;

	b3TracyCZoneEnd( store_impulses );
}

#undef B3_SIMD_WIDTH
#endif
