// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "contact_solver.h"

#include "body.h"
#include "constraint_graph.h"
#include "contact.h"
#include "core.h"
#include "math_internal.h"
#include "physics_world.h"
#include "platform.h"
#include "simd.h"
#include "solver_set.h"

#if B3_ENABLE_VALIDATION
#include "shape.h"
#endif

// contact separation for sub-stepping
// s = s0 + dot(cB + rB - cA - rA, normal)
// normal is held constant
// body positions c can translation and anchors r can rotate
// s(t) = s0 + dot(cB(t) + rB(t) - cA(t) - rA(t), normal)
// s(t) = s0 + dot(cB0 + dpB + rot(dqB, rB0) - cA0 - dpA - rot(dqA, rA0), normal)
// s(t) = s0 + dot(cB0 - cA0, normal) + dot(dpB - dpA + rot(dqB, rB0) - rot(dqA, rA0), normal)
// s_base = s0 + dot(cB0 - cA0, normal)

// Prepare a mesh constraints
void b3PrepareContacts_Mesh( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( prepare_contact, "Prepare Contact", b3_colorYellow, true );

	b3World* world = context->world;
	b3BodySim* bodySims = context->sims;
	b3BodyState* states = context->states;

	float warmStartScale = world->enableWarmStarting ? 1.0f : 0.0f;
	bool anyRestitution = false;

	// Used for friction center weighting.
	float invTau = 1.0f / B3_SPECULATIVE_DISTANCE;

	// Need to use spans in order to find the associated b2Contact, which is per color
	b3ContactPrepareSpan* spans = context->contactPrepareSpans;
	b3ManifoldConstraint* manifoldBase = context->manifoldConstraints;
	b3ContactConstraint* base = context->contactConstraints;

	// Overflow constraints are stored separately
	if ( block.blockType == b3_overflowBlock )
	{
		b3GraphColor* overflow = world->constraintGraph.colors + B3_OVERFLOW_INDEX;
		spans = context->overflowSpans;
		manifoldBase = overflow->manifoldConstraints;
		base = overflow->contactConstraints;
	}

	int index = block.startIndex;
	int endIndex = block.startIndex + block.count;

	// Find color for start index. Linear search but fast.
	int colorIndex = 0;
	while ( spans[colorIndex + 1].start <= index )
	{
		colorIndex += 1;
	}

	// Loop over block
	while ( index < endIndex )
	{
		int colorStart = spans[colorIndex].start;
		int colorEndIndex = b3MinInt( spans[colorIndex + 1].start, endIndex );
		b3ContactSpec* specs = spans[colorIndex].contacts;

		// Loop over color
		for ( ; index < colorEndIndex; ++index )
		{
			b3ContactConstraint* contactConstraint = base + index;

			int localIndex = index - colorStart;
			B3_ASSERT( 0 <= localIndex && localIndex < spans[colorIndex].count );
			int contactId = specs[localIndex].contactId;
			b3Contact* contact = b3Array_Get( world->contacts, contactId );
			B3_ASSERT( contact->contactId == contactId );

			int indexA = b3DecodeAwakeIndex( contact->encodedBodySimA );
			int indexB = b3DecodeAwakeIndex( contact->encodedBodySimB );

#if B3_ENABLE_VALIDATION
			{
				b3Body* bodyA = b3Array_Get( world->bodies, contact->edges[0].bodyId );
				b3Body* bodyB = b3Array_Get( world->bodies, contact->edges[1].bodyId );
				B3_ASSERT( contact->encodedBodySimA == b3EncodeBodySimIndex( bodyA ) );
				B3_ASSERT( contact->encodedBodySimB == b3EncodeBodySimIndex( bodyB ) );
			}
#endif

			// Body A data
			float mA;
			b3Matrix3 iA;

			if ( indexA == B3_NULL_INDEX )
			{
				mA = 0.0f;
				iA = b3Mat3_zero;
			}
			else
			{
				b3BodySim* simA = bodySims + indexA;
				mA = simA->invMass;
				iA = simA->invInertiaWorld;
			}

			// Body B data
			float mB;
			b3Matrix3 iB;

			if ( indexB == B3_NULL_INDEX )
			{
				mB = 0.0f;
				iB = b3Mat3_zero;
			}
			else
			{
				b3BodySim* simB = bodySims + indexB;
				mB = simB->invMass;
				iB = simB->invInertiaWorld;
			}

			int manifoldCount = contact->manifoldCount;
			contactConstraint->contact = contact;
			contactConstraint->manifoldCount = manifoldCount;
			contactConstraint->indexA = indexA;
			contactConstraint->indexB = indexB;
			contactConstraint->invIA = iA;
			contactConstraint->invMassA = mA;
			contactConstraint->invIB = iB;
			contactConstraint->invMassB = mB;
			contactConstraint->rollingMass = b3InvertMatrix( b3AddMM( iA, iB ) );
			contactConstraint->softness =
				( contact->flags & b3_contactStaticFlag ) != 0 ? context->staticSoftness : context->contactSoftness;
			contactConstraint->friction = contact->friction;
			contactConstraint->restitution = contact->restitution;
			contactConstraint->rollingResistance = contact->rollingResistance;

			// Only sample contact point normal velocity if needed.
			bool haveRestitution = contact->restitution > 0.0f;
			bool hitEvents = ( contact->flags & b3_simEnableHitEvent ) != 0;
			bool sampleVelocity = haveRestitution || hitEvents;
			anyRestitution = anyRestitution || haveRestitution;

			b3Vec3 vA = b3Vec3_zero;
			b3Vec3 wA = b3Vec3_zero;
			b3Vec3 vB = b3Vec3_zero;
			b3Vec3 wB = b3Vec3_zero;

			if ( sampleVelocity )
			{
				if ( indexA != B3_NULL_INDEX )
				{
					vA = states[indexA].linearVelocity;
					wA = states[indexA].angularVelocity;
				}

				if ( indexB != B3_NULL_INDEX )
				{
					vB = states[indexB].linearVelocity;
					wB = states[indexB].angularVelocity;
				}
			}

			b3ManifoldConstraint* manifoldConstraints = manifoldBase + specs[localIndex].manifoldStart;
			contactConstraint->constraints = manifoldConstraints;

			for ( int manifoldIndex = 0; manifoldIndex < manifoldCount; ++manifoldIndex )
			{
				b3Manifold* manifold = contact->manifolds + manifoldIndex;
				b3ManifoldConstraint* constraint = manifoldConstraints + manifoldIndex;
				int pointCount = manifold->pointCount;
				b3Vec3 normal = manifold->normal;
				b3Vec3 tangent1 = b3Perp( normal );
				b3Vec3 tangent2 = b3Cross( tangent1, normal );

				constraint->pointCount = pointCount;
				constraint->normal = normal;
				constraint->tangent1 = tangent1;
				constraint->tangent2 = tangent2;

				// Stiffer for static contacts to avoid bodies getting pushed through the ground
				constraint->tangentVelocity1 = b3Dot( contact->tangentVelocity, constraint->tangent1 );
				constraint->tangentVelocity2 = b3Dot( contact->tangentVelocity, constraint->tangent2 );

				b3Vec3 centerA = b3Vec3_zero;
				b3Vec3 centerB = b3Vec3_zero;
				float totalFrictionWeight = 0.0f;

				for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
				{
					b3ManifoldConstraintPoint* cp = constraint->points + pointIndex;

					// Copy data from manifold point
					b3ManifoldPoint* mp = manifold->points + pointIndex;
					cp->rA = mp->anchorA;
					cp->rB = mp->anchorB;

					float s = mp->separation;
					cp->baseSeparation = s - b3Dot( b3Sub( cp->rB, cp->rA ), normal );
					cp->normalImpulse = warmStartScale * mp->normalImpulse;
					cp->totalNormalImpulse = 0.0f;
					cp->restitutionImpulse = 0.0f;

					b3Vec3 rA = cp->rA;
					b3Vec3 rB = cp->rB;

					b3Vec3 rnA = b3Cross( rA, normal );
					b3Vec3 rnB = b3Cross( rB, normal );
					float kNormal = mA + mB + b3Dot( rnA, b3MulMV( iA, rnA ) ) + b3Dot( rnB, b3MulMV( iB, rnB ) );
					cp->normalMass = kNormal > 0.0f ? 1.0f / kNormal : 0.0f;

					// Only compute the normal velocity terms if needed.
					if ( sampleVelocity )
					{
						b3Vec3 vrA = b3Add( vA, b3Cross( wA, rA ) );
						b3Vec3 vrB = b3Add( vB, b3Cross( wB, rB ) );
						float vn = b3Dot( normal, b3Sub( vrB, vrA ) );

						cp->relativeVelocity = vn;
						mp->normalVelocity = hitEvents ? vn : 0.0f;
					}
					else
					{
						cp->relativeVelocity = 0.0f;
						mp->normalVelocity = 0.0f;
					}

					// C0 friction center decay. Needed to prevent spinning top drift (GyroscopicPrecession sample).
					// Contacts with separation greater than twice the speculative distance only matter for CCD and
					// should not contribute to the friction center. They are not important for jitter reduction. Closer
					// points may begin to touch on and off, so the friction center needs to move smoothly.
					// Epsilon to avoid a branch below (or divide by zero). Small enough to get washed out normally.
					float weight = b3ClampFloat( 2.0f - s * invTau, B3_MIN_FRICTION_WEIGHT, 1.0f );
					centerA = b3MulAdd( centerA, weight, rA );
					centerB = b3MulAdd( centerB, weight, rB );
					totalFrictionWeight += weight;
				}

				float invWeight = 1.0f / totalFrictionWeight;
				centerA = b3MulSV( invWeight, centerA );
				centerB = b3MulSV( invWeight, centerB );
				constraint->centerA = centerA;
				constraint->centerB = centerB;

				for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
				{
					b3ManifoldConstraintPoint* cp = constraint->points + pointIndex;
					cp->leverArm = b3Distance( cp->rA, centerA );
				}

				b3Vec3 rtA1 = b3Cross( centerA, tangent1 );
				b3Vec3 rtA2 = b3Cross( centerA, tangent2 );
				b3Vec3 rtB1 = b3Cross( centerB, tangent1 );
				b3Vec3 rtB2 = b3Cross( centerB, tangent2 );

				{
					b3Matrix2 k;
					k.cx.x = mA + mB + b3Dot( rtA1, b3MulMV( iA, rtA1 ) ) + b3Dot( rtB1, b3MulMV( iB, rtB1 ) );
					k.cy.y = mA + mB + b3Dot( rtA2, b3MulMV( iA, rtA2 ) ) + b3Dot( rtB2, b3MulMV( iB, rtB2 ) );
					k.cx.y = k.cy.x = b3Dot( rtA1, b3MulMV( iA, rtA2 ) ) + b3Dot( rtB1, b3MulMV( iB, rtB2 ) );

					constraint->tangentMass = b3Invert2( k );
					constraint->frictionImpulse.x = warmStartScale * b3Dot( manifold->frictionImpulse, tangent1 );
					constraint->frictionImpulse.y = warmStartScale * b3Dot( manifold->frictionImpulse, tangent2 );
				}

				{
					float k = b3Dot( normal, b3MulMV( b3AddMM( iA, iB ), normal ) );
					constraint->twistMass = k > 0.0f ? 1.0f / k : 0.0f;
					constraint->twistImpulse = warmStartScale * manifold->twistImpulse;
				}

				{
					constraint->rollingImpulse = b3MulSV( warmStartScale, manifold->rollingImpulse );
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

void b3WarmStartContacts_Mesh( b3SolverBlock block, b3StepContext* context )
{
	b3World* world = context->world;
	b3GraphColor* color = world->constraintGraph.colors + block.colorIndex;
	b3SolverSet* awakeSet = b3Array_Get( world->solverSets, b3_awakeSet );
	b3BodyState* states = awakeSet->bodyStates.data;
	b3ContactConstraint* constraints = color->contactConstraints;

	// This is a dummy state to represent a static body because static bodies don't have a solver body.
	b3BodyState dummyState = b3_identityBodyState;

	int startIndex = block.startIndex;
	int endIndex = startIndex + block.count;

	for ( int constraintIndex = startIndex; constraintIndex < endIndex; ++constraintIndex )
	{
		const b3ContactConstraint* contactConstraint = constraints + constraintIndex;
		int indexA = contactConstraint->indexA;
		int indexB = contactConstraint->indexB;

		b3BodyState* stateA = indexA == B3_NULL_INDEX ? &dummyState : states + indexA;
		b3BodyState* stateB = indexB == B3_NULL_INDEX ? &dummyState : states + indexB;

		b3Vec3 vA = stateA->linearVelocity;
		b3Vec3 wA = stateA->angularVelocity;
		b3Vec3 vB = stateB->linearVelocity;
		b3Vec3 wB = stateB->angularVelocity;

		float mA = contactConstraint->invMassA;
		b3Matrix3 iA = contactConstraint->invIA;
		float mB = contactConstraint->invMassB;
		b3Matrix3 iB = contactConstraint->invIB;

		int manifoldCount = contactConstraint->manifoldCount;
		for ( int manifoldIndex = 0; manifoldIndex < manifoldCount; ++manifoldIndex )
		{
			b3ManifoldConstraint* constraint = contactConstraint->constraints + manifoldIndex;

			// Normal impulses
			b3Vec3 normal = constraint->normal;
			int pointCount = constraint->pointCount;
			for ( int j = 0; j < pointCount; ++j )
			{
				const b3ManifoldConstraintPoint* cp = constraint->points + j;

				// fixed anchors
				b3Vec3 rA = cp->rA;
				b3Vec3 rB = cp->rB;

				b3Vec3 impulse = b3MulSV( cp->normalImpulse, normal );
				wA = b3Sub( wA, b3MulMV( iA, b3Cross( rA, impulse ) ) );
				vA = b3MulSub( vA, mA, impulse );
				wB = b3Add( wB, b3MulMV( iB, b3Cross( rB, impulse ) ) );
				vB = b3MulAdd( vB, mB, impulse );
			}

			// Central friction
			{
				b3Vec3 rA = constraint->centerA;
				b3Vec3 rB = constraint->centerB;
				b3Vec3 impulse = b3MulSV( constraint->frictionImpulse.x, constraint->tangent1 );
				impulse = b3Add( impulse, b3MulSV( constraint->frictionImpulse.y, constraint->tangent2 ) );

				wA = b3Sub( wA, b3MulMV( iA, b3Cross( rA, impulse ) ) );
				vA = b3MulSub( vA, mA, impulse );
				wB = b3Add( wB, b3MulMV( iB, b3Cross( rB, impulse ) ) );
				vB = b3MulAdd( vB, mB, impulse );
			}

			// Central twist friction
			{
				b3Vec3 impulse = b3MulSV( constraint->twistImpulse, constraint->normal );
				wA = b3Sub( wA, b3MulMV( iA, impulse ) );
				wB = b3Add( wB, b3MulMV( iB, impulse ) );
			}

			// Rolling resistance
			{
				b3Vec3 impulse = constraint->rollingImpulse;
				wA = b3Sub( wA, b3MulMV( iA, impulse ) );
				wB = b3Add( wB, b3MulMV( iB, impulse ) );
			}
		}

		if ( stateA->flags & b3_dynamicFlag )
		{
			stateA->linearVelocity = vA;
			stateA->angularVelocity = wA;
		}

		if ( stateB->flags & b3_dynamicFlag )
		{
			stateB->linearVelocity = vB;
			stateB->angularVelocity = wB;
		}
	}
}

// Merged normal and friction loops. This is much more stable for the Jenga stack.
// Solve the non-penetration constraints with the soft bias. No friction and no restitution.
void b3PushContacts_Mesh( b3SolverBlock block, b3StepContext* context )
{
	b3World* world = context->world;
	b3GraphColor* color = world->constraintGraph.colors + block.colorIndex;
	b3ContactConstraint* contactConstraints = color->contactConstraints;
	b3BodyState* states = context->states;

	// This is a dummy state to represent a static body because static bodies have no solver body.
	b3BodyState dummyState = b3_identityBodyState;

	// The last block might not be full
	int startIndex = block.startIndex;
	int endIndex = startIndex + block.count;

	float inv_h = context->inv_h;
	const float contactSpeed = context->world->contactSpeed;

	for ( int i = startIndex; i < endIndex; ++i )
	{
		b3ContactConstraint* contactConstraint = contactConstraints + i;
		int manifoldCount = contactConstraint->manifoldCount;

		int indexA = contactConstraint->indexA;
		int indexB = contactConstraint->indexB;

		float mA = contactConstraint->invMassA;
		b3Matrix3 iA = contactConstraint->invIA;
		float mB = contactConstraint->invMassB;
		b3Matrix3 iB = contactConstraint->invIB;

		b3BodyState* stateA = indexA == B3_NULL_INDEX ? &dummyState : states + indexA;
		b3Vec3 vA = stateA->linearVelocity;
		b3Vec3 wA = stateA->angularVelocity;
		b3Quat dqA = stateA->deltaRotation;

		b3BodyState* stateB = indexB == B3_NULL_INDEX ? &dummyState : states + indexB;
		b3Vec3 vB = stateB->linearVelocity;
		b3Vec3 wB = stateB->angularVelocity;
		b3Quat dqB = stateB->deltaRotation;

		b3Vec3 dp = b3Sub( stateB->deltaPosition, stateA->deltaPosition );
		b3Softness softness = contactConstraint->softness;

		for ( int j = 0; j < manifoldCount; ++j )
		{
			b3ManifoldConstraint* constraint = contactConstraint->constraints + j;

			int pointCount = constraint->pointCount;
			b3Vec3 normal = constraint->normal;

			for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
			{
				b3ManifoldConstraintPoint* cp = constraint->points + pointIndex;

				// Fixed anchor points for applying impulses
				b3Vec3 rA = cp->rA;
				b3Vec3 rB = cp->rB;

				// compute current separation
				// this is subject to round-off error if the anchor is far from the body center of mass
				b3Vec3 ds = b3Add( dp, b3Sub( b3RotateVector( dqB, rB ), b3RotateVector( dqA, rA ) ) );
				float s = b3Dot( ds, normal ) + cp->baseSeparation;

				float velocityBias;
				float massScale;
				float impulseScale;
				if ( s > 0.0f )
				{
					// speculative bias is positive
					velocityBias = s * inv_h;
					massScale = 1.0f;
					impulseScale = 0.0f;
				}
				else
				{
					// overlap bias is negative
					velocityBias = b3MaxFloat( softness.massScale * softness.biasRate * s, -contactSpeed );
					massScale = softness.massScale;
					impulseScale = softness.impulseScale;
				}

				// relative normal velocity at contact
				b3Vec3 vrA = b3Add( vA, b3Cross( wA, rA ) );
				b3Vec3 vrB = b3Add( vB, b3Cross( wB, rB ) );
				float vn = b3Dot( b3Sub( vrB, vrA ), normal );

				// incremental normal impulse
				float deltaImpulse = -cp->normalMass * ( massScale * vn + velocityBias ) - impulseScale * cp->normalImpulse;

				// clamp the accumulated impulse
				float newImpulse = b3MaxFloat( cp->normalImpulse + deltaImpulse, 0.0f );
				deltaImpulse = newImpulse - cp->normalImpulse;
				cp->normalImpulse = newImpulse;

				// apply normal impulse
				b3Vec3 P = b3MulSV( deltaImpulse, normal );
				vA = b3MulSub( vA, mA, P );
				wA = b3Sub( wA, b3MulMV( iA, b3Cross( rA, P ) ) );

				vB = b3MulAdd( vB, mB, P );
				wB = b3Add( wB, b3MulMV( iB, b3Cross( rB, P ) ) );
			}
		}

		if ( stateA->flags & b3_dynamicFlag )
		{
			stateA->linearVelocity = vA;
			stateA->angularVelocity = wA;
		}

		if ( stateB->flags & b3_dynamicFlag )
		{
			stateB->linearVelocity = vB;
			stateB->angularVelocity = wB;
		}
	}
}

// Solve contacts: normal, friction and rolling resistance.
void b3SolveContacts_Mesh( b3SolverBlock block, b3StepContext* context )
{
	b3World* world = context->world;
	b3GraphColor* color = world->constraintGraph.colors + block.colorIndex;
	b3ContactConstraint* contactConstraints = color->contactConstraints;
	b3BodyState* states = context->states;

	// This is a dummy state to represent a static body because static bodies have no solver body.
	b3BodyState dummyState = b3_identityBodyState;

	// The last block might not be full
	int startIndex = block.startIndex;
	int endIndex = startIndex + block.count;

	float inv_h = context->inv_h;

	for ( int i = startIndex; i < endIndex; ++i )
	{
		b3ContactConstraint* contactConstraint = contactConstraints + i;
		int manifoldCount = contactConstraint->manifoldCount;

		int indexA = contactConstraint->indexA;
		int indexB = contactConstraint->indexB;

		float mA = contactConstraint->invMassA;
		b3Matrix3 iA = contactConstraint->invIA;
		float mB = contactConstraint->invMassB;
		b3Matrix3 iB = contactConstraint->invIB;

		b3BodyState* stateA = indexA == B3_NULL_INDEX ? &dummyState : states + indexA;
		b3Vec3 vA = stateA->linearVelocity;
		b3Vec3 wA = stateA->angularVelocity;
		b3Quat dqA = stateA->deltaRotation;

		b3BodyState* stateB = indexB == B3_NULL_INDEX ? &dummyState : states + indexB;
		b3Vec3 vB = stateB->linearVelocity;
		b3Vec3 wB = stateB->angularVelocity;
		b3Quat dqB = stateB->deltaRotation;

		b3Vec3 dp = b3Sub( stateB->deltaPosition, stateA->deltaPosition );
		float friction = contactConstraint->friction;
		float rollingResistance = contactConstraint->rollingResistance;

		for ( int j = 0; j < manifoldCount; ++j )
		{
			b3ManifoldConstraint* constraint = contactConstraint->constraints + j;

			int pointCount = constraint->pointCount;
			b3Vec3 normal = constraint->normal;

			float totalNormalImpulse = 0.0f;
			float totalTwistLimit = 0.0f;

			for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
			{
				b3ManifoldConstraintPoint* cp = constraint->points + pointIndex;

				// Fixed anchor points for applying impulses
				b3Vec3 rA = cp->rA;
				b3Vec3 rB = cp->rB;

				// compute current separation
				// this is subject to round-off error if the anchor is far from the body center of mass
				b3Vec3 ds = b3Add( dp, b3Sub( b3RotateVector( dqB, rB ), b3RotateVector( dqA, rA ) ) );
				float s = b3Dot( ds, normal ) + cp->baseSeparation;

				// speculative bias, zero when overlapped
				float velocityBias = s > 0.0f ? s * inv_h : 0.0f;

				// relative normal velocity at contact
				b3Vec3 vrA = b3Add( vA, b3Cross( wA, rA ) );
				b3Vec3 vrB = b3Add( vB, b3Cross( wB, rB ) );
				float vn = b3Dot( b3Sub( vrB, vrA ), normal );

				// incremental normal impulse
				float deltaImpulse = -cp->normalMass * ( vn + velocityBias );

				// clamp the accumulated impulse
				float newImpulse = b3MaxFloat( cp->normalImpulse + deltaImpulse, 0.0f );
				deltaImpulse = newImpulse - cp->normalImpulse;
				cp->normalImpulse = newImpulse;
				cp->totalNormalImpulse += newImpulse;
				totalNormalImpulse += newImpulse;
				totalTwistLimit += cp->leverArm * newImpulse;

				// apply normal impulse
				b3Vec3 P = b3MulSV( deltaImpulse, normal );
				vA = b3MulSub( vA, mA, P );
				wA = b3Sub( wA, b3MulMV( iA, b3Cross( rA, P ) ) );

				vB = b3MulAdd( vB, mB, P );
				wB = b3Add( wB, b3MulMV( iB, b3Cross( rB, P ) ) );
			}

			// Central twist friction
			{
				float twistSpeed = b3Dot( constraint->normal, b3Sub( wB, wA ) );
				float maxImpulse = friction * totalTwistLimit;
				float deltaImpulse = -constraint->twistMass * twistSpeed;
				float oldImpulse = constraint->twistImpulse;
				constraint->twistImpulse = b3ClampFloat( oldImpulse + deltaImpulse, -maxImpulse, maxImpulse );
				deltaImpulse = constraint->twistImpulse - oldImpulse;

				wA = b3Sub( wA, b3MulMV( iA, b3MulSV( deltaImpulse, constraint->normal ) ) );
				wB = b3Add( wB, b3MulMV( iB, b3MulSV( deltaImpulse, constraint->normal ) ) );
			}

			// Rolling resistance
			if ( rollingResistance > 0.0f )
			{
				b3Vec3 deltaImpulse = b3Neg( b3MulMV( contactConstraint->rollingMass, b3Sub( wB, wA ) ) );
				b3Vec3 oldImpulse = constraint->rollingImpulse;
				constraint->rollingImpulse = b3Add( oldImpulse, deltaImpulse );

				float maxImpulse = rollingResistance * totalNormalImpulse;
				float magSqr = b3Dot( constraint->rollingImpulse, constraint->rollingImpulse );
				if ( magSqr > maxImpulse * maxImpulse + FLT_EPSILON )
				{
					constraint->rollingImpulse = b3MulSV( maxImpulse / sqrtf( magSqr ), constraint->rollingImpulse );
				}

				deltaImpulse = b3Sub( constraint->rollingImpulse, oldImpulse );

				wA = b3Sub( wA, b3MulMV( iA, deltaImpulse ) );
				wB = b3Add( wB, b3MulMV( iB, deltaImpulse ) );
			}

			// Central friction
			{
				b3Vec3 tangent1 = constraint->tangent1;
				b3Vec3 tangent2 = constraint->tangent2;

				// Fixed anchor points for applying impulses
				b3Vec3 rA = constraint->centerA;
				b3Vec3 rB = constraint->centerB;

				// Relative tangent velocity at contact
				b3Vec3 vrA = b3Add( vA, b3Cross( wA, rA ) );
				b3Vec3 vrB = b3Add( vB, b3Cross( wB, rB ) );
				b3Vec3 vr = b3Sub( vrB, vrA );
				b3Vec2 vt = {
					b3Dot( vr, tangent1 ) - constraint->tangentVelocity1,
					b3Dot( vr, tangent2 ) - constraint->tangentVelocity2,
				};

				// Incremental tangent impulse
				b3Vec2 tm = b3MulMV2( constraint->tangentMass, vt );
				b3Vec2 deltaImpulse = { -tm.x, -tm.y };
				b3Vec2 newImpulse = {
					constraint->frictionImpulse.x + deltaImpulse.x,
					constraint->frictionImpulse.y + deltaImpulse.y,
				};

				float maxImpulse = friction * totalNormalImpulse;

				// Clamp the accumulated impulse
				float lengthSquared = b3Dot2( newImpulse, newImpulse );
				if ( lengthSquared > maxImpulse * maxImpulse )
				{
					float scale = maxImpulse / sqrtf( lengthSquared );
					newImpulse.x *= scale;
					newImpulse.y *= scale;
				}
				deltaImpulse = b3Sub2( newImpulse, constraint->frictionImpulse );
				constraint->frictionImpulse = newImpulse;

				// Apply delta impulse
				b3Vec3 P = b3Blend2( deltaImpulse.x, tangent1, deltaImpulse.y, tangent2 );
				vA = b3MulSub( vA, mA, P );
				wA = b3Sub( wA, b3MulMV( iA, b3Cross( rA, P ) ) );
				vB = b3MulAdd( vB, mB, P );
				wB = b3Add( wB, b3MulMV( iB, b3Cross( rB, P ) ) );
			}
		}

		if ( stateA->flags & b3_dynamicFlag )
		{
			stateA->linearVelocity = vA;
			stateA->angularVelocity = wA;
		}

		if ( stateB->flags & b3_dynamicFlag )
		{
			stateB->linearVelocity = vB;
			stateB->angularVelocity = wB;
		}
	}
}

void b3ApplyRestitution_Mesh( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( restitution_mesh, "Restitution Mesh", b3_colorViolet, true );

	b3World* world = context->world;
	b3GraphColor* color = world->constraintGraph.colors + block.colorIndex;
	b3ContactConstraint* contactConstraints = color->contactConstraints;
	b3BodyState* states = context->states;

	float threshold = world->restitutionThreshold;
	float inv_h = context->inv_h;
	bool propagate = world->enableRestitutionPropagation;

	b3BodyState dummyState = b3_identityBodyState;

	int startIndex = block.startIndex;
	int endIndex = startIndex + block.count;

	for ( int i = startIndex; i < endIndex; ++i )
	{
		b3ContactConstraint* contactConstraint = contactConstraints + i;
		float restitution = contactConstraint->restitution;
		if ( propagate == false && restitution == 0.0f )
		{
			continue;
		}

		int manifoldCount = contactConstraint->manifoldCount;

		int indexA = contactConstraint->indexA;
		int indexB = contactConstraint->indexB;

		float mA = contactConstraint->invMassA;
		b3Matrix3 iA = contactConstraint->invIA;
		float mB = contactConstraint->invMassB;
		b3Matrix3 iB = contactConstraint->invIB;

		b3BodyState* stateA = indexA == B3_NULL_INDEX ? &dummyState : states + indexA;
		b3Vec3 vA = stateA->linearVelocity;
		b3Vec3 wA = stateA->angularVelocity;
		b3Quat dqA = stateA->deltaRotation;

		b3BodyState* stateB = indexB == B3_NULL_INDEX ? &dummyState : states + indexB;
		b3Vec3 vB = stateB->linearVelocity;
		b3Vec3 wB = stateB->angularVelocity;
		b3Quat dqB = stateB->deltaRotation;

		b3Vec3 dp = b3Sub( stateB->deltaPosition, stateA->deltaPosition );

		for ( int j = 0; j < manifoldCount; ++j )
		{
			b3ManifoldConstraint* constraint = contactConstraint->constraints + j;

			int pointCount = constraint->pointCount;
			b3Vec3 normal = constraint->normal;

			for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
			{
				b3ManifoldConstraintPoint* cp = constraint->points + pointIndex;

				b3Vec3 rA = cp->rA;
				b3Vec3 rB = cp->rB;

				// The total normal impulse is 0 for speculative points.
				float compressionImpulse = cp->totalNormalImpulse - cp->restitutionImpulse;
				bool armed = restitution > 0.0f && cp->relativeVelocity < -threshold && compressionImpulse > 0.0f;

				float velocityBias;
				if ( armed )
				{
					velocityBias = restitution * cp->relativeVelocity;
				}
				else
				{
					b3Vec3 ds = b3Add( dp, b3Sub( b3RotateVector( dqB, rB ), b3RotateVector( dqA, rA ) ) );
					float s = b3Dot( ds, normal ) + cp->baseSeparation;

					velocityBias = s > 0.0f ? s * inv_h : 0.0f;
				}

				b3Vec3 vrA = b3Add( vA, b3Cross( wA, rA ) );
				b3Vec3 vrB = b3Add( vB, b3Cross( wB, rB ) );
				float vn = b3Dot( b3Sub( vrB, vrA ), normal );

				float impulse = -cp->normalMass * ( vn + velocityBias );

				float newImpulse = b3MaxFloat( cp->normalImpulse + impulse, 0.0f );
				impulse = newImpulse - cp->normalImpulse;

				float approachImpulse = b3MinFloat( b3MaxFloat( -cp->normalMass * vn, 0.0f ), b3MaxFloat( impulse, 0.0f ) );

				if ( armed )
				{
					// Poisson kinetic restitution guarantees no energy gain.
					float allowance = restitution * ( compressionImpulse + approachImpulse ) - cp->restitutionImpulse;
					impulse = b3MinFloat( impulse, approachImpulse + b3MaxFloat( allowance, 0.0f ) );
				}

				cp->normalImpulse += impulse;
				cp->restitutionImpulse += impulse - approachImpulse;
				cp->totalNormalImpulse += impulse;

				b3Vec3 P = b3MulSV( impulse, normal );
				vA = b3MulSub( vA, mA, P );
				wA = b3Sub( wA, b3MulMV( iA, b3Cross( rA, P ) ) );

				vB = b3MulAdd( vB, mB, P );
				wB = b3Add( wB, b3MulMV( iB, b3Cross( rB, P ) ) );
			}
		}

		if ( stateA->flags & b3_dynamicFlag )
		{
			stateA->linearVelocity = vA;
			stateA->angularVelocity = wA;
		}

		if ( stateB->flags & b3_dynamicFlag )
		{
			stateB->linearVelocity = vB;
			stateB->angularVelocity = wB;
		}
	}

	b3TracyCZoneEnd( restitution_mesh );
}

// Don't need to use spans for colors for this because the constraint to contact association
// is already linked by pointer.
void b3StoreImpulses_Mesh( b3SolverBlock block, b3StepContext* context, int workerIndex )
{
	b3World* world = context->world;

	// Mirror b3PrepareContacts_Mesh: the per-color flat arrays and the overflow color
	// each have their own (base, spans, manifoldBase).
	b3ContactPrepareSpan* spans = context->contactPrepareSpans;
	b3ContactConstraint* base = context->contactConstraints;

	if ( block.blockType == b3_overflowBlock )
	{
		b3GraphColor* overflow = world->constraintGraph.colors + B3_OVERFLOW_INDEX;
		spans = context->overflowSpans;
		base = overflow->contactConstraints;
	}

	b3TaskContext* taskContext = world->taskContexts.data + workerIndex;
	b3BitSet* hitEventBitSet = &taskContext->hitEventBitSet;
	bool hasHitEvents = taskContext->hasHitEvents;
	float negHitThreshold = -world->hitEventThreshold;

	int index = block.startIndex;
	int endIndex = block.startIndex + block.count;

	// Find color for start index. Linear search but fast.
	int colorIndex = 0;
	while ( spans[colorIndex + 1].start <= index )
	{
		colorIndex += 1;
	}

	// Loop over block
	while ( index < endIndex )
	{
		int colorStart = spans[colorIndex].start;
		int colorEndIndex = b3MinInt( spans[colorIndex + 1].start, endIndex );

		// Loop over color
		for ( ; index < colorEndIndex; ++index )
		{
			b3ContactConstraint* contactConstraint = base + index;

			int localIndex = index - colorStart;
			B3_UNUSED( localIndex );
			B3_ASSERT( 0 <= localIndex && localIndex < spans[colorIndex].count );

			// Having this contact pointer simplifies impulse storage
			b3Contact* contact = contactConstraint->contact;
			B3_ASSERT( contact != NULL );

			// Catches the wrong-(base, spans) pairing: the contact pointer stashed by
			// b3PrepareContacts_Mesh at this flat slot must reference the same contact
			// the span at this slot describes.
			B3_VALIDATE( contact->contactId == spans[colorIndex].contacts[localIndex].contactId );

			int manifoldCount = contactConstraint->manifoldCount;
			B3_ASSERT( manifoldCount == contact->manifoldCount );

			bool checkHitEvents = ( contact->flags & b3_simEnableHitEvent ) != 0;
			bool flagged = false;

			for ( int manifoldIndex = 0; manifoldIndex < manifoldCount; ++manifoldIndex )
			{
				b3Manifold* manifold = contact->manifolds + manifoldIndex;
				b3ManifoldConstraint* constraint = contactConstraint->constraints + manifoldIndex;
				manifold->twistImpulse = constraint->twistImpulse;
				manifold->frictionImpulse = b3Blend2( constraint->frictionImpulse.x, constraint->tangent1,
													  constraint->frictionImpulse.y, constraint->tangent2 );
				manifold->rollingImpulse = constraint->rollingImpulse;

				int count = constraint->pointCount;
				B3_ASSERT( count == manifold->pointCount );
				for ( int pointIndex = 0; pointIndex < count; ++pointIndex )
				{
					b3ManifoldConstraintPoint* cp = constraint->points + pointIndex;
					b3ManifoldPoint* mp = manifold->points + pointIndex;
					mp->normalImpulse = cp->normalImpulse;
					mp->totalNormalImpulse = cp->totalNormalImpulse;

					if ( checkHitEvents && flagged == false && mp->normalVelocity < negHitThreshold &&
						 mp->totalNormalImpulse > 0.0f )
					{
						b3SetBit( hitEventBitSet, contact->contactId );
						hasHitEvents = true;
						flagged = true;
					}
				}
			}
		}

		// Advance to next color
		colorIndex += 1;
	}

	taskContext->hasHitEvents = hasHitEvents;
}

#define B3_SIMD_WIDTH 4

// Wide vec2
typedef struct b3Vec2W4
{
	b3FloatW4 x, y;
} b3Vec2W4;

// Wide vec3
typedef struct b3Vec3W4
{
	b3FloatW4 X, Y, Z;
} b3Vec3W4;

// Wide quaternion
typedef struct b3QuatW4
{
	b3Vec3W4 V;
	b3FloatW4 S;
} b3QuatW4;

// Wide symmetric matrix2
typedef struct b3SymMatrix2W4
{
	b3FloatW4 cxx, cxy, cyy;
} b3SymMatrix2W4;

// Wide symmetric matrix3
typedef struct b3SymMatrix3W4
{
	b3FloatW4 cxx, cxy, cxz, cyy, cyz, czz;
} b3SymMatrix3W4;

typedef struct b3Matrix3W4
{
	b3Vec3W4 cx, cy, cz;
} b3Matrix3W4;

// s * a
static inline b3Vec3W4 b3MulSVW4( b3FloatW4 s, b3Vec3W4 a )
{
	return (b3Vec3W4){ b3MulW4( s, a.X ), b3MulW4( s, a.Y ), b3MulW4( s, a.Z ) };
}

// a - s * b
static inline b3Vec3W4 b3MulSubSVW4( b3Vec3W4 a, b3FloatW4 s, b3Vec3W4 b )
{
	return (b3Vec3W4){ b3SubW4( a.X, b3MulW4( s, b.X ) ), b3SubW4( a.Y, b3MulW4( s, b.Y ) ), b3SubW4( a.Z, b3MulW4( s, b.Z ) ) };
}

// a + s * b
static inline b3Vec3W4 b3MulAddSVW4( b3Vec3W4 a, b3FloatW4 s, b3Vec3W4 b )
{
	return (b3Vec3W4){ b3AddW4( a.X, b3MulW4( s, b.X ) ), b3AddW4( a.Y, b3MulW4( s, b.Y ) ), b3AddW4( a.Z, b3MulW4( s, b.Z ) ) };
}

// a + b
static inline b3Vec2W4 b3AddV2W4( b3Vec2W4 a, b3Vec2W4 b )
{
	return (b3Vec2W4){
		b3AddW4( a.x, b.x ),
		b3AddW4( a.y, b.y ),
	};
}

// a - b
static inline b3Vec3W4 b3SubVW4( b3Vec3W4 a, b3Vec3W4 b )
{
	return (b3Vec3W4){
		b3SubW4( a.X, b.X ),
		b3SubW4( a.Y, b.Y ),
		b3SubW4( a.Z, b.Z ),
	};
}

// a + b
static inline b3Vec3W4 b3AddVW4( b3Vec3W4 a, b3Vec3W4 b )
{
	return (b3Vec3W4){
		b3AddW4( a.X, b.X ),
		b3AddW4( a.Y, b.Y ),
		b3AddW4( a.Z, b.Z ),
	};
}

// m * a
static inline b3Vec2W4 b3MulMV2W4( b3SymMatrix2W4 m, b3Vec2W4 a )
{
	b3Vec2W4 b = {
		b3AddW4( b3MulW4( m.cxx, a.x ), b3MulW4( m.cxy, a.y ) ),
		b3AddW4( b3MulW4( m.cxy, a.x ), b3MulW4( m.cyy, a.y ) ),
	};

	return b;
}

// m * a
static inline b3Vec3W4 b3MulMVW4( b3SymMatrix3W4 m, b3Vec3W4 a )
{
	b3Vec3W4 b = {
		b3AddW4( b3MulW4( m.cxx, a.X ), b3AddW4( b3MulW4( m.cxy, a.Y ), b3MulW4( m.cxz, a.Z ) ) ),
		b3AddW4( b3MulW4( m.cxy, a.X ), b3AddW4( b3MulW4( m.cyy, a.Y ), b3MulW4( m.cyz, a.Z ) ) ),
		b3AddW4( b3MulW4( m.cxz, a.X ), b3AddW4( b3MulW4( m.cyz, a.Y ), b3MulW4( m.czz, a.Z ) ) ),
	};

	return b;
}

// a - m * b
static inline b3Vec3W4 b3MulSubMVW4( b3Vec3W4 a, b3SymMatrix3W4 m, b3Vec3W4 b )
{
	b3Vec3W4 c = {
		b3AddW4( b3MulW4( m.cxx, b.X ), b3AddW4( b3MulW4( m.cxy, b.Y ), b3MulW4( m.cxz, b.Z ) ) ),
		b3AddW4( b3MulW4( m.cxy, b.X ), b3AddW4( b3MulW4( m.cyy, b.Y ), b3MulW4( m.cyz, b.Z ) ) ),
		b3AddW4( b3MulW4( m.cxz, b.X ), b3AddW4( b3MulW4( m.cyz, b.Y ), b3MulW4( m.czz, b.Z ) ) ),
	};

	return (b3Vec3W4){ b3SubW4( a.X, c.X ), b3SubW4( a.Y, c.Y ), b3SubW4( a.Z, c.Z ) };
}

// a + m * b
static inline b3Vec3W4 b3MulAddMVW4( b3Vec3W4 a, b3SymMatrix3W4 m, b3Vec3W4 b )
{
	b3Vec3W4 c = {
		b3AddW4( b3MulW4( m.cxx, b.X ), b3AddW4( b3MulW4( m.cxy, b.Y ), b3MulW4( m.cxz, b.Z ) ) ),
		b3AddW4( b3MulW4( m.cxy, b.X ), b3AddW4( b3MulW4( m.cyy, b.Y ), b3MulW4( m.cyz, b.Z ) ) ),
		b3AddW4( b3MulW4( m.cxz, b.X ), b3AddW4( b3MulW4( m.cyz, b.Y ), b3MulW4( m.czz, b.Z ) ) ),
	};

	return (b3Vec3W4){ b3AddW4( a.X, c.X ), b3AddW4( a.Y, c.Y ), b3AddW4( a.Z, c.Z ) };
}

static inline b3FloatW4 b3DotW4( b3Vec3W4 a, b3Vec3W4 b )
{
	return b3AddW4( b3AddW4( b3MulW4( a.X, b.X ), b3MulW4( a.Y, b.Y ) ), b3MulW4( a.Z, b.Z ) );
}

static inline b3Vec3W4 b3CrossW4( b3Vec3W4 a, b3Vec3W4 b )
{
	b3Vec3W4 c;
	c.X = b3SubW4( b3MulW4( a.Y, b.Z ), b3MulW4( a.Z, b.Y ) );
	c.Y = b3SubW4( b3MulW4( a.Z, b.X ), b3MulW4( a.X, b.Z ) );
	c.Z = b3SubW4( b3MulW4( a.X, b.Y ), b3MulW4( a.Y, b.X ) );
	return c;
}

static inline b3Matrix3W4 b3MakeMatrixFromQuatW4( b3QuatW4 q )
{
	b3FloatW4 x2 = b3AddW4( q.V.X, q.V.X );
	b3FloatW4 y2 = b3AddW4( q.V.Y, q.V.Y );
	b3FloatW4 z2 = b3AddW4( q.V.Z, q.V.Z );
	b3FloatW4 xx2 = b3MulW4( q.V.X, x2 );
	b3FloatW4 yy2 = b3MulW4( q.V.Y, y2 );
	b3FloatW4 zz2 = b3MulW4( q.V.Z, z2 );
	b3FloatW4 xy2 = b3MulW4( q.V.X, y2 );
	b3FloatW4 xz2 = b3MulW4( q.V.X, z2 );
	b3FloatW4 yz2 = b3MulW4( q.V.Y, z2 );
	b3FloatW4 xw2 = b3MulW4( q.S, x2 );
	b3FloatW4 yw2 = b3MulW4( q.S, y2 );
	b3FloatW4 zw2 = b3MulW4( q.S, z2 );
	b3FloatW4 one = b3SplatW4( 1.0f );

	b3Matrix3W4 m;
	m.cx.X = b3SubW4( one, b3AddW4( yy2, zz2 ) );
	m.cx.Y = b3AddW4( xy2, zw2 );
	m.cx.Z = b3SubW4( xz2, yw2 );
	m.cy.X = b3SubW4( xy2, zw2 );
	m.cy.Y = b3SubW4( one, b3AddW4( xx2, zz2 ) );
	m.cy.Z = b3AddW4( yz2, xw2 );
	m.cz.X = b3AddW4( xz2, yw2 );
	m.cz.Y = b3SubW4( yz2, xw2 );
	m.cz.Z = b3SubW4( one, b3AddW4( xx2, yy2 ) );
	return m;
}

static inline b3Vec3W4 b3MulM3VW4( b3Matrix3W4 m, b3Vec3W4 a )
{
	b3Vec3W4 b = {
		b3AddW4( b3MulW4( m.cx.X, a.X ), b3AddW4( b3MulW4( m.cy.X, a.Y ), b3MulW4( m.cz.X, a.Z ) ) ),
		b3AddW4( b3MulW4( m.cx.Y, a.X ), b3AddW4( b3MulW4( m.cy.Y, a.Y ), b3MulW4( m.cz.Y, a.Z ) ) ),
		b3AddW4( b3MulW4( m.cx.Z, a.X ), b3AddW4( b3MulW4( m.cy.Z, a.Y ), b3MulW4( m.cz.Z, a.Z ) ) ),
	};

	return b;
}

static inline b3Vec3W4 b3InvRotateVectorW4( b3QuatW4 q, b3Vec3W4 a )
{
	b3Vec3W4 t = b3CrossW4( q.V, a );
	t = (b3Vec3W4){ b3AddW4( t.X, t.X ), b3AddW4( t.Y, t.Y ), b3AddW4( t.Z, t.Z ) };
	b3Vec3W4 u = b3CrossW4( q.V, t );
	return (b3Vec3W4){
		b3AddW4( b3SubW4( a.X, b3MulW4( q.S, t.X ) ), u.X ),
		b3AddW4( b3SubW4( a.Y, b3MulW4( q.S, t.Y ) ), u.Y ),
		b3AddW4( b3SubW4( a.Z, b3MulW4( q.S, t.Z ) ), u.Z ),
	};
}

// Soft contact constraints with sub-stepping support
// Uses fixed anchors for Jacobians for better behavior on rolling shapes (circles & capsules)
// http://mmacklin.com/smallsteps.pdf
// https://box2d.org/files/ErinCatto_SoftConstraints_GDC2011.pdf

typedef struct b3ContactConstraintPointW4
{
	b3Vec3W4 anchorAs, anchorBs;
	b3FloatW4 baseSeparations;
	b3FloatW4 normalImpulses;
	b3FloatW4 totalNormalImpulses;
	b3FloatW4 normalMasses;
	b3FloatW4 leverArms;
	b3FloatW4 relativeVelocities;
	b3FloatW4 restitutionImpulses;
} b3ContactConstraintPointW4;

// Solves four points
typedef struct b3ContactConstraintW4
{
	// These are base 1
	int indexA[B3_SIMD_WIDTH];
	int indexB[B3_SIMD_WIDTH];

	int pointCounts[B3_SIMD_WIDTH];

	b3FloatW4 invMassA, invMassB;
	b3SymMatrix3W4 invIA, invIB;
	b3Vec3W4 normal;

	// todo test computing the tangents on the fly, at least tangent2
	b3Vec3W4 tangent1;
	b3Vec3W4 tangent2;

	// Friction centers
	b3Vec3W4 centerA, centerB;
	b3FloatW4 twistMass;
	b3FloatW4 twistImpulse;
	b3SymMatrix2W4 tangentMass;
	b3Vec2W4 frictionImpulse;
	b3SymMatrix3W4 rollingMass;
	b3Vec3W4 rollingImpulse;
	b3FloatW4 friction;
	b3FloatW4 rollingResistance;
	b3FloatW4 tangentVelocity1;
	b3FloatW4 tangentVelocity2;
	b3FloatW4 restitution;

	b3Manifold* manifolds[B3_SIMD_WIDTH];

	// todo store the maximum point count per wide constraint
	// to make this work I need zero initialization which is too
	// expensive for all the wide constraint data. Instead
	// the graph color should store the point count as a compact secondary
	// transient array with zero initialization.
	b3ContactConstraintPointW4 points[B3_MAX_MANIFOLD_POINTS];

} b3ContactConstraintW4;

int b3GetWideContactConstraintByteCountW4( void )
{
	return sizeof( b3ContactConstraintW4 );
}

// wide version of b3BodyState
typedef struct b3BodyStateW4
{
	b3Vec3W4 v;
	b3Vec3W4 w;
	b3Vec3W4 dp;
	b3QuatW4 dq;
} b3BodyStateW4;

#if defined( B3_SIMD_ENABLED )

_Static_assert( sizeof( b3BodyState ) == 64, "body state layout" );
_Static_assert( offsetof( b3BodyState, linearVelocity ) == 0 && offsetof( b3BodyState, angularVelocity ) == 16 &&
					offsetof( b3BodyState, deltaPosition ) == 32 && offsetof( b3BodyState, deltaRotation ) == 48,
				"body state layout" );

B3_FORCE_INLINE b3BodyStateW4 b3GatherBodiesW4( const b3BodyState* B3_RESTRICT states, const int* B3_RESTRICT indices )
{
	const float* identity = (const float*)&b3_identityBodyState;

	// Indices are 0 for null
	const float* p1 = indices[0] == 0 ? identity : (const float*)( states + indices[0] - 1 );
	const float* p2 = indices[1] == 0 ? identity : (const float*)( states + indices[1] - 1 );
	const float* p3 = indices[2] == 0 ? identity : (const float*)( states + indices[2] - 1 );
	const float* p4 = indices[3] == 0 ? identity : (const float*)( states + indices[3] - 1 );

	b3BodyStateW4 s;
	b3FloatW4 pad;
	b3TransposeW4( b3LoadW4( p1 ), b3LoadW4( p2 ), b3LoadW4( p3 ), b3LoadW4( p4 ), &s.v.X, &s.v.Y, &s.v.Z, &pad );
	b3TransposeW4( b3LoadW4( p1 + 4 ), b3LoadW4( p2 + 4 ), b3LoadW4( p3 + 4 ), b3LoadW4( p4 + 4 ), &s.w.X, &s.w.Y, &s.w.Z, &pad );
	b3TransposeW4( b3LoadW4( p1 + 8 ), b3LoadW4( p2 + 8 ), b3LoadW4( p3 + 8 ), b3LoadW4( p4 + 8 ), &s.dp.X, &s.dp.Y, &s.dp.Z,
				   &pad );
	b3TransposeW4( b3LoadW4( p1 + 12 ), b3LoadW4( p2 + 12 ), b3LoadW4( p3 + 12 ), b3LoadW4( p4 + 12 ), &s.dq.V.X, &s.dq.V.Y,
				   &s.dq.V.Z, &s.dq.S );
	return s;
}

B3_FORCE_INLINE void b3StoreBodyVelocityW4( b3BodyState* B3_RESTRICT states, int index, b3FloatW4 v, b3FloatW4 w )
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

	b3StoreW4( (float*)state, v );
	b3StoreW4( (float*)state + 4, w );
}

// This writes only the velocities back to the solver bodies
B3_FORCE_INLINE void b3ScatterBodiesW4( b3BodyState* B3_RESTRICT states, const int* B3_RESTRICT indices,
									   const b3BodyStateW4* B3_RESTRICT simdBody )
{
	// I don't use any dummy body in the body array because this will lead to multithreaded sharing and the
	// associated cache flushing.
	b3FloatW4 zero = b3ZeroW4();
	b3FloatW4 v1, v2, v3, v4;
	b3TransposeW4( simdBody->v.X, simdBody->v.Y, simdBody->v.Z, zero, &v1, &v2, &v3, &v4 );
	b3FloatW4 w1, w2, w3, w4;
	b3TransposeW4( simdBody->w.X, simdBody->w.Y, simdBody->w.Z, zero, &w1, &w2, &w3, &w4 );

	b3StoreBodyVelocityW4( states, indices[0], v1, w1 );
	b3StoreBodyVelocityW4( states, indices[1], v2, w2 );
	b3StoreBodyVelocityW4( states, indices[2], v3, w3 );
	b3StoreBodyVelocityW4( states, indices[3], v4, w4 );
}

#else // non-simd

B3_FORCE_INLINE b3BodyStateW4 b3GatherBodiesW4( const b3BodyState* B3_RESTRICT states, const int* B3_RESTRICT indices )
{
	b3BodyState identity = b3_identityBodyState;

	b3BodyState s1 = indices[0] == 0 ? identity : states[indices[0] - 1];
	b3BodyState s2 = indices[1] == 0 ? identity : states[indices[1] - 1];
	b3BodyState s3 = indices[2] == 0 ? identity : states[indices[2] - 1];
	b3BodyState s4 = indices[3] == 0 ? identity : states[indices[3] - 1];

	b3BodyStateW4 simdBody;
	simdBody.v.X = (b3FloatW){ s1.linearVelocity.x, s2.linearVelocity.x, s3.linearVelocity.x, s4.linearVelocity.x };
	simdBody.v.Y = (b3FloatW){ s1.linearVelocity.y, s2.linearVelocity.y, s3.linearVelocity.y, s4.linearVelocity.y };
	simdBody.v.Z = (b3FloatW){ s1.linearVelocity.z, s2.linearVelocity.z, s3.linearVelocity.z, s4.linearVelocity.z };
	simdBody.w.X = (b3FloatW){ s1.angularVelocity.x, s2.angularVelocity.x, s3.angularVelocity.x, s4.angularVelocity.x };
	simdBody.w.Y = (b3FloatW){ s1.angularVelocity.y, s2.angularVelocity.y, s3.angularVelocity.y, s4.angularVelocity.y };
	simdBody.w.Z = (b3FloatW){ s1.angularVelocity.z, s2.angularVelocity.z, s3.angularVelocity.z, s4.angularVelocity.z };
	simdBody.dp.X = (b3FloatW){ s1.deltaPosition.x, s2.deltaPosition.x, s3.deltaPosition.x, s4.deltaPosition.x };
	simdBody.dp.Y = (b3FloatW){ s1.deltaPosition.y, s2.deltaPosition.y, s3.deltaPosition.y, s4.deltaPosition.y };
	simdBody.dp.Z = (b3FloatW){ s1.deltaPosition.z, s2.deltaPosition.z, s3.deltaPosition.z, s4.deltaPosition.z };
	simdBody.dq.V.X = (b3FloatW){ s1.deltaRotation.v.x, s2.deltaRotation.v.x, s3.deltaRotation.v.x, s4.deltaRotation.v.x };
	simdBody.dq.V.Y = (b3FloatW){ s1.deltaRotation.v.y, s2.deltaRotation.v.y, s3.deltaRotation.v.y, s4.deltaRotation.v.y };
	simdBody.dq.V.Z = (b3FloatW){ s1.deltaRotation.v.z, s2.deltaRotation.v.z, s3.deltaRotation.v.z, s4.deltaRotation.v.z };
	simdBody.dq.S = (b3FloatW){ s1.deltaRotation.s, s2.deltaRotation.s, s3.deltaRotation.s, s4.deltaRotation.s };

	return simdBody;
}

// This writes only the velocities back to the solver bodies
B3_FORCE_INLINE void b3ScatterBodiesW4( b3BodyState* B3_RESTRICT states, const int* B3_RESTRICT indices,
									  const b3BodyStateW4* B3_RESTRICT simdBody )
{
	int index1 = indices[0] - 1;
	if ( index1 != -1 && ( states[index1].flags & b3_dynamicFlag ) != 0 )
	{
		b3BodyState* state = states + index1;
		state->linearVelocity.x = simdBody->v.X.x;
		state->linearVelocity.y = simdBody->v.Y.x;
		state->linearVelocity.z = simdBody->v.Z.x;
		state->angularVelocity.x = simdBody->w.X.x;
		state->angularVelocity.y = simdBody->w.Y.x;
		state->angularVelocity.z = simdBody->w.Z.x;
	}

	int index2 = indices[1] - 1;
	if ( index2 != -1 && ( states[index2].flags & b3_dynamicFlag ) != 0 )
	{
		b3BodyState* state = states + index2;
		state->linearVelocity.x = simdBody->v.X.y;
		state->linearVelocity.y = simdBody->v.Y.y;
		state->linearVelocity.z = simdBody->v.Z.y;
		state->angularVelocity.x = simdBody->w.X.y;
		state->angularVelocity.y = simdBody->w.Y.y;
		state->angularVelocity.z = simdBody->w.Z.y;
	}

	int index3 = indices[2] - 1;
	if ( index3 != -1 && ( states[index3].flags & b3_dynamicFlag ) != 0 )
	{
		b3BodyState* state = states + index3;
		state->linearVelocity.x = simdBody->v.X.z;
		state->linearVelocity.y = simdBody->v.Y.z;
		state->linearVelocity.z = simdBody->v.Z.z;
		state->angularVelocity.x = simdBody->w.X.z;
		state->angularVelocity.y = simdBody->w.Y.z;
		state->angularVelocity.z = simdBody->w.Z.z;
	}

	int index4 = indices[3] - 1;
	if ( index4 != -1 && ( states[index4].flags & b3_dynamicFlag ) != 0 )
	{
		b3BodyState* state = states + index4;
		state->linearVelocity.x = simdBody->v.X.w;
		state->linearVelocity.y = simdBody->v.Y.w;
		state->linearVelocity.z = simdBody->v.Z.w;
		state->angularVelocity.x = simdBody->w.X.w;
		state->angularVelocity.y = simdBody->w.Y.w;
		state->angularVelocity.z = simdBody->w.Z.w;
	}
}
#endif

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

#define B3_GATHER_LANES_W4( wide, lanes, field ) wide = b3SetW4( lanes[0]->field, lanes[1]->field, lanes[2]->field, lanes[3]->field )

static inline b3SymMatrix3W4 b3GatherInvInertiaW4( const b3BodySim* const* simLanes )
{
	const float* i0 = &simLanes[0]->invInertiaWorld.cx.x;
	const float* i1 = &simLanes[1]->invInertiaWorld.cx.x;
	const float* i2 = &simLanes[2]->invInertiaWorld.cx.x;
	const float* i3 = &simLanes[3]->invInertiaWorld.cx.x;

	b3SymMatrix3W4 m;
	b3FloatW4 unused;
	b3TransposeW4( b3LoadW4( i0 ), b3LoadW4( i1 ), b3LoadW4( i2 ), b3LoadW4( i3 ), &m.cxx, &m.cxy, &m.cxz, &unused );
	b3TransposeW4( b3LoadW4( i0 + 4 ), b3LoadW4( i1 + 4 ), b3LoadW4( i2 + 4 ), b3LoadW4( i3 + 4 ), &m.cyy, &m.cyz, &unused,
				   &unused );
	m.czz = b3SetW4( i0[8], i1[8], i2[8], i3[8] );
	return m;
}

static inline b3SymMatrix3W4 b3AddSymW4( b3SymMatrix3W4 a, b3SymMatrix3W4 b )
{
	return (b3SymMatrix3W4){
		b3AddW4( a.cxx, b.cxx ), b3AddW4( a.cxy, b.cxy ), b3AddW4( a.cxz, b.cxz ),
		b3AddW4( a.cyy, b.cyy ), b3AddW4( a.cyz, b.cyz ), b3AddW4( a.czz, b.czz ),
	};
}

static inline b3SymMatrix3W4 b3InvertSymW4( b3SymMatrix3W4 m )
{
	b3FloatW4 cxx = b3SubW4( b3MulW4( m.cyy, m.czz ), b3MulW4( m.cyz, m.cyz ) );
	b3FloatW4 cxy = b3SubW4( b3MulW4( m.cxz, m.cyz ), b3MulW4( m.cxy, m.czz ) );
	b3FloatW4 cxz = b3SubW4( b3MulW4( m.cxy, m.cyz ), b3MulW4( m.cxz, m.cyy ) );
	b3FloatW4 cyy = b3SubW4( b3MulW4( m.cxx, m.czz ), b3MulW4( m.cxz, m.cxz ) );
	b3FloatW4 cyz = b3SubW4( b3MulW4( m.cxy, m.cxz ), b3MulW4( m.cxx, m.cyz ) );
	b3FloatW4 czz = b3SubW4( b3MulW4( m.cxx, m.cyy ), b3MulW4( m.cxy, m.cxy ) );

	b3FloatW4 det = b3AddW4( b3MulW4( m.cxx, cxx ), b3AddW4( b3MulW4( m.cxy, cxy ), b3MulW4( m.cxz, cxz ) ) );
	b3FloatW4 valid = b3GreaterThanW4( b3AbsW4( det ), b3SplatW4( 1000.0f * FLT_MIN ) );
	b3FloatW4 invDet = b3BlendW4( b3ZeroW4(), b3DivW4( b3SplatW4( 1.0f ), det ), valid );

	return (b3SymMatrix3W4){
		b3MulW4( invDet, cxx ), b3MulW4( invDet, cxy ), b3MulW4( invDet, cxz ),
		b3MulW4( invDet, cyy ), b3MulW4( invDet, cyz ), b3MulW4( invDet, czz ),
	};
}

static inline b3Vec3W4 b3PerpW4( b3Vec3W4 a )
{
	b3FloatW4 zero = b3ZeroW4();
	b3FloatW4 half = b3SplatW4( 0.5f );
	b3FloatW4 mask = b3OrW4( b3LessThanW4( a.X, b3NegW4( half ) ), b3GreaterThanW4( a.X, half ) );

	b3Vec3W4 p;
	p.X = b3BlendW4( zero, a.Y, mask );
	p.Y = b3BlendW4( a.Z, b3NegW4( a.X ), mask );
	p.Z = b3BlendW4( b3NegW4( a.Y ), zero, mask );

	b3FloatW4 lengthSquared = b3DotW4( p, p );
	b3FloatW4 valid = b3GreaterThanW4( lengthSquared, b3SplatW4( 1000.0f * FLT_MIN ) );
	b3FloatW4 s = b3BlendW4( zero, b3DivW4( b3SplatW4( 1.0f ), b3SqrtW4( lengthSquared ) ), valid );
	return b3MulSVW4( s, p );
}

void b3PrepareContacts_ConvexW4( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( prepare_contact, "Prepare Contact", b3_colorYellow, true );
	b3World* world = context->world;
	b3BodySim* sims = context->sims;
	b3BodyState* states = context->states;
#if B3_ENABLE_VALIDATION
	b3Body* bodies = world->bodies.data;
#endif
	b3WidePrepareSpan* spans = context->widePrepareSpans;
	b3ContactConstraintW4* wideBase = context->wideConstraints;

	b3FloatW4 zeroW = b3ZeroW4();
	b3FloatW4 oneW = b3SplatW4( 1.0f );
	b3FloatW4 twoW = b3SplatW4( 2.0f );
	b3FloatW4 warmStartScale = world->enableWarmStarting ? oneW : zeroW;
	b3FloatW4 invTau = b3SplatW4( 1.0f / B3_SPECULATIVE_DISTANCE );
	b3FloatW4 minFrictionWeight = b3SplatW4( B3_MIN_FRICTION_WEIGHT );
	b3FloatW4 minDet = b3SplatW4( 1000.0f * FLT_MIN );
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
			b3ContactConstraintW4* c = wideBase + wideIndex;
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

			b3FloatW4 mA, mB;
			B3_GATHER_LANES_W4( mA, simLanesA, invMass );
			B3_GATHER_LANES_W4( mB, simLanesB, invMass );
			b3SymMatrix3W4 iA = b3GatherInvInertiaW4( simLanesA );
			b3SymMatrix3W4 iB = b3GatherInvInertiaW4( simLanesB );
			c->invMassA = mA;
			c->invMassB = mB;
			c->invIA = iA;
			c->invIB = iB;

			b3Vec3W4 tangentVelocity;
			B3_GATHER_LANES_W4( c->friction, contactLanes, friction );
			B3_GATHER_LANES_W4( c->rollingResistance, contactLanes, rollingResistance );
			B3_GATHER_LANES_W4( c->restitution, contactLanes, restitution );
			B3_GATHER_LANES_W4( tangentVelocity.X, contactLanes, tangentVelocity.x );
			B3_GATHER_LANES_W4( tangentVelocity.Y, contactLanes, tangentVelocity.y );
			B3_GATHER_LANES_W4( tangentVelocity.Z, contactLanes, tangentVelocity.z );

			b3Vec3W4 normal, frictionImpulse, rollingImpulse;
			b3FloatW4 twistImpulse;
			{
				const float* m0 = &manifoldLanes[0]->normal.x;
				const float* m1 = &manifoldLanes[1]->normal.x;
				const float* m2 = &manifoldLanes[2]->normal.x;
				const float* m3 = &manifoldLanes[3]->normal.x;
				b3TransposeW4( b3LoadW4( m0 ), b3LoadW4( m1 ), b3LoadW4( m2 ), b3LoadW4( m3 ), &normal.X, &normal.Y, &normal.Z,
							   &twistImpulse );
				b3TransposeW4( b3LoadW4( m0 + 4 ), b3LoadW4( m1 + 4 ), b3LoadW4( m2 + 4 ), b3LoadW4( m3 + 4 ), &frictionImpulse.X,
							   &frictionImpulse.Y, &frictionImpulse.Z, &rollingImpulse.X );
				rollingImpulse.Y = b3SetW4( m0[8], m1[8], m2[8], m3[8] );
				rollingImpulse.Z = b3SetW4( m0[9], m1[9], m2[9], m3[9] );
			}

			b3Vec3W4 tangent1 = b3PerpW4( normal );
			b3Vec3W4 tangent2 = b3CrossW4( tangent1, normal );
			c->normal = normal;
			c->tangent1 = tangent1;
			c->tangent2 = tangent2;
			c->tangentVelocity1 = b3DotW4( tangentVelocity, tangent1 );
			c->tangentVelocity2 = b3DotW4( tangentVelocity, tangent2 );

			b3FloatW4 rollingMask = b3GreaterThanW4( c->rollingResistance, zeroW );
			c->twistImpulse = b3MulW4( warmStartScale, twistImpulse );
			c->rollingImpulse.X = b3BlendW4( zeroW, b3MulW4( warmStartScale, rollingImpulse.X ), rollingMask );
			c->rollingImpulse.Y = b3BlendW4( zeroW, b3MulW4( warmStartScale, rollingImpulse.Y ), rollingMask );
			c->rollingImpulse.Z = b3BlendW4( zeroW, b3MulW4( warmStartScale, rollingImpulse.Z ), rollingMask );
			c->frictionImpulse.x = b3MulW4( warmStartScale, b3DotW4( frictionImpulse, tangent1 ) );
			c->frictionImpulse.y = b3MulW4( warmStartScale, b3DotW4( frictionImpulse, tangent2 ) );

			b3FloatW4 pointCountW =
				b3SetW4( (float)c->pointCounts[0], (float)c->pointCounts[1], (float)c->pointCounts[2], (float)c->pointCounts[3] );

			b3Vec3W4 centerA = { zeroW, zeroW, zeroW };
			b3Vec3W4 centerB = { zeroW, zeroW, zeroW };
			b3FloatW4 totalFrictionWeight = zeroW;

			for ( int pointIndex = 0; pointIndex < B3_MAX_MANIFOLD_POINTS; ++pointIndex )
			{
				b3ContactConstraintPointW4* cp = c->points + pointIndex;
				b3FloatW4 pointMask = b3GreaterThanW4( pointCountW, b3SplatW4( (float)pointIndex ) );

				const float* p0 = (const float*)( manifoldLanes[0]->points + pointIndex );
				const float* p1 = (const float*)( manifoldLanes[1]->points + pointIndex );
				const float* p2 = (const float*)( manifoldLanes[2]->points + pointIndex );
				const float* p3 = (const float*)( manifoldLanes[3]->points + pointIndex );

				b3Vec3W4 rA, rB;
				b3FloatW4 separation, normalImpulse;
				b3TransposeW4( b3LoadW4( p0 ), b3LoadW4( p1 ), b3LoadW4( p2 ), b3LoadW4( p3 ), &rA.X, &rA.Y, &rA.Z, &rB.X );
				b3TransposeW4( b3LoadW4( p0 + 4 ), b3LoadW4( p1 + 4 ), b3LoadW4( p2 + 4 ), b3LoadW4( p3 + 4 ), &rB.Y, &rB.Z,
							   &separation, &normalImpulse );

				rA.X = b3BlendW4( zeroW, rA.X, pointMask );
				rA.Y = b3BlendW4( zeroW, rA.Y, pointMask );
				rA.Z = b3BlendW4( zeroW, rA.Z, pointMask );
				rB.X = b3BlendW4( zeroW, rB.X, pointMask );
				rB.Y = b3BlendW4( zeroW, rB.Y, pointMask );
				rB.Z = b3BlendW4( zeroW, rB.Z, pointMask );
				separation = b3BlendW4( zeroW, separation, pointMask );
				normalImpulse = b3BlendW4( zeroW, normalImpulse, pointMask );

				// C0 friction center decay. Needed to prevent spinning top drift (GyroscopicPrecession sample).
				// See details in b3PrepareContacts_Mesh. This code should stay in sync.
				b3FloatW4 weight = b3MinW4( b3MaxW4( b3SubW4( twoW, b3MulW4( separation, invTau ) ), minFrictionWeight ), oneW );
				weight = b3BlendW4( zeroW, weight, pointMask );
				centerA = b3MulAddSVW4( centerA, weight, rA );
				centerB = b3MulAddSVW4( centerB, weight, rB );
				totalFrictionWeight = b3AddW4( totalFrictionWeight, weight );

				cp->anchorAs = rA;
				cp->anchorBs = rB;
				cp->baseSeparations = b3SubW4( separation, b3DotW4( b3SubVW4( rB, rA ), normal ) );
				cp->normalImpulses = b3MulW4( warmStartScale, normalImpulse );
				cp->totalNormalImpulses = zeroW;
				cp->relativeVelocities = zeroW;
				cp->restitutionImpulses = zeroW;

				b3Vec3W4 rnA = b3CrossW4( rA, normal );
				b3Vec3W4 rnB = b3CrossW4( rB, normal );
				b3FloatW4 kNormal = b3AddW4( mA, mB );
				kNormal = b3AddW4( kNormal, b3DotW4( rnA, b3MulMVW4( iA, rnA ) ) );
				kNormal = b3AddW4( kNormal, b3DotW4( rnB, b3MulMVW4( iB, rnB ) ) );
				b3FloatW4 valid = b3AndW4( b3GreaterThanW4( kNormal, zeroW ), pointMask );
				cp->normalMasses = b3BlendW4( zeroW, b3DivW4( oneW, kNormal ), valid );
			}

			b3FloatW4 invWeight =
				b3BlendW4( zeroW, b3DivW4( oneW, totalFrictionWeight ), b3GreaterThanW4( totalFrictionWeight, zeroW ) );
			centerA = b3MulSVW4( invWeight, centerA );
			centerB = b3MulSVW4( invWeight, centerB );
			c->centerA = centerA;
			c->centerB = centerB;

			for ( int pointIndex = 0; pointIndex < B3_MAX_MANIFOLD_POINTS; ++pointIndex )
			{
				b3ContactConstraintPointW4* cp = c->points + pointIndex;
				b3FloatW4 pointMask = b3GreaterThanW4( pointCountW, b3SplatW4( (float)pointIndex ) );
				b3Vec3W4 d = b3SubVW4( cp->anchorAs, centerA );
				cp->leverArms = b3BlendW4( zeroW, b3SqrtW4( b3DotW4( d, d ) ), pointMask );
			}

			{
				b3Vec3W4 rtA1 = b3CrossW4( centerA, tangent1 );
				b3Vec3W4 rtA2 = b3CrossW4( centerA, tangent2 );
				b3Vec3W4 rtB1 = b3CrossW4( centerB, tangent1 );
				b3Vec3W4 rtB2 = b3CrossW4( centerB, tangent2 );
				b3Vec3W4 iArtA1 = b3MulMVW4( iA, rtA1 );
				b3Vec3W4 iArtA2 = b3MulMVW4( iA, rtA2 );
				b3Vec3W4 iBrtB1 = b3MulMVW4( iB, rtB1 );
				b3Vec3W4 iBrtB2 = b3MulMVW4( iB, rtB2 );

				b3FloatW4 kxx = b3AddW4( b3AddW4( b3AddW4( mA, mB ), b3DotW4( rtA1, iArtA1 ) ), b3DotW4( rtB1, iBrtB1 ) );
				b3FloatW4 kyy = b3AddW4( b3AddW4( b3AddW4( mA, mB ), b3DotW4( rtA2, iArtA2 ) ), b3DotW4( rtB2, iBrtB2 ) );
				b3FloatW4 kxy = b3AddW4( b3DotW4( rtA1, iArtA2 ), b3DotW4( rtB1, iBrtB2 ) );

				b3FloatW4 det = b3SubW4( b3MulW4( kxx, kyy ), b3MulW4( kxy, kxy ) );
				b3FloatW4 valid = b3GreaterThanW4( b3AbsW4( det ), minDet );
				b3FloatW4 invDet = b3BlendW4( zeroW, b3DivW4( oneW, det ), valid );
				c->tangentMass.cxx = b3MulW4( invDet, kyy );
				c->tangentMass.cxy = b3NegW4( b3MulW4( invDet, kxy ) );
				c->tangentMass.cyy = b3MulW4( invDet, kxx );
			}

			b3SymMatrix3W4 invIAB = b3AddSymW4( iA, iB );

			{
				b3FloatW4 kTwist = b3DotW4( normal, b3MulMVW4( invIAB, normal ) );
				c->twistMass = b3BlendW4( zeroW, b3DivW4( oneW, kTwist ), b3GreaterThanW4( kTwist, zeroW ) );
			}

			if ( b3AllZeroW4( c->rollingResistance ) == false )
			{
				c->rollingMass = b3InvertSymW4( invIAB );
			}
			else
			{
				c->rollingMass = (b3SymMatrix3W4){ zeroW, zeroW, zeroW, zeroW, zeroW, zeroW };
			}

			// Only sample contact point normal velocity if needed.
			bool haveRestitution = b3AnyTrueW4( b3GreaterThanW4( c->restitution, zeroW ) );
			anyRestitution = anyRestitution || haveRestitution;

			if ( hitEventLanes != 0 || haveRestitution )
			{
				b3BodyStateW4 bA = b3GatherBodiesW4( states, c->indexA );
				b3BodyStateW4 bB = b3GatherBodiesW4( states, c->indexB );

				for ( int pointIndex = 0; pointIndex < B3_MAX_MANIFOLD_POINTS; ++pointIndex )
				{
					b3ContactConstraintPointW4* cp = c->points + pointIndex;

					b3Vec3W4 vrA = b3AddVW4( bA.v, b3CrossW4( bA.w, cp->anchorAs ) );
					b3Vec3W4 vrB = b3AddVW4( bB.v, b3CrossW4( bB.w, cp->anchorBs ) );
					b3FloatW4 vn = b3DotW4( normal, b3SubVW4( vrB, vrA ) );
					cp->relativeVelocities = vn;

					if ( hitEventLanes != 0 )
					{
						float normalVelocities[B3_SIMD_WIDTH];
						b3StoreW4( normalVelocities, vn );

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

void b3WarmStartContacts_ConvexW4( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( warm_start_contact, "Warm Start", b3_colorGreen, true );

	b3BodyState* states = context->states;
	b3ContactConstraintW4* constraints = context->graph->colors[block.colorIndex].wideConstraints;

	for ( int i = block.startIndex; i < block.startIndex + block.count; ++i )
	{
		b3ContactConstraintW4* c = constraints + i;
		b3BodyStateW4 bA = b3GatherBodiesW4( states, c->indexA );
		b3BodyStateW4 bB = b3GatherBodiesW4( states, c->indexB );

		int pointCount1 = b3MaxInt( c->pointCounts[0], c->pointCounts[1] );
		int pointCount2 = b3MaxInt( c->pointCounts[2], c->pointCounts[3] );
		int pointCount = b3MaxInt( pointCount1, pointCount2 );
		B3_VALIDATE( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3FloatW4 zeroW = b3ZeroW4();
		b3FloatW4 totalNormalImpulse = zeroW;
		b3Vec3W4 momentA = { zeroW, zeroW, zeroW };
		b3Vec3W4 momentB = { zeroW, zeroW, zeroW };

		for ( int j = 0; j < pointCount; ++j )
		{
			b3ContactConstraintPointW4* cp = c->points + j;
			b3FloatW4 normalImpulse = cp->normalImpulses;
			totalNormalImpulse = b3AddW4( totalNormalImpulse, normalImpulse );
			momentA = b3MulAddSVW4( momentA, normalImpulse, cp->anchorAs );
			momentB = b3MulAddSVW4( momentB, normalImpulse, cp->anchorBs );
		}

		b3Vec3W4 normal = c->normal;
		b3Vec3W4 frictionImpulse = b3MulSVW4( c->frictionImpulse.x, c->tangent1 );
		frictionImpulse = b3MulAddSVW4( frictionImpulse, c->frictionImpulse.y, c->tangent2 );

		b3Vec3W4 linearImpulse = b3MulAddSVW4( frictionImpulse, totalNormalImpulse, normal );

		b3Vec3W4 twistImpulse = b3MulSVW4( c->twistImpulse, normal );
		b3Vec3W4 angularImpulseA =
			b3AddVW4( b3AddVW4( b3CrossW4( momentA, normal ), b3CrossW4( c->centerA, frictionImpulse ) ), twistImpulse );
		b3Vec3W4 angularImpulseB =
			b3AddVW4( b3AddVW4( b3CrossW4( momentB, normal ), b3CrossW4( c->centerB, frictionImpulse ) ), twistImpulse );

		if ( b3AllZeroW4( c->rollingResistance ) == false )
		{
			angularImpulseA = b3AddVW4( angularImpulseA, c->rollingImpulse );
			angularImpulseB = b3AddVW4( angularImpulseB, c->rollingImpulse );
		}

		bA.w = b3MulSubMVW4( bA.w, c->invIA, angularImpulseA );
		bA.v = b3MulSubSVW4( bA.v, c->invMassA, linearImpulse );
		bB.w = b3MulAddMVW4( bB.w, c->invIB, angularImpulseB );
		bB.v = b3MulAddSVW4( bB.v, c->invMassB, linearImpulse );

		b3ScatterBodiesW4( states, c->indexA, &bA );
		b3ScatterBodiesW4( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( warm_start_contact );
}

// Solve the non-penetration constraints with the soft bias. No friction and no restitution.
void b3PushContacts_ConvexW4( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( push_contact, "Push Contact", b3_colorAliceBlue, true );

	b3BodyState* states = context->states;
	b3ContactConstraintW4* constraints = context->graph->colors[block.colorIndex].wideConstraints;
	b3FloatW4 inv_h = b3SplatW4( context->inv_h );
	b3FloatW4 contactSpeed = b3SplatW4( -context->world->contactSpeed );
	b3FloatW4 oneW = b3SplatW4( 1.0f );

	// Stiffer for static contacts to avoid bodies getting pushed through the ground. Selected per
	// lane from the null body index instead of stored per constraint.
	b3FloatW4 dynamicBiasRate = b3SplatW4( context->contactSoftness.massScale * context->contactSoftness.biasRate );
	b3FloatW4 dynamicMassScale = b3SplatW4( context->contactSoftness.massScale );
	b3FloatW4 dynamicImpulseScale = b3SplatW4( context->contactSoftness.impulseScale );
	b3FloatW4 staticBiasRate = b3SplatW4( context->staticSoftness.massScale * context->staticSoftness.biasRate );
	b3FloatW4 staticMassScale = b3SplatW4( context->staticSoftness.massScale );
	b3FloatW4 staticImpulseScale = b3SplatW4( context->staticSoftness.impulseScale );

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3ContactConstraintW4* c = constraints + wideIndex;

		int pointCount1 = b3MaxInt( c->pointCounts[0], c->pointCounts[1] );
		int pointCount2 = b3MaxInt( c->pointCounts[2], c->pointCounts[3] );
		int pointCount = b3MaxInt( pointCount1, pointCount2 );
		B3_VALIDATE( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3BodyStateW4 bA = b3GatherBodiesW4( states, c->indexA );
		b3BodyStateW4 bB = b3GatherBodiesW4( states, c->indexB );

		b3FloatW4 softMask = b3SoftMaskW4( c->indexA, c->indexB );
		b3FloatW4 biasRate = b3BlendW4( dynamicBiasRate, staticBiasRate, softMask );
		b3FloatW4 massScale = b3BlendW4( dynamicMassScale, staticMassScale, softMask );
		b3FloatW4 impulseScale = b3BlendW4( dynamicImpulseScale, staticImpulseScale, softMask );

		b3Vec3W4 dp = b3SubVW4( bB.dp, bA.dp );

		// Convert to normals to local space to reduce transform math.
		b3FloatW4 normalSeparation = b3DotW4( c->normal, dp );
		b3Vec3W4 normalA = b3InvRotateVectorW4( bA.dq, c->normal );
		b3Vec3W4 normalB = b3InvRotateVectorW4( bB.dq, c->normal );

		for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
		{
			b3ContactConstraintPointW4* cp = c->points + pointIndex;

			// Fixed anchor points for applying impulses
			b3Vec3W4 rA = cp->anchorAs;
			b3Vec3W4 rB = cp->anchorBs;

			b3FloatW4 s =
				b3AddW4( b3AddW4( normalSeparation, b3SubW4( b3DotW4( normalB, rB ), b3DotW4( normalA, rA ) ) ), cp->baseSeparations );

			// Apply speculative bias if separation is greater than zero, otherwise apply soft constraint bias
			b3FloatW4 separated = b3GreaterThanW4( s, b3ZeroW4() );

			// Speculative bias - positive
			b3FloatW4 specBias = b3MulW4( s, inv_h );

			// Overlap bias - negative
			b3FloatW4 overlapBias = b3MaxW4( b3MulW4( biasRate, s ), contactSpeed );
			b3FloatW4 velocityBias = b3BlendW4( overlapBias, specBias, separated );

			b3FloatW4 pointMassScale = b3BlendW4( massScale, oneW, separated );
			b3FloatW4 pointImpulseScale = b3BlendW4( impulseScale, b3ZeroW4(), separated );

			// Relative velocity at contact
			b3Vec3W4 vrA = b3AddVW4( bA.v, b3CrossW4( bA.w, rA ) );
			b3Vec3W4 vrB = b3AddVW4( bB.v, b3CrossW4( bB.w, rB ) );
			b3FloatW4 vn = b3DotW4( b3SubVW4( vrB, vrA ), c->normal );

			// Compute normal impulse
			b3FloatW4 negImpulse = b3AddW4( b3MulW4( cp->normalMasses, b3AddW4( b3MulW4( pointMassScale, vn ), velocityBias ) ),
										  b3MulW4( pointImpulseScale, cp->normalImpulses ) );

			// Clamp the accumulated impulse
			b3FloatW4 newImpulse = b3MaxW4( b3SubW4( cp->normalImpulses, negImpulse ), b3ZeroW4() );
			b3FloatW4 deltaImpulse = b3SubW4( newImpulse, cp->normalImpulses );
			cp->normalImpulses = newImpulse;

			// Apply contact impulse
			b3Vec3W4 P = b3MulSVW4( deltaImpulse, c->normal );
			bA.w = b3MulSubMVW4( bA.w, c->invIA, b3CrossW4( rA, P ) );
			bA.v = b3MulSubSVW4( bA.v, c->invMassA, P );
			bB.w = b3MulAddMVW4( bB.w, c->invIB, b3CrossW4( rB, P ) );
			bB.v = b3MulAddSVW4( bB.v, c->invMassB, P );
		}

		b3ScatterBodiesW4( states, c->indexA, &bA );
		b3ScatterBodiesW4( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( push_contact );
}

// Solve the normal constraint, friction, and rolling resistance.
void b3SolveContacts_ConvexW4( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( solve_contact, "Solve Contact", b3_colorAliceBlue, true );

	b3BodyState* states = context->states;
	b3ContactConstraintW4* constraints = context->graph->colors[block.colorIndex].wideConstraints;
	b3FloatW4 inv_h = b3SplatW4( context->inv_h );
	b3FloatW4 oneW = b3SplatW4( 1.0f );
	b3FloatW4 epsilonW = b3SplatW4( FLT_EPSILON );

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3ContactConstraintW4* c = constraints + wideIndex;

		int pointCount1 = b3MaxInt( c->pointCounts[0], c->pointCounts[1] );
		int pointCount2 = b3MaxInt( c->pointCounts[2], c->pointCounts[3] );
		int pointCount = b3MaxInt( pointCount1, pointCount2 );
		B3_VALIDATE( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3BodyStateW4 bA = b3GatherBodiesW4( states, c->indexA );
		b3BodyStateW4 bB = b3GatherBodiesW4( states, c->indexB );

		b3Vec3W4 dp = b3SubVW4( bB.dp, bA.dp );
		b3FloatW4 normalSeparation = b3DotW4( c->normal, dp );
		b3Vec3W4 normalA = b3InvRotateVectorW4( bA.dq, c->normal );
		b3Vec3W4 normalB = b3InvRotateVectorW4( bB.dq, c->normal );

		b3FloatW4 totalNormalImpulse = b3ZeroW4();
		b3FloatW4 totalTwistLimit = b3ZeroW4();

		for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
		{
			b3ContactConstraintPointW4* cp = c->points + pointIndex;

			// Fixed anchor points for applying impulses
			b3Vec3W4 rA = cp->anchorAs;
			b3Vec3W4 rB = cp->anchorBs;

			b3FloatW4 s =
				b3AddW4( b3AddW4( normalSeparation, b3SubW4( b3DotW4( normalB, rB ), b3DotW4( normalA, rA ) ) ), cp->baseSeparations );

			// Speculative bias, positive if separated and zero if overlapped
			b3FloatW4 velocityBias = b3MaxW4( b3ZeroW4(), b3MulW4( s, inv_h ) );

			// Relative velocity at contact
			b3Vec3W4 vrA = b3AddVW4( bA.v, b3CrossW4( bA.w, rA ) );
			b3Vec3W4 vrB = b3AddVW4( bB.v, b3CrossW4( bB.w, rB ) );
			b3FloatW4 vn = b3DotW4( b3SubVW4( vrB, vrA ), c->normal );

			// Compute normal impulse
			b3FloatW4 negImpulse = b3MulW4( cp->normalMasses, b3AddW4( vn, velocityBias ) );

			// Clamp the accumulated impulse
			b3FloatW4 newImpulse = b3MaxW4( b3SubW4( cp->normalImpulses, negImpulse ), b3ZeroW4() );
			b3FloatW4 deltaImpulse = b3SubW4( newImpulse, cp->normalImpulses );
			cp->normalImpulses = newImpulse;
			cp->totalNormalImpulses = b3AddW4( cp->totalNormalImpulses, newImpulse );
			totalNormalImpulse = b3AddW4( totalNormalImpulse, newImpulse );
			totalTwistLimit = b3AddW4( totalTwistLimit, b3MulW4( cp->leverArms, newImpulse ) );

			// Apply contact impulse
			b3Vec3W4 P = b3MulSVW4( deltaImpulse, c->normal );
			bA.w = b3MulSubMVW4( bA.w, c->invIA, b3CrossW4( rA, P ) );
			bA.v = b3MulSubSVW4( bA.v, c->invMassA, P );
			bB.w = b3MulAddMVW4( bB.w, c->invIB, b3CrossW4( rB, P ) );
			bB.v = b3MulAddSVW4( bB.v, c->invMassB, P );
		}

		// Rolling resistance
		if ( b3AllZeroW4( c->rollingResistance ) == false )
		{
			// flip A/B order to negate
			b3Vec3W4 deltaImpulse = b3MulMVW4( c->rollingMass, b3SubVW4( bA.w, bB.w ) );
			b3Vec3W4 oldImpulse = c->rollingImpulse;
			c->rollingImpulse = b3AddVW4( oldImpulse, deltaImpulse );

			b3FloatW4 maxImpulse = b3MulW4( c->rollingResistance, totalNormalImpulse );
			b3FloatW4 lengthSquared = b3DotW4( c->rollingImpulse, c->rollingImpulse );

			// if ( magSqr > maxLambda * maxLambda + FLT_EPSILON )
			//{
			//	c->rollingImpulse *= maxLambda / sqrtf( magSqr );
			// }

			b3FloatW4 mask = b3GreaterThanW4( lengthSquared, b3MulAddW4( epsilonW, maxImpulse, maxImpulse ) );

			// No approximate _mm_rsqrt_ps here to maintain cross-platform determinism
			b3FloatW4 normalize = b3DivW4( maxImpulse, b3AddW4( b3SqrtW4( lengthSquared ), epsilonW ) );
			b3FloatW4 scale = b3BlendW4( oneW, normalize, mask );

			// Ensure zero rolling resistance yields no impulse
			b3FloatW4 rollingMask = b3GreaterThanW4( c->rollingResistance, b3ZeroW4() );
			scale = b3BlendW4( b3ZeroW4(), scale, rollingMask );

			c->rollingImpulse = b3MulSVW4( scale, c->rollingImpulse );

			deltaImpulse = b3SubVW4( c->rollingImpulse, oldImpulse );

			bA.w = b3MulSubMVW4( bA.w, c->invIA, deltaImpulse );
			bB.w = b3MulAddMVW4( bB.w, c->invIB, deltaImpulse );
		}

		// Central twist friction
		{
			b3FloatW4 twistSpeed = b3DotW4( c->normal, b3SubVW4( bB.w, bA.w ) );
			b3FloatW4 maxLambda = b3MulW4( c->friction, totalTwistLimit );
			b3FloatW4 deltaImpulse = b3NegW4( b3MulW4( c->twistMass, twistSpeed ) );
			b3FloatW4 oldImpulse = c->twistImpulse;
			c->twistImpulse = b3SymClampW4( b3AddW4( oldImpulse, deltaImpulse ), maxLambda );
			deltaImpulse = b3SubW4( c->twistImpulse, oldImpulse );

			b3Vec3W4 L = b3MulSVW4( deltaImpulse, c->normal );
			bA.w = b3MulSubMVW4( bA.w, c->invIA, L );
			bB.w = b3MulAddMVW4( bB.w, c->invIB, L );
		}

		// Central friction
		{
			b3Vec3W4 tangent1 = c->tangent1;
			b3Vec3W4 tangent2 = c->tangent2;

			// Fixed anchor points for applying impulses
			b3Vec3W4 rA = c->centerA;
			b3Vec3W4 rB = c->centerB;

			// Relative tangent velocity at contact
			b3Vec3W4 vrA = b3AddVW4( bA.v, b3CrossW4( bA.w, rA ) );
			b3Vec3W4 vrB = b3AddVW4( bB.v, b3CrossW4( bB.w, rB ) );
			b3Vec3W4 vr = b3SubVW4( vrB, vrA );
			b3Vec2W4 vt = {
				b3SubW4( b3DotW4( vr, tangent1 ), c->tangentVelocity1 ),
				b3SubW4( b3DotW4( vr, tangent2 ), c->tangentVelocity2 ),
			};

			// Incremental tangent impulse
			b3Vec2W4 deltaImpulse = b3MulMV2W4( c->tangentMass, vt );
			deltaImpulse = (b3Vec2W4){ b3NegW4( deltaImpulse.x ), b3NegW4( deltaImpulse.y ) };
			b3Vec2W4 newImpulse = b3AddV2W4( c->frictionImpulse, deltaImpulse );

			b3FloatW4 friction = c->friction;
			b3FloatW4 maxImpulse = b3MulW4( friction, totalNormalImpulse );

			// Clamp the accumulated impulse
			b3FloatW4 lengthSquared = b3AddW4( b3MulW4( newImpulse.x, newImpulse.x ), b3MulW4( newImpulse.y, newImpulse.y ) );

			// Max impulse can be zero
			b3FloatW4 mask = b3GreaterThanW4( lengthSquared, b3MulW4( maxImpulse, maxImpulse ) );

			// No approximate _mm_rsqrt_ps here to maintain cross-platform determinism. Add epsilon to avoid divide by
			// zero.
			b3FloatW4 normalize = b3DivW4( maxImpulse, b3AddW4( b3SqrtW4( lengthSquared ), epsilonW ) );
			b3FloatW4 scale = b3BlendW4( oneW, normalize, mask );
			newImpulse = (b3Vec2W4){
				b3MulW4( scale, newImpulse.x ),
				b3MulW4( scale, newImpulse.y ),
			};

			deltaImpulse = (b3Vec2W4){
				b3SubW4( newImpulse.x, c->frictionImpulse.x ),
				b3SubW4( newImpulse.y, c->frictionImpulse.y ),
			};

			c->frictionImpulse = newImpulse;

			// Apply delta impulse
			b3Vec3W4 P = b3AddVW4( b3MulSVW4( deltaImpulse.x, tangent1 ), b3MulSVW4( deltaImpulse.y, tangent2 ) );
			bA.w = b3MulSubMVW4( bA.w, c->invIA, b3CrossW4( rA, P ) );
			bA.v = b3MulSubSVW4( bA.v, c->invMassA, P );
			bB.w = b3MulAddMVW4( bB.w, c->invIB, b3CrossW4( rB, P ) );
			bB.v = b3MulAddSVW4( bB.v, c->invMassB, P );
		}

		b3ScatterBodiesW4( states, c->indexA, &bA );
		b3ScatterBodiesW4( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( solve_contact );
}

void b3ApplyRestitution_ConvexW4( b3SolverBlock block, b3StepContext* context )
{
	b3TracyCZoneNC( restitution, "Restitution", b3_colorDodgerBlue, true );

	b3BodyState* states = context->states;
	b3ContactConstraintW4* constraints = context->graph->colors[block.colorIndex].wideConstraints;
	b3FloatW4 inv_h = b3SplatW4( context->inv_h );
	b3FloatW4 negRestitutionThreshold = b3SplatW4( -context->world->restitutionThreshold );
	b3FloatW4 zeroW = b3ZeroW4();
	bool propagate = context->world->enableRestitutionPropagation;

	for ( int wideIndex = block.startIndex; wideIndex < block.startIndex + block.count; ++wideIndex )
	{
		b3ContactConstraintW4* c = constraints + wideIndex;
		if ( propagate == false && b3AllZeroW4( c->restitution ) )
		{
			continue;
		}

		int pointCount1 = b3MaxInt( c->pointCounts[0], c->pointCounts[1] );
		int pointCount2 = b3MaxInt( c->pointCounts[2], c->pointCounts[3] );
		int pointCount = b3MaxInt( pointCount1, pointCount2 );
		B3_VALIDATE( 0 < pointCount && pointCount <= B3_MAX_MANIFOLD_POINTS );

		b3BodyStateW4 bA = b3GatherBodiesW4( states, c->indexA );
		b3BodyStateW4 bB = b3GatherBodiesW4( states, c->indexB );

		b3FloatW4 restitutionMask = b3GreaterThanW4( c->restitution, zeroW );
		b3Vec3W4 dp = b3SubVW4( bB.dp, bA.dp );
		b3Matrix3W4 dqA = b3MakeMatrixFromQuatW4( bA.dq );
		b3Matrix3W4 dqB = b3MakeMatrixFromQuatW4( bB.dq );

		for ( int pointIndex = 0; pointIndex < pointCount; ++pointIndex )
		{
			b3ContactConstraintPointW4* cp = c->points + pointIndex;

			b3Vec3W4 rA = cp->anchorAs;
			b3Vec3W4 rB = cp->anchorBs;

			b3FloatW4 normalMass = propagate ? cp->normalMasses : b3BlendW4( zeroW, cp->normalMasses, restitutionMask );

			b3FloatW4 compressionImpulse = b3SubW4( cp->totalNormalImpulses, cp->restitutionImpulses );
			b3FloatW4 armed = b3AndW4( b3AndW4( restitutionMask, b3LessThanW4( cp->relativeVelocities, negRestitutionThreshold ) ),
									 b3GreaterThanW4( compressionImpulse, zeroW ) );

			b3Vec3W4 rsA = b3MulM3VW4( dqA, rA );
			b3Vec3W4 rsB = b3MulM3VW4( dqB, rB );
			b3Vec3W4 ds = b3AddVW4( dp, b3SubVW4( rsB, rsA ) );
			b3FloatW4 s = b3AddW4( b3DotW4( c->normal, ds ), cp->baseSeparations );

			b3FloatW4 specBias = b3MaxW4( zeroW, b3MulW4( s, inv_h ) );
			b3FloatW4 velocityBias = b3BlendW4( specBias, b3MulW4( c->restitution, cp->relativeVelocities ), armed );

			b3Vec3W4 vrA = b3AddVW4( bA.v, b3CrossW4( bA.w, rA ) );
			b3Vec3W4 vrB = b3AddVW4( bB.v, b3CrossW4( bB.w, rB ) );
			b3FloatW4 vn = b3DotW4( b3SubVW4( vrB, vrA ), c->normal );

			b3FloatW4 negImpulse = b3MulW4( normalMass, b3AddW4( vn, velocityBias ) );

			b3FloatW4 newImpulse = b3MaxW4( b3SubW4( cp->normalImpulses, negImpulse ), zeroW );
			b3FloatW4 impulse = b3SubW4( newImpulse, cp->normalImpulses );

			b3FloatW4 approachImpulse = b3MinW4( b3MaxW4( b3NegW4( b3MulW4( normalMass, vn ) ), zeroW ), b3MaxW4( impulse, zeroW ) );
			b3FloatW4 allowance =
				b3SubW4( b3MulW4( c->restitution, b3AddW4( compressionImpulse, approachImpulse ) ), cp->restitutionImpulses );
			b3FloatW4 maxImpulse = b3AddW4( approachImpulse, b3MaxW4( allowance, zeroW ) );
			impulse = b3BlendW4( impulse, b3MinW4( impulse, maxImpulse ), armed );

			cp->normalImpulses = b3AddW4( cp->normalImpulses, impulse );
			cp->restitutionImpulses = b3AddW4( cp->restitutionImpulses, b3SubW4( impulse, approachImpulse ) );
			cp->totalNormalImpulses = b3AddW4( cp->totalNormalImpulses, impulse );

			b3Vec3W4 P = b3MulSVW4( impulse, c->normal );
			bA.w = b3MulSubMVW4( bA.w, c->invIA, b3CrossW4( rA, P ) );
			bA.v = b3MulSubSVW4( bA.v, c->invMassA, P );
			bB.w = b3MulAddMVW4( bB.w, c->invIB, b3CrossW4( rB, P ) );
			bB.v = b3MulAddSVW4( bB.v, c->invMassB, P );
		}

		b3ScatterBodiesW4( states, c->indexA, &bA );
		b3ScatterBodiesW4( states, c->indexB, &bB );
	}

	b3TracyCZoneEnd( restitution );
}

// Store impulses by contact constraint
void b3StoreImpulses_ConvexW4( b3SolverBlock block, b3StepContext* context, int workerIndex )
{
	b3TracyCZoneNC( store_impulses, "Store", b3_colorFireBrick, true );

	b3World* world = context->world;
	b3WidePrepareSpan* spans = context->widePrepareSpans;
	const b3ContactConstraintW4* wideBase = context->wideConstraints;
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
			const b3ContactConstraintW4* c = wideBase + wideIndex;
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
					const b3ContactConstraintPointW4* cp = c->points + pointIndex;
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

void b3PrepareContacts_Convex( b3SolverBlock block, b3StepContext* context )
{
	b3PrepareContacts_ConvexW4( block, context );
}

void b3WarmStartContacts_Convex( b3SolverBlock block, b3StepContext* context )
{
	b3WarmStartContacts_ConvexW4( block, context );
}

void b3PushContacts_Convex( b3SolverBlock block, b3StepContext* context )
{
	b3PushContacts_ConvexW4( block, context );
}

void b3SolveContacts_Convex( b3SolverBlock block, b3StepContext* context )
{
	b3SolveContacts_ConvexW4( block, context );
}

void b3ApplyRestitution_Convex( b3SolverBlock block, b3StepContext* context )
{
	b3ApplyRestitution_ConvexW4( block, context );
}

void b3StoreImpulses_Convex( b3SolverBlock block, b3StepContext* context, int workerIndex )
{
	b3StoreImpulses_ConvexW4( block, context, workerIndex );
}

#undef B3_SIMD_WIDTH

void b3PrepareContacts_Overflow( b3StepContext* context )
{
	b3ConstraintGraph* graph = context->graph;
	b3GraphColor* color = graph->colors + B3_OVERFLOW_INDEX;

	B3_ASSERT( color->contacts.count <= UINT16_MAX );
	uint16_t count = (uint16_t)color->contacts.count;
	if ( count == 0 )
	{
		return;
	}

	b3SolverBlock block = {
		.startIndex = 0,
		.count = count,
		.blockType = b3_overflowBlock,
		.colorIndex = B3_OVERFLOW_INDEX,
	};

	b3PrepareContacts_Mesh( block, context );
}

void b3WarmStartContacts_Overflow( b3StepContext* context )
{
	b3ConstraintGraph* graph = context->graph;
	b3GraphColor* color = graph->colors + B3_OVERFLOW_INDEX;

	uint16_t count = (uint16_t)color->contacts.count;
	if ( count == 0 )
	{
		return;
	}

	b3SolverBlock block = {
		.startIndex = 0,
		.count = count,
		.blockType = b3_overflowBlock,
		.colorIndex = B3_OVERFLOW_INDEX,
	};

	b3WarmStartContacts_Mesh( block, context );
}

void b3PushContacts_Overflow( b3StepContext* context )
{
	b3ConstraintGraph* graph = context->graph;
	b3GraphColor* color = graph->colors + B3_OVERFLOW_INDEX;

	uint16_t count = (uint16_t)color->contacts.count;
	if ( count == 0 )
	{
		return;
	}

	b3SolverBlock block = {
		.startIndex = 0,
		.count = count,
		.blockType = b3_overflowBlock,
		.colorIndex = B3_OVERFLOW_INDEX,
	};

	b3PushContacts_Mesh( block, context );
}

void b3SolveContacts_Overflow( b3StepContext* context )
{
	b3ConstraintGraph* graph = context->graph;
	b3GraphColor* color = graph->colors + B3_OVERFLOW_INDEX;

	uint16_t count = (uint16_t)color->contacts.count;
	if ( count == 0 )
	{
		return;
	}

	b3SolverBlock block = {
		.startIndex = 0,
		.count = count,
		.blockType = b3_overflowBlock,
		.colorIndex = B3_OVERFLOW_INDEX,
	};

	b3SolveContacts_Mesh( block, context );
}

void b3ApplyRestitution_Overflow( b3StepContext* context )
{
	b3ConstraintGraph* graph = context->graph;
	b3GraphColor* color = graph->colors + B3_OVERFLOW_INDEX;

	uint16_t count = (uint16_t)color->contacts.count;
	if ( count == 0 )
	{
		return;
	}

	b3SolverBlock block = {
		.startIndex = 0,
		.count = count,
		.blockType = b3_overflowBlock,
		.colorIndex = B3_OVERFLOW_INDEX,
	};

	b3ApplyRestitution_Mesh( block, context );
}

void b3StoreImpulses_Overflow( b3StepContext* context )
{
	b3ConstraintGraph* graph = context->graph;
	b3GraphColor* color = graph->colors + B3_OVERFLOW_INDEX;

	uint16_t count = (uint16_t)color->contacts.count;
	if ( count == 0 )
	{
		return;
	}

	b3SolverBlock block = {
		.startIndex = 0,
		.count = count,
		.blockType = b3_overflowBlock,
		.colorIndex = B3_OVERFLOW_INDEX,
	};

	b3StoreImpulses_Mesh( block, context, 0 );
}
