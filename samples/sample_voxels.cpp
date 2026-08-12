// SPDX-FileCopyrightText: 2026 Ethan Rutherford
// SPDX-License-Identifier: MIT

#include "gfx/draw.h"
#include "human.h"
#include "imgui.h"
#include "mesh_loader.h"
#include "sample.h"
#include "utils.h"

#include "box3d/box3d.h"

static constexpr float fourThirds = 4.0f / 3.0f;

b3VoxelData* createBoxy()
{
	b3VoxelDef voxels[101] = {
		// Bottom back row
		{ b3EncodeVoxel( 0, 0, 0 ), 0 },
		{ b3EncodeVoxel( 1, 0, 0 ), 0 },
		{ b3EncodeVoxel( 2, 0, 0 ), 0 },
		{ b3EncodeVoxel( 3, 0, 0 ), 0 },
		{ b3EncodeVoxel( 4, 0, 0 ), 0 },
		{ b3EncodeVoxel( 5, 0, 0 ), 0 },
		{ b3EncodeVoxel( 6, 0, 0 ), 0 },
		{ b3EncodeVoxel( 7, 0, 0 ), 0 },
		{ b3EncodeVoxel( 8, 0, 0 ), 0 },
		// Top back row
		{ b3EncodeVoxel( 0, 8, 0 ), 0 },
		{ b3EncodeVoxel( 1, 8, 0 ), 0 },
		{ b3EncodeVoxel( 2, 8, 0 ), 0 },
		{ b3EncodeVoxel( 3, 8, 0 ), 0 },
		{ b3EncodeVoxel( 4, 8, 0 ), 0 },
		{ b3EncodeVoxel( 5, 8, 0 ), 0 },
		{ b3EncodeVoxel( 6, 8, 0 ), 0 },
		{ b3EncodeVoxel( 7, 8, 0 ), 0 },
		{ b3EncodeVoxel( 8, 8, 0 ), 0 },
		// Bottom front row
		{ b3EncodeVoxel( 0, 0, 8 ), 0 },
		{ b3EncodeVoxel( 1, 0, 8 ), 0 },
		{ b3EncodeVoxel( 2, 0, 8 ), 0 },
		{ b3EncodeVoxel( 3, 0, 8 ), 0 },
		{ b3EncodeVoxel( 4, 0, 8 ), 0 },
		{ b3EncodeVoxel( 5, 0, 8 ), 0 },
		{ b3EncodeVoxel( 6, 0, 8 ), 0 },
		{ b3EncodeVoxel( 7, 0, 8 ), 0 },
		{ b3EncodeVoxel( 8, 0, 8 ), 0 },
		// Top front row
		{ b3EncodeVoxel( 0, 8, 8 ), 0 },
		{ b3EncodeVoxel( 1, 8, 8 ), 0 },
		{ b3EncodeVoxel( 2, 8, 8 ), 0 },
		{ b3EncodeVoxel( 3, 8, 8 ), 0 },
		{ b3EncodeVoxel( 4, 8, 8 ), 0 },
		{ b3EncodeVoxel( 5, 8, 8 ), 0 },
		{ b3EncodeVoxel( 6, 8, 8 ), 0 },
		{ b3EncodeVoxel( 7, 8, 8 ), 0 },
		{ b3EncodeVoxel( 8, 8, 8 ), 0 },
		// Left back column
		{ b3EncodeVoxel( 0, 1, 0 ), 0 },
		{ b3EncodeVoxel( 0, 2, 0 ), 0 },
		{ b3EncodeVoxel( 0, 3, 0 ), 0 },
		{ b3EncodeVoxel( 0, 4, 0 ), 0 },
		{ b3EncodeVoxel( 0, 5, 0 ), 0 },
		{ b3EncodeVoxel( 0, 6, 0 ), 0 },
		{ b3EncodeVoxel( 0, 7, 0 ), 0 },
		// Right back column
		{ b3EncodeVoxel( 8, 1, 0 ), 0 },
		{ b3EncodeVoxel( 8, 2, 0 ), 0 },
		{ b3EncodeVoxel( 8, 3, 0 ), 0 },
		{ b3EncodeVoxel( 8, 4, 0 ), 0 },
		{ b3EncodeVoxel( 8, 5, 0 ), 0 },
		{ b3EncodeVoxel( 8, 6, 0 ), 0 },
		{ b3EncodeVoxel( 8, 7, 0 ), 0 },
		// Left front column
		{ b3EncodeVoxel( 0, 1, 8 ), 0 },
		{ b3EncodeVoxel( 0, 2, 8 ), 0 },
		{ b3EncodeVoxel( 0, 3, 8 ), 0 },
		{ b3EncodeVoxel( 0, 4, 8 ), 0 },
		{ b3EncodeVoxel( 0, 5, 8 ), 0 },
		{ b3EncodeVoxel( 0, 6, 8 ), 0 },
		{ b3EncodeVoxel( 0, 7, 8 ), 0 },
		// Right front column
		{ b3EncodeVoxel( 8, 1, 8 ), 0 },
		{ b3EncodeVoxel( 8, 2, 8 ), 0 },
		{ b3EncodeVoxel( 8, 3, 8 ), 0 },
		{ b3EncodeVoxel( 8, 4, 8 ), 0 },
		{ b3EncodeVoxel( 8, 5, 8 ), 0 },
		{ b3EncodeVoxel( 8, 6, 8 ), 0 },
		{ b3EncodeVoxel( 8, 7, 8 ), 0 },
		// Bottom left line
		{ b3EncodeVoxel( 0, 0, 1 ), 0 },
		{ b3EncodeVoxel( 0, 0, 2 ), 0 },
		{ b3EncodeVoxel( 0, 0, 3 ), 0 },
		{ b3EncodeVoxel( 0, 0, 4 ), 0 },
		{ b3EncodeVoxel( 0, 0, 5 ), 0 },
		{ b3EncodeVoxel( 0, 0, 6 ), 0 },
		{ b3EncodeVoxel( 0, 0, 7 ), 0 },
		// Bottom right line
		{ b3EncodeVoxel( 8, 0, 1 ), 0 },
		{ b3EncodeVoxel( 8, 0, 2 ), 0 },
		{ b3EncodeVoxel( 8, 0, 3 ), 0 },
		{ b3EncodeVoxel( 8, 0, 4 ), 0 },
		{ b3EncodeVoxel( 8, 0, 5 ), 0 },
		{ b3EncodeVoxel( 8, 0, 6 ), 0 },
		{ b3EncodeVoxel( 8, 0, 7 ), 0 },
		// Top left line
		{ b3EncodeVoxel( 0, 8, 1 ), 0 },
		{ b3EncodeVoxel( 0, 8, 2 ), 0 },
		{ b3EncodeVoxel( 0, 8, 3 ), 0 },
		{ b3EncodeVoxel( 0, 8, 4 ), 0 },
		{ b3EncodeVoxel( 0, 8, 5 ), 0 },
		{ b3EncodeVoxel( 0, 8, 6 ), 0 },
		{ b3EncodeVoxel( 0, 8, 7 ), 0 },
		// Top right line
		{ b3EncodeVoxel( 8, 8, 1 ), 0 },
		{ b3EncodeVoxel( 8, 8, 2 ), 0 },
		{ b3EncodeVoxel( 8, 8, 3 ), 0 },
		{ b3EncodeVoxel( 8, 8, 4 ), 0 },
		{ b3EncodeVoxel( 8, 8, 5 ), 0 },
		{ b3EncodeVoxel( 8, 8, 6 ), 0 },
		{ b3EncodeVoxel( 8, 8, 7 ), 0 },
		// Boxy face
		{ b3EncodeVoxel( 3, 6, 0 ), 0 },
		{ b3EncodeVoxel( 3, 5, 0 ), 0 },
		{ b3EncodeVoxel( 5, 6, 0 ), 0 },
		{ b3EncodeVoxel( 5, 5, 0 ), 0 },
		{ b3EncodeVoxel( 2, 3, 0 ), 0 },
		{ b3EncodeVoxel( 3, 2, 0 ), 0 },
		{ b3EncodeVoxel( 4, 2, 0 ), 0 },
		{ b3EncodeVoxel( 5, 2, 0 ), 0 },
		{ b3EncodeVoxel( 6, 3, 0 ), 0 },
	};

	b3VoxelsDef voxelsDef = {};
	voxelsDef.voxelCount = sizeof( voxels ) / sizeof( b3VoxelDef );
	voxelsDef.voxels = voxels;
	return b3CreateVoxels( &voxelsDef );
}

b3VoxelData* createVoxelSphere( float radius )
{
	// reserve enough space for the maximum number of voxels in a sphere of the given radius
	// volume of a sphere is (4/3) * r^3 * pi, so we use pi = 3.3 to account for voxelization.
	b3VoxelDef* voxels = new b3VoxelDef[(int)( radius * radius * radius * fourThirds * 3.3 )];

	float r2 = radius * radius;

	int count = 0;
	for ( float x = -radius + 0.5f; x < radius; x += 1.0f )
	{
		for ( float y = -radius + 0.5f; y < radius; y += 1.0f )
		{
			for ( float z = -radius + 0.5f; z < radius; z += 1.0f )
			{
				if ( x * x + y * y + z * z <= r2 )
				{
					voxels[count++] = { b3EncodeVoxel( x + radius, y + radius, z + radius ), 0 };
				}
			}
		}
	}

	b3VoxelsDef voxelsDef = {};
	voxelsDef.voxelCount = count;
	voxelsDef.voxels = voxels;
	auto result = b3CreateVoxels( &voxelsDef );
	delete[] voxels;
	return result;
}

b3VoxelData* createVoxelCapsule( float length, float radius )
{
	// volume of a capsule is (4/3 * r + l) * r^2 * pi, so we use pi = 3.3 to account for voxelization.
	b3VoxelDef* voxels = new b3VoxelDef[(int)( ( fourThirds * radius + length ) * radius * radius * 3.3 )];

	float halfLength = length * 0.5f + radius;
	float r2 = radius * radius;

	int count = 0;
	for ( float x = -halfLength + 0.5f; x < halfLength; x += 1.0f )
	{
		for ( float y = -radius + 0.5f; y < radius; y += 1.0f )
		{
			for ( float z = -radius + 0.5f; z < radius; z += 1.0f )
			{
				if ( x < -length * 0.5f || x > length * 0.5f )
				{
					float dx = abs( x ) - length * 0.5f;
					if ( dx * dx + y * y + z * z <= r2 )
					{
						voxels[count++] = { b3EncodeVoxel( x + halfLength, y + radius, z + radius ), 0 };
					}
				}
				else
				{
					if ( y * y + z * z <= r2 )
					{
						voxels[count++] = { b3EncodeVoxel( x + halfLength, y + radius, z + radius ), 0 };
					}
				}
			}
		}
	}

	b3VoxelsDef voxelsDef = {};
	voxelsDef.voxelCount = count;
	voxelsDef.voxels = voxels;
	auto result = b3CreateVoxels( &voxelsDef );
	delete[] voxels;
	return result;
}

b3VoxelData* createVoxelBox( int width, int height, int depth )
{
	// reserve enough space for the maximum number of voxels in a box of the given dimensions
	b3VoxelDef* voxels = new b3VoxelDef[width * height * depth];

	int count = 0;
	for ( int x = 0; x < width; ++x )
	{
		for ( int y = 0; y < height; ++y )
		{
			for ( int z = 0; z < depth; ++z )
			{
				voxels[count++] = { b3EncodeVoxel( x, y, z ), 0 };
			}
		}
	}

	b3VoxelsDef voxelsDef = {};
	voxelsDef.voxelCount = count;
	voxelsDef.voxels = voxels;
	auto result = b3CreateVoxels( &voxelsDef );
	delete[] voxels;
	return result;
}

b3VoxelData* createVoxelTorus( float majorRadius, float minorRadius )
{
	// reserve enough space for the maximum number of voxels in a torus of the given dimensions
	// volume of a torus is 2 * R * r^2 * pi^2, so we use pi = 3.3 to account for voxelization.
	b3VoxelDef* voxels = new b3VoxelDef[(int)( 2.0f * majorRadius * minorRadius * minorRadius * 3.3 * 3.3 )];

	float halfLength = majorRadius + minorRadius;
	float r2 = minorRadius * minorRadius;

	int count = 0;
	for ( float x = -halfLength + 0.5f; x < halfLength; x += 1.0f )
	{
		for ( float y = -halfLength + 0.5f; y < halfLength; y += 1.0f )
		{
			for ( float z = -minorRadius + 0.5f; z < minorRadius; z += 1.0f )
			{
				float dist2 = ( sqrtf( x * x + y * y ) - majorRadius ) * ( sqrtf( x * x + y * y ) - majorRadius ) + z * z;
				if ( dist2 <= r2 )
				{
					voxels[count++] = { b3EncodeVoxel( x + halfLength, y + halfLength, z + minorRadius ), 0 };
				}
			}
		}
	}

	b3VoxelsDef voxelsDef = {};
	voxelsDef.voxelCount = count;
	voxelsDef.voxels = voxels;
	auto result = b3CreateVoxels( &voxelsDef );
	delete[] voxels;
	return result;
}

b3Vec3 getVoxelCentroid( b3VoxelData* voxels, float scale )
{
	b3Vec3 centroid = b3AABB_Center( voxels->bounds );
	return b3MulSV( scale, centroid );
}

class BasicVoxels : public Sample
{
public:
	explicit BasicVoxels( SampleContext* context )
		: Sample( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 0.0f, 25.0f, 10.0f, b3Pos_zero );
		}

		AddGroundBox( 20.0f );

		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3BodyType::b3_dynamicBody;
		bodyDef.position = { 0.0f, 2.5f, 0.0f };
		b3BodyId voxelBody = b3CreateBody( m_worldId, &bodyDef );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		m_voxels = createBoxy();

		b3CreateVoxelShape( voxelBody, &shapeDef, m_voxels, 0.1f );
	}

	~BasicVoxels() override
	{
		b3DestroyVoxels( m_voxels );
	}

	static Sample* Create( SampleContext* context )
	{
		return new BasicVoxels( context );
	}

	b3VoxelData* m_voxels = nullptr;
};

static int sampleBasicVoxels = RegisterSample( "Voxels", "Basic Voxel Shape", BasicVoxels::Create );

class VoxelRayCurtain : public Sample
{
public:
	explicit VoxelRayCurtain( SampleContext* context )
		: Sample( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 45.0f, 30.0f, 20.0f, b3Pos_zero );
		}

		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3BodyType::b3_kinematicBody;

		b3MassData massData = {};
		massData.mass = 0.0f;
		massData.center = b3Vec3_zero;
		massData.inertia = b3Mat3_identity;

		b3ShapeDef shapeDef = b3DefaultShapeDef();

		b3BodyId sphereBody = b3CreateBody( m_worldId, &bodyDef );
		m_sphereVoxels = createVoxelSphere( 9.0f );
		b3ShapeId sphereShape = b3CreateVoxelShape( sphereBody, &shapeDef, m_sphereVoxels, 0.1f );
		massData.center = b3MulSV( 0.1f, b3AABB_Center( b3Shape_GetVoxels( sphereShape ).data->bounds ) );
		b3Body_SetMassData( sphereBody, massData );
		b3Body_SetTransform( sphereBody, b3ToPos( b3Sub( { -6.0f, 3.0f, 0.0f }, massData.center ) ), b3Quat_identity );
		b3Body_SetAngularVelocity( sphereBody, { 0.8f, 0.4f, 0.8f } );

		b3BodyId capsuleBody = b3CreateBody( m_worldId, &bodyDef );
		m_capsuleVoxels = createVoxelCapsule( 10.0f, 8.0f );
		b3ShapeId capsuleShape = b3CreateVoxelShape( capsuleBody, &shapeDef, m_capsuleVoxels, 0.1f );
		massData.center = b3MulSV( 0.1f, b3AABB_Center( b3Shape_GetVoxels( capsuleShape ).data->bounds ) );
		b3Body_SetMassData( capsuleBody, massData );
		b3Body_SetTransform( capsuleBody, b3ToPos( b3Sub( { -2.0f, 3.0f, 0.0f }, massData.center ) ), b3Quat_identity );
		b3Body_SetAngularVelocity( capsuleBody, { 0.8f, 0.4f, 0.8f } );

		b3BodyId cubeBody = b3CreateBody( m_worldId, &bodyDef );
		m_cubeVoxels = createVoxelBox( 12.0f, 12.0f, 12.0f );
		b3ShapeId cubeShape = b3CreateVoxelShape( cubeBody, &shapeDef, m_cubeVoxels, 0.1f );
		massData.center = b3MulSV( 0.1f, b3AABB_Center( b3Shape_GetVoxels( cubeShape ).data->bounds ) );
		b3Body_SetMassData( cubeBody, massData );
		b3Body_SetTransform( cubeBody, b3ToPos( b3Sub( { 2.0f, 3.0f, 0.0f }, massData.center ) ), b3Quat_identity );
		b3Body_SetAngularVelocity( cubeBody, { 0.8f, 0.4f, 0.8f } );

		b3BodyId torusBody = b3CreateBody( m_worldId, &bodyDef );
		m_torusVoxels = createVoxelTorus( 6.0f, 4.0f );
		b3ShapeId torusShape = b3CreateVoxelShape( torusBody, &shapeDef, m_torusVoxels, 0.1f );
		massData.center = b3MulSV( 0.1f, b3AABB_Center( b3Shape_GetVoxels( torusShape ).data->bounds ) );
		b3Body_SetMassData( torusBody, massData );
		b3Body_SetTransform( torusBody, b3ToPos( b3Sub( { 6.0f, 3.0f, 0.0f }, massData.center ) ), b3Quat_identity );
		b3Body_SetAngularVelocity( torusBody, { 0.8f, 0.4f, 0.8f } );

		m_absSpeed = 0.015f;
		m_offset = 2.0f;
		m_speed = -m_absSpeed;
	}

	~VoxelRayCurtain() override
	{
		b3DestroyVoxels( m_sphereVoxels );
		b3DestroyVoxels( m_cubeVoxels );
		b3DestroyVoxels( m_capsuleVoxels );
		b3DestroyVoxels( m_torusVoxels );
	}

	void Render() override
	{
		Sample::Render();

		DrawGroundGrid( 10 );
		DrawLine( b3Pos_zero, b3OffsetPos( b3Pos_zero, 0.4f * b3Vec3_axisX ), MakeColor( b3_colorRed ) );
		DrawLine( b3Pos_zero, b3OffsetPos( b3Pos_zero, 0.4f * b3Vec3_axisY ), MakeColor( b3_colorGreen ) );
		DrawLine( b3Pos_zero, b3OffsetPos( b3Pos_zero, 0.4f * b3Vec3_axisZ ), MakeColor( b3_colorBlue ) );

		for ( float x = -8.0f; x <= 8.0f; x += 0.1f )
		{
			b3Pos rayOrigin = { x, 8.0f, m_offset };
			b3Pos rayEnd = { x, 0.0f, m_offset };
			b3Vec3 rayTranslation = b3SubPos( rayEnd, rayOrigin );

			b3RayResult result = b3World_CastRayClosest( m_worldId, rayOrigin, rayTranslation, b3DefaultQueryFilter() );
			if ( result.hit )
			{
				DrawLine( result.point, b3OffsetPos( result.point, 0.5f * result.normal ), MakeColor( b3_colorGreen ) );
				rayTranslation = b3MulSV( result.fraction, rayTranslation );
			}

			DrawPoint( rayOrigin, 4.0f, MakeColor( b3_colorGreen ) );
			DrawPoint( b3OffsetPos( rayOrigin, rayTranslation ), 4.0f, MakeColor( b3_colorRed ) );
			DrawLine( rayOrigin, b3OffsetPos( rayOrigin, rayTranslation ), MakeColor( b3_colorYellow ) );
		}

		if ( m_offset > 2.0f )
		{
			m_speed = -m_absSpeed;
		}
		else if ( m_offset < -2.0f )
		{
			m_speed = m_absSpeed;
		}

		if ( m_context->pause == false )
		{
			m_offset += m_speed;
		}
	}

	static Sample* Create( SampleContext* context )
	{
		return new VoxelRayCurtain( context );
	}

	float m_offset;
	float m_absSpeed;
	float m_speed;
	b3VoxelData* m_sphereVoxels = nullptr;
	b3VoxelData* m_capsuleVoxels = nullptr;
	b3VoxelData* m_cubeVoxels = nullptr;
	b3VoxelData* m_torusVoxels = nullptr;
};

static int sampleRayCastVoxels = RegisterSample( "Voxels", "Voxel Ray Curtain", VoxelRayCurtain::Create );

class VoxelSphereCollision : public Sample
{
public:
	explicit VoxelSphereCollision( SampleContext* context )
		: Sample( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 30.0f, 20.0f, 30.0f, b3Pos_zero );
		}

		AddGroundBox( 20.0f );

		b3BodyDef bodyDef = b3DefaultBodyDef();
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3Sphere sphere = { b3Vec3_zero, 1.0f };

		m_voxelsFlat = createVoxelBox( 50, 5, 50 );
		bodyDef.type = b3BodyType::b3_staticBody;
		bodyDef.position = b3Sub( { -10.0f, 1.0f, 0.0f }, getVoxelCentroid( m_voxelsFlat, 0.1f ) );
		bodyDef.rotation = b3Quat_identity;
		b3BodyId voxelBody1 = b3CreateBody( m_worldId, &bodyDef );
		b3CreateVoxelShape( voxelBody1, &shapeDef, m_voxelsFlat, 0.1f );
		b3Vec3 center = b3MulSV( 0.1f, b3AABB_Center( m_voxelsFlat->bounds ) );

		bodyDef.type = b3BodyType::b3_dynamicBody;
		bodyDef.position = { -10.0f, 3.0f, 0.0f };
		bodyDef.rotation = b3Quat_identity;
		b3BodyId sphereBody1 = b3CreateBody( m_worldId, &bodyDef );
		b3CreateSphereShape( sphereBody1, &shapeDef, &sphere );

		m_voxelsAngled = createVoxelBox( 50, 5, 100 );
		bodyDef.type = b3BodyType::b3_staticBody;
		bodyDef.position = b3Sub( { 0.0f, 2.0f, 0.0f }, getVoxelCentroid( m_voxelsAngled, 0.1f ) );
		bodyDef.rotation = b3MakeQuatFromAxisAngle( b3Vec3_axisX, 0.0625 * B3_PI );
		b3BodyId voxelBody2 = b3CreateBody( m_worldId, &bodyDef );
		b3CreateVoxelShape( voxelBody2, &shapeDef, m_voxelsAngled, 0.1f );

		bodyDef.type = b3BodyType::b3_dynamicBody;
		bodyDef.position = { 0.0f, 3.0f, 0.0f };
		bodyDef.rotation = b3Quat_identity;
		b3BodyId sphereBody2 = b3CreateBody( m_worldId, &bodyDef );
		b3CreateSphereShape( sphereBody2, &shapeDef, &sphere );

		m_voxelsTorus = createVoxelTorus( 10.0f, 6.0f );
		bodyDef.type = b3BodyType::b3_staticBody;
		b3Vec3 rotatedCenter =
			b3RotateVector( b3MakeQuatFromAxisAngle( b3Vec3_axisX, 0.5 * B3_PI ), getVoxelCentroid( m_voxelsTorus, 0.1f ) );
		bodyDef.position = b3Sub( { 10.0f, 1.0f, 0.0f }, rotatedCenter );
		bodyDef.rotation = b3MakeQuatFromAxisAngle( b3Vec3_axisX, 0.5 * B3_PI );
		b3BodyId voxelBody3 = b3CreateBody( m_worldId, &bodyDef );
		b3CreateVoxelShape( voxelBody3, &shapeDef, m_voxelsTorus, 0.1f );

		bodyDef.type = b3BodyType::b3_dynamicBody;
		bodyDef.position = { 10.0f, 3.0f, 0.0f };
		bodyDef.rotation = b3Quat_identity;
		b3BodyId sphereBody3 = b3CreateBody( m_worldId, &bodyDef );
		b3CreateSphereShape( sphereBody3, &shapeDef, &sphere );
	}

	~VoxelSphereCollision() override
	{
		b3DestroyVoxels( m_voxelsFlat );
		b3DestroyVoxels( m_voxelsAngled );
		b3DestroyVoxels( m_voxelsTorus );
	}

	static Sample* Create( SampleContext* context )
	{
		return new VoxelSphereCollision( context );
	}

private:
	b3VoxelData* m_voxelsFlat = nullptr;
	b3VoxelData* m_voxelsAngled = nullptr;
	b3VoxelData* m_voxelsTorus = nullptr;
};

static int sampleVoxelSphereCollision = RegisterSample( "Voxels", "Voxel Sphere Collision", VoxelSphereCollision::Create );
