// SPDX-FileCopyrightText: 2026 Ethan Rutherford
// SPDX-License-Identifier: MIT

#include "gfx/draw.h"
#include "human.h"
#include "imgui.h"
#include "mesh_loader.h"
#include "sample.h"
#include "utils.h"

#include "box3d/box3d.h"

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
	b3VoxelDef voxels[10000];

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
	return b3CreateVoxels( &voxelsDef );
}

b3VoxelData* createVoxelCapsule( float length, float radius )
{
	b3VoxelDef voxels[10000];

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
	return b3CreateVoxels( &voxelsDef );
}

b3VoxelData* createVoxelCube( int width, int height, int depth )
{
	b3VoxelDef voxels[10000];

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
	return b3CreateVoxels( &voxelsDef );
}

b3VoxelData* createVoxelTorus( float majorRadius, float minorRadius )
{
	b3VoxelDef voxels[10000];

	float halfLength = majorRadius + minorRadius;
	float r2 = minorRadius * minorRadius;

	int count = 0;
	for ( float x = -halfLength + 0.5f; x < halfLength; x += 1.0f )
	{
		for ( float y = -halfLength + 0.5f; y < halfLength; y += 1.0f )
		{
			for ( float z = -minorRadius + 0.5f; z < minorRadius; z += 1.0f )
			{
				float dist2 = (sqrtf( x * x + y * y ) - majorRadius) * (sqrtf( x * x + y * y ) - majorRadius) + z * z;
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
	return b3CreateVoxels( &voxelsDef );
}

class StaticVoxels : public Sample
{
public:
	explicit StaticVoxels( SampleContext* context )
		: Sample( context )
	{
		if ( context->restart == false )
		{
			m_camera->SetView( 0.0f, 25.0f, 10.0f, b3Pos_zero );
		}

		AddGroundBox( 20.0f );

		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3BodyType::b3_staticBody;
		bodyDef.position = { 0.0f, 0.5f, 0.0f };
		b3BodyId voxelBody = b3CreateBody( m_worldId, &bodyDef );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		m_voxels = createBoxy();

		b3CreateVoxelShape( voxelBody, &shapeDef, m_voxels, 0.1f );
	}

	~StaticVoxels() override
	{
		b3DestroyVoxels( m_voxels );
	}

	static Sample* Create( SampleContext* context )
	{
		return new StaticVoxels( context );
	}

	b3VoxelData* m_voxels = nullptr;
};

static int sampleStaticVoxels = RegisterSample( "Voxels", "Static Voxels", StaticVoxels::Create );

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
		b3Body_SetTransform( sphereBody, b3ToPos( b3Sub( { -6.0f, 3.0f, 0.0f }, massData.center )), b3Quat_identity );
		b3Body_SetAngularVelocity( sphereBody, { 0.8f, 0.4f, 0.8f } );

		b3BodyId capsuleBody = b3CreateBody( m_worldId, &bodyDef );
		m_capsuleVoxels = createVoxelCapsule( 10.0f, 8.0f );
		b3ShapeId capsuleShape = b3CreateVoxelShape( capsuleBody, &shapeDef, m_capsuleVoxels, 0.1f );
		massData.center = b3MulSV( 0.1f, b3AABB_Center( b3Shape_GetVoxels( capsuleShape ).data->bounds ) );
		b3Body_SetMassData( capsuleBody, massData );
		b3Body_SetTransform( capsuleBody, b3ToPos( b3Sub( { -2.0f, 3.0f, 0.0f }, massData.center )), b3Quat_identity );
		b3Body_SetAngularVelocity( capsuleBody, { 0.8f, 0.4f, 0.8f } );

		b3BodyId cubeBody = b3CreateBody( m_worldId, &bodyDef );
		m_cubeVoxels = createVoxelCube( 12.0f, 12.0f, 12.0f );
		b3ShapeId cubeShape = b3CreateVoxelShape( cubeBody, &shapeDef, m_cubeVoxels, 0.1f );
		massData.center = b3MulSV( 0.1f, b3AABB_Center( b3Shape_GetVoxels( cubeShape ).data->bounds ) );
		b3Body_SetMassData( cubeBody, massData );
		b3Body_SetTransform( cubeBody, b3ToPos( b3Sub( { 2.0f, 3.0f, 0.0f }, massData.center )), b3Quat_identity );
		b3Body_SetAngularVelocity( cubeBody, { 0.8f, 0.4f, 0.8f } );

		b3BodyId torusBody = b3CreateBody( m_worldId, &bodyDef );
		m_torusVoxels = createVoxelTorus( 6.0f, 4.0f );
		b3ShapeId torusShape = b3CreateVoxelShape( torusBody, &shapeDef, m_torusVoxels, 0.1f );
		massData.center = b3MulSV( 0.1f, b3AABB_Center( b3Shape_GetVoxels( torusShape ).data->bounds ) );
		b3Body_SetMassData( torusBody, massData );
		b3Body_SetTransform( torusBody, b3ToPos( b3Sub( { 6.0f, 3.0f, 0.0f }, massData.center )), b3Quat_identity );
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
