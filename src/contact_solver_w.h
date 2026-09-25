#pragma once

#include "contact_solver.h"

void b3PrepareContacts_ConvexW4( b3SolverBlock block, b3StepContext* context );
void b3WarmStartContacts_ConvexW4( b3SolverBlock block, b3StepContext* context );
void b3PushContacts_ConvexW4( b3SolverBlock block, b3StepContext* context );
void b3SolveContacts_ConvexW4( b3SolverBlock block, b3StepContext* context );
void b3ApplyRestitution_ConvexW4( b3SolverBlock block, b3StepContext* context );
void b3StoreImpulses_ConvexW4( b3SolverBlock block, b3StepContext* context, int workerIndex );
int b3GetWideContactConstraintByteCountW4( void );

void b3PrepareContacts_ConvexW8( b3SolverBlock block, b3StepContext* context );
void b3WarmStartContacts_ConvexW8( b3SolverBlock block, b3StepContext* context );
void b3PushContacts_ConvexW8( b3SolverBlock block, b3StepContext* context );
void b3SolveContacts_ConvexW8( b3SolverBlock block, b3StepContext* context );
void b3ApplyRestitution_ConvexW8( b3SolverBlock block, b3StepContext* context );
void b3StoreImpulses_ConvexW8( b3SolverBlock block, b3StepContext* context, int workerIndex );
int b3GetWideContactConstraintByteCountW8( void );
