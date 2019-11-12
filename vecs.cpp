static char help[] = "Creates vectors.\n\n";

/* PETSC VECTORS AND VECTOR OPERATIONS */

/* -----------------------------------------------------------------------------------------------

First include "petscksp.h" so that we can use KSP solvers. This will automatically include

petscsys.h 		for base petsc routines
petscvec.h 		for vectors - this is the header we need for this example right now
petscmat.h 		for matrices - this is the header we will need for making matrices
petscis.h 		for index sets - idk what this does right now
petscksp.h 		for using linear solvers - iterative or direct this must be included
petscviewer.h 		to view vectors matrices etc
petscpc.h 		for preconditioning

---------------------------------------------------------------------------------------------- */

#include <iostream>
#include <petscksp.h>
using namespace std;

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **args)
{

	Vec				x, b, u; 			// This is how petsc vectors are defined
	PetscScalar 	alpha = 2.0, beta = 1, add = 0;		// this is how scalars are defined
	PetscErrorCode	ierr;				// just keep it there for debugging in petsc
	PetscInt		i, n = 10; // this is integers in petsc

	// Every process in PETSc must begin with a PetscInitialize()
	PetscInitialize(&argc,&args,(char*)0,help);

	// 1) DEFINE VECTORS, COPY THEM, FILL THEM, AND PRINT THEM

	// Define a vector x

	ierr = VecCreateSeq(PETSC_COMM_SELF, n, &x); CHKERRQ(ierr);

	// Duplicate that vector

	ierr = VecDuplicate(x, &b); CHKERRQ(ierr);
	ierr = VecDuplicate(x, &u); CHKERRQ(ierr);

	// Now assign value alpha to x

	ierr = VecSet(x, alpha); CHKERRQ(ierr);
	ierr = VecSet(b, beta); CHKERRQ(ierr);

	// Now we set values for vector u

	for(i = 0; i < n; i++) {


	add = i+1;
	VecSetValues(u, 1, &i, &add, INSERT_VALUES);


	}

	VecAssemblyBegin(x);
	VecAssemblyEnd(x);


	// Now Print the vector x to terminal

	ierr = PetscPrintf(PETSC_COMM_WORLD, "x vector is:: \n"); CHKERRQ(ierr);
	ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD, " \n b vector is:: \n"); CHKERRQ(ierr);
	ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "\n u vector is:: \n");
	ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);

	// Now we want to print the vectors to a file called ex1_vecs.

	// don't know how to do this right now will think about this later.


	// Must destroy all the vectors that were created
	VecDestroy(&x);
	VecDestroy(&b);
	VecDestroy(&u);

	cout << "\n add is \n" << add;

	/* Always call PetscFinalize() before exiting a program*/

	ierr = PetscFinalize();
	return 0;


}
