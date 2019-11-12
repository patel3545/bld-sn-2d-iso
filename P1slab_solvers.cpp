static char help[] = "P1 Slab Solve.\n\n";

/* ==============================================================================================

SOLVE P1 EQUATIONS USING PETSC - USING DIFFERENT SOLVERS AND PRECONDITIONERS

1) GENERATE THE SOURCE VECTOR AND DISPLAY TO CHECK IT
2) GENERATE THE MATRIX AND DISPLAY TO CHECK IT
3) SOLVE THE MATRIX VECTOR SYSTEM
4) EXTRACT SCALAR FLUX AND GET CELL CENTERED VALUES
5) DISPLAY SPECTRUM OF P1 OPERATOR
6) SOLVE THE EQUATION SYSTEM USING A DIRECT SOLVER

==============================================================================*/

/* -----------------------------------------------------------------------------

First include "petscksp.h" so that we can use KSP solvers.
This will automatically include

petscsys.h 		for base petsc routines
petscvec.h 		for vectors - this is the header we need for this example right now
petscmat.h 		for matrices - this is the header we will need for making matrices
petscis.h 		for index sets - idk what this does right now
petscksp.h 		for using linear solvers - iterative or direct this must be included
petscviewer.h 		to view vectors matrices etc
petscpc.h 		for preconditioning

------------------------------------------------------------------------------*/
//#include <iostream>
#include <petscksp.h>
#include <petscdraw.h>
//using namespace std;

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **args)
{

		/*==========================================================================
		 Every process in PETSc must begin with a PetscInitialize()
		==========================================================================*/
		PetscInitialize(&argc,&args,(char*)0,help);

		// Declare all variables

		// VECTOR OBJECTS

		Vec soln, src;// For any system that solves one matrix, we only have two
									// vec objects - solution and source all others must be input
									// as arrays

		// MATRIX OBJECTS

		Mat SM;				// For any system that solves one matrix system we must only
									// generate one matrix object

		// PROBLEM PARAMETERS - non integer

		PetscScalar alpha = 0.25, beta = 0.5, add = 0.0, L = 4, xt = 2, xs = 0.99*xt;
		PetscScalar xa = xt-xs, zr = 0, one = 1.0;
		PetscScalar *moments;

		// INTEGERS

		PetscInt	i, j, its, n_elem = 50;

		// REAL SCAL

		PetscReal tol = 1e-10, norm, dx = L/n_elem;

		// REAL ARRAYS

		PetscReal fluxc[n_elem], sigma_t[n_elem], sigma_s[n_elem], sigma_a[n_elem];
		PetscReal q0[2*n_elem], q1[2*n_elem], x_nodal[n_elem+1], x_center[n_elem], deltax[n_elem];
		PetscReal rlpart[n_elem], cxpart[n_elem];

		// SOLVER CONTEXT AND PRECODITIONER CONTEXT

		 KSP ksp;
		 PC pc;

		 // ERROR CHECKING

		 PetscErrorCode ierr;

		 // Allocate numbers to real ARRAYS

		 for(i=0; i < n_elem; i++) {

			 fluxc[i] = 0;
			 sigma_t[i] = xt;
			 sigma_s[i] = xs;
			 sigma_a[i] = xa;
			 q0[2*i] = 1;
			 q0[2*i+1] = 1;
			 q1[2*i] = 0;
			 q1[2*i+1] = 0;
			 deltax[i] = dx;

		 }

		 // nodal x coordinates

		 x_nodal[0] = 0;
		 for(i = 1; i < n_elem+1; i++){

			 x_nodal[i] = x_nodal[i-1] + deltax[i];

		 }

		 // cell centers

		 for(i=0; i < n_elem; i++){

			 x_center[i] = (x_nodal[i] + x_nodal[i+1])/2;

		 }


		 // NOW WE GENERATE THE SOURCE VECTOR. THIS WILL BE A PETSC VEC OBJECT SINCE
		 // WE WANT TO USE INBUILT PETSC SOLVER

		 ierr = VecCreateSeq(PETSC_COMM_SELF, 4*n_elem, &soln); CHKERRQ(ierr);
		 ierr = VecDuplicate(soln, &src); CHKERRQ(ierr);

		 ierr = VecSet(soln, zr); CHKERRQ(ierr);

		 for(i = 0; i<n_elem; i++) {

			 j = 4*i;
			 add = deltax[i]*(q0[2*i]/3 + q0[2*i + 1]/6);
			 VecSetValue(src, j, add, INSERT_VALUES);

			 j = 4*i+1;
			 add = deltax[i]*(q1[2*i]/3 + q1[2*i + 1]/6);
			 VecSetValue(src, j, add, INSERT_VALUES);

			 j = 4*i+2;
			 add = deltax[i]*(q0[2*i]/6 + q0[2*i + 1]/3);
			 VecSetValue(src, j, add, INSERT_VALUES);

			 j=4*i+3;
			 add = deltax[i]*(q1[2*i]/6 + q1[2*i + 1]/3);
			 VecSetValue(src, j, add, INSERT_VALUES);

		}

		VecAssemblyBegin(src);
		VecAssemblyEnd(src);

		// OUTPUT SRC (to terminal) VECTOR AND VERIFY

		ierr = PetscPrintf(PETSC_COMM_SELF, "src vector is:: \n"); CHKERRQ(ierr);
		ierr = VecView(src, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);

		// NOW THAT ALL THE RELEVANT ARRAYS AND VECTOR OBJECTS HAVE BEEN SETUP
		// WE WILL GENERATE OUR MATRIX OBJECT
		// WE WILL EXTENSIVELY USE MatSetValue(SM, i, j, value, INSERT_VALUES)
		// THAT IS MatSetValue(Matrix_Name, Row_Index, Column_Index, Value, INSERT_VALUES)

		// First create the matrix SM

		ierr = MatCreate(PETSC_COMM_WORLD, &SM); CHKERRQ(ierr);
		ierr = MatSetSizes(SM, PETSC_DECIDE, PETSC_DECIDE, 4*n_elem, 4*n_elem); CHKERRQ(ierr);
		ierr = MatSetFromOptions(SM); CHKERRQ(ierr);
		ierr = MatSetUp(SM); CHKERRQ(ierr);

		for(i=0; i < n_elem; i++) {

			// DIAGONAL BLOCK

				// ROW 4I

				add = (alpha/(2*beta)) + (sigma_a[i]*deltax[i]/3);
				ierr = MatSetValue(SM, 4*i, 4*i, add, INSERT_VALUES); CHKERRQ(ierr);
				add = 0;
				ierr = MatSetValue(SM, 4*i, 4*i+1, add, INSERT_VALUES); CHKERRQ(ierr);
				add = sigma_a[i]*deltax[i]/6;
				ierr = MatSetValue(SM, 4*i, 4*i+2, add, INSERT_VALUES); CHKERRQ(ierr);
				add = 0.5;
				ierr = MatSetValue(SM, 4*i, 4*i+3, add, INSERT_VALUES); CHKERRQ(ierr);

				// ROW 4I+1

				add = 0;
				ierr = MatSetValue(SM, 4*i+1, 4*i, add, INSERT_VALUES); CHKERRQ(ierr);
				add = (beta/(2*alpha)) + (sigma_t[i]*deltax[i]);
				ierr = MatSetValue(SM, 4*i+1, 4*i+1, add, INSERT_VALUES); CHKERRQ(ierr);
				add = 0.5;
				ierr = MatSetValue(SM, 4*i+1, 4*i+2, add, INSERT_VALUES); CHKERRQ(ierr);
				add = sigma_t[i]*deltax[i]/2;
				ierr = MatSetValue(SM, 4*i+1, 4*i+3, add, INSERT_VALUES); CHKERRQ(ierr);

				// ROW 4I+2

				add = sigma_a[i]*deltax[i]/6;
				ierr = MatSetValue(SM, 4*i+2, 4*i, add, INSERT_VALUES); CHKERRQ(ierr);
				add = -0.5;
				ierr = MatSetValue(SM, 4*i+2, 4*i+1, add, INSERT_VALUES); CHKERRQ(ierr);
				add = (alpha/(2*beta)) + (sigma_a[i]*deltax[i]/3);
				ierr = MatSetValue(SM, 4*i+2, 4*i+2, add, INSERT_VALUES); CHKERRQ(ierr);
				add = 0;
				ierr = MatSetValue(SM, 4*i+2, 4*i+3, add, INSERT_VALUES); CHKERRQ(ierr);

				// ROW 4I+3

				add = -0.5;
				ierr = MatSetValue(SM, 4*i+3, 4*i, add, INSERT_VALUES); CHKERRQ(ierr);
				add = sigma_t[i]*deltax[i]/2;
				ierr = MatSetValue(SM, 4*i+3, 4*i+1, add, INSERT_VALUES); CHKERRQ(ierr);
				add = 0;
				ierr = MatSetValue(SM, 4*i+3, 4*i+2, add, INSERT_VALUES); CHKERRQ(ierr);
				add = (beta/(2*alpha)) + (sigma_t[i]*deltax[i]);
				ierr = MatSetValue(SM, 4*i+3, 4*i+3, add, INSERT_VALUES); CHKERRQ(ierr);




			// NONZERO ON BLOCK TO RIGHT

			if(i != n_elem-1) {


				// (i+1)th - CONTRIBUTION GOES TO ROWS 4I+2 AND 4I+3; COLS 4I+4 AND 4I+5

				add = -alpha/(2*beta);
				ierr = MatSetValue(SM, 4*i+2, 4*i+4, add, INSERT_VALUES); CHKERRQ(ierr);
				add = 0.5;
				ierr = MatSetValue(SM, 4*i+2, 4*i+5, add, INSERT_VALUES); CHKERRQ(ierr);
				add = 0.5;
				ierr = MatSetValue(SM, 4*i+3, 4*i+4, add, INSERT_VALUES); CHKERRQ(ierr);
				add = -beta/(2*alpha);
				ierr = MatSetValue(SM, 4*i+3, 4*i+5, add, INSERT_VALUES); CHKERRQ(ierr);

			}

			// NONZERO ON BLOCK TO LEFT

			if(i != 0) {

				// (i-1) th - CONTRIBUTIONS ARE TO 4I, 4I+1 ROWS AND 4I-2, 4I-1 COLS

				add = -alpha/(2*beta);
				ierr = MatSetValue(SM, 4*i, 4*i-2, add, INSERT_VALUES); CHKERRQ(ierr);
				add = -0.5;
				ierr = MatSetValue(SM, 4*i, 4*i-1, add, INSERT_VALUES); CHKERRQ(ierr);
				add = -0.5;
				ierr = MatSetValue(SM, 4*i+1, 4*i-2, add, INSERT_VALUES); CHKERRQ(ierr);
				add = -beta/(2*alpha);
				ierr = MatSetValue(SM, 4*i+1, 4*i-1, add, INSERT_VALUES); CHKERRQ(ierr);


				// (i+1)th - CONTRIBUTION GOES TO ROWS 4I+2 AND 4I+3; COLS 4I+4 AND 4I+5

				// NO CONTRIBUTION ----------------------------------------------------

			}

		}


		ierr = MatAssemblyBegin(SM, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
		ierr = MatAssemblyEnd(SM, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

//		ierr = PetscPrintf(PETSC_COMM_SELF, "SM Matrix is:: \n"); CHKERRQ(ierr);
//	ierr = MatView(SM, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//==============================================================================
//==============================================================================
//==============================================================================
		// Now that we have setup Matrix and vector objects, we solve the system of EQUATIONS

		// Create a linear solver context

		ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

		// Set Operators now

		ierr = KSPSetOperators(ksp,SM, SM); CHKERRQ(ierr);

		// SET SOLVER TYPE 1) KSPRICHARDSON 2) KSPGMRES 3) KSPPREONLY

		ierr = KSPSetType(ksp, KSPPREONLY); CHKERRQ(ierr);

		// Get SOLVER AND PC

		ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);

		// NOW SET PC TYPE 1) PCJACOBI 2) PCNONE 3) PCILU 4)PCLU

			ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);

		// NOW SET TOLERANCES

		ierr = KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);

		// NOW SETUP THE SOLVER

		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		// NOW ANALYSE SPECTRUM OF LHS_OPERATOR

 //		ierr = KSPSetComputeEigenvalues(ksp, PETSC_TRUE); CHKERRQ(ierr);
	// 	ierr = KSPComputeEigenvalues(ksp, n_elem, &rlpart, &cxpart, n_elem); CHKERRQ(ierr);
	//	ierr = PetscDrawSP*(); CHKERRQ(ierr);

		// Now solve the linear system KSPSolve(context, rhsvec, soln)

		ierr = KSPSolve(ksp,src,soln);CHKERRQ(ierr);

		// get number of iterations only when using iterative solve

//		ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
//==============================================================================
//==============================================================================
//==============================================================================
		// Print Solution characteristics to terminal

		ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

		// Check Solution

		ierr = VecView(soln, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

		// Answer verified for n_elem=3. We must now extract the values from vector
		// object and get necessary values

		ierr = VecGetArray(soln, &moments); CHKERRQ(ierr);
	  ierr = PetscPrintf(PETSC_COMM_SELF, " \n \n \n", fluxc[i]); CHKERRQ(ierr);

		for (i = 0; i < n_elem; i++) {

			fluxc[i] = 0.5*(moments[4*i] + moments[4*i+2]);
    	ierr = PetscPrintf(PETSC_COMM_SELF, " %f \n", fluxc[i]); CHKERRQ(ierr);
		}
	//	PetscPrintf(PETSC_COMM_SELF, " %d \n", its); CHKERRQ(ierr);
		ierr = VecRestoreArray(soln, &moments); CHKERRQ(ierr);

		// Now destroy vecs and Mat

		VecDestroy(&soln);
		VecDestroy(&src);
		MatDestroy(&SM);
		KSPDestroy(&ksp);

	/*============================================================================
	 Every process in PETSc must end with a PetscFinalize()
	============================================================================*/

	ierr = PetscFinalize();
	return 0;


}
