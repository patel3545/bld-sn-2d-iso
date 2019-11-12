/******************
 * NE 520 DFEM-P1 *
 ******************/
static char help[] = "Solves Linear Discontinous Finite Element P1 Transport Problem.\n\n";

#include <iostream>
#include <petscksp.h>

using namespace std;

#undef __FUNCT__
#define __FUNCT__ "main"


int main(int argc,char **args)
{

	Vec x, b, u; /* approx solution, RHS, exact solution */
	Mat A; /* linear system matrix */
	KSP ksp; /* linear solver context */
	PC pc; /* preconditioner context */

	PetscReal q[4];
	PetscErrorCode ierr;
	PetscInt i, j;
	PetscInt n = 50;
	PetscMPIInt size;
	PetscScalar one = 1.0;
	PetscScalar *solution, phi[n], current[n];
	PetscBool nonzeroguess = PETSC_FALSE;
	PetscInitialize(&argc,&args,(char*)0,help);

	PetscScalar alpha = 0.25, beta = 0.5;

	PetscReal sigma_t = 1.0, c = 0.99, sigma_a = 1.0-c*sigma_t;
	PetscReal xl = 0.0, xr = 5.0, dx = (xr-xl)/n;

	q[0] = 1.0;
	q[1] = 0.0;
	q[2] = 1.0;
	q[3] = 0.0;

	ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
	if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor example only!");

	ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetBool(NULL,"-nonzero_guess",&nonzeroguess,NULL);CHKERRQ(ierr);


	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		Compute the matrix and right-hand-side vector that define
		the linear system, Ax = b.
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	/*
		Create vectors. Note that we form 1 vector from scratch and
		then duplicate as needed.
	*/

	ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
	ierr = VecSetSizes(x, PETSC_DECIDE, 4*n);CHKERRQ(ierr);
	ierr = VecSetFromOptions(x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

	/*
		Create matrix. When using MatCreate(), the matrix format can
		be specified at runtime.
		Performance tuning note: For problems of substantial size,
		preallocation of matrix memory is crucial for attaining good
		performance. See the matrix chapter of the users manual for details.
	*/

	ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4*n,4*n);CHKERRQ(ierr);
	ierr = MatSetFromOptions(A);CHKERRQ(ierr);
	ierr = MatSetUp(A);CHKERRQ(ierr);

	/*
		Assemble matrix
	*/
	for (i=0; i<4*n; i++) {
		for(j=0; j<4*n; j++) {
			ierr = MatSetValue(A,i,j,0.0, INSERT_VALUES); CHKERRQ(ierr);
		}
	}

	ierr = VecSet(u, one); CHKERRQ(ierr);
	ierr = VecSet(x, one); CHKERRQ(ierr);

	// Block Diagonal
	for (i=0; i<n; i++) {

		// Source Vector
		ierr = VecSetValue(b, 4*i, dx/3.0*q[0] + dx/6.0*q[2], INSERT_VALUES); CHKERRQ(ierr);
		ierr = VecSetValue(b, 4*i+1, dx/3.0*q[1] + dx/6.0*q[3], INSERT_VALUES); CHKERRQ(ierr);
		ierr = VecSetValue(b, 4*i+2, dx/6.0*q[0] + dx/3.0*q[2], INSERT_VALUES); CHKERRQ(ierr);
		ierr = VecSetValue(b, 4*i+3, dx/6.0*q[1] + dx/3.0*q[3], INSERT_VALUES); CHKERRQ(ierr);


		// Block Diagonal Terms
		ierr = MatSetValue(A, 4*i,   4*i,   alpha/2.0/beta + sigma_a/3.0*dx, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i+1, beta/2.0/alpha + sigma_t*dx, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i+2, alpha/2.0/beta + sigma_a/3.0*dx, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i+3, beta/2.0/alpha + sigma_t*dx, INSERT_VALUES); CHKERRQ(ierr);

		// Inverse Diagonal Terms
		ierr = MatSetValue(A, 4*i+3, 4*i,   -0.5, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i+1, -0.5, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i+2,  0.5, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i,   4*i+3,  0.5, INSERT_VALUES); CHKERRQ(ierr);

		// Off Diagonal Terms
		ierr = MatSetValue(A, 4*i+1, 4*i,   0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i,   4*i+1, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i+3, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i+2, 0.0, INSERT_VALUES); CHKERRQ(ierr);

		ierr = MatSetValue(A, 4*i+2, 4*i,   sigma_a*dx/6.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i,   4*i+2, sigma_a*dx/6.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i+3, sigma_t*dx/2.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i+1, sigma_t*dx/2.0, INSERT_VALUES); CHKERRQ(ierr);

		if(i != 0 ) {
		// Pre-Main Block Diagonal Coupling
		ierr = MatSetValue(A, 4*i, 4*i-1, -0.5, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i, 4*i-2, -alpha/2.0/beta, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i, 4*i-3, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i, 4*i-4, 0.0, INSERT_VALUES); CHKERRQ(ierr);

		ierr = MatSetValue(A, 4*i+1, 4*i-1, -beta/2.0/alpha, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i-2, -0.5, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i-3, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i-4, 0.0, INSERT_VALUES); CHKERRQ(ierr);

		ierr = MatSetValue(A, 4*i+2, 4*i-1, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i-2, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i-3, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i-4, 0.0, INSERT_VALUES); CHKERRQ(ierr);

		ierr = MatSetValue(A, 4*i+3, 4*i-1, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i-2, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i-3, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i-4, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		}

		if(i != n-1) {
		// Post-Main Block Diagonal Coupling
		ierr = MatSetValue(A, 4*i, 4*i+4, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i, 4*i+5, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i, 4*i+6, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i, 4*i+7, 0.0, INSERT_VALUES); CHKERRQ(ierr);

		ierr = MatSetValue(A, 4*i+1, 4*i+4, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i+5, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i+6, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+1, 4*i+7, 0.0, INSERT_VALUES); CHKERRQ(ierr);

		ierr = MatSetValue(A, 4*i+2, 4*i+4, -alpha/2.0/beta, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i+5, 0.5, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i+6, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+2, 4*i+7, 0.0, INSERT_VALUES); CHKERRQ(ierr);

		ierr = MatSetValue(A, 4*i+3, 4*i+4, 0.5, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i+5, -beta/2.0/alpha, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i+6, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatSetValue(A, 4*i+3, 4*i+7, 0.0, INSERT_VALUES); CHKERRQ(ierr);
		}

	}

	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	//ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	//ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = VecCopy(b,x);CHKERRQ(ierr);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		Create the linear solver and set various options
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	/*
		Create linear solver context
	*/

	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

	/*
		Set operators. Here the matrix that defines the linear system
		also serves as the preconditioning matrix.
	*/

	ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

	/*
		Set linear solver defaults for this problem (optional).

		- By extracting the KSP and PC contexts from the KSP context,
		we can then directly call any KSP and PC routines to set
		various options.

		- The following four statements are optional; all of these
		parameters could alternatively be specified at runtime via
		KSPSetFromOptions();
	*/

	ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
	ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
	ierr = KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

	/*
		Set runtime options, e.g.,
		-ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
		These options will override those specified above as long as
		KSPSetFromOptions() is called _after_ any other customization
		routines.
	*/

	ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

	if (nonzeroguess) {
		PetscScalar p = 0.5;
		ierr = VecSet(x,p);CHKERRQ(ierr);
		ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
	}


	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		Solve the linear system
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	/*
		Solve linear system
	*/

	ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

	/*
		View solver info; we could instead use the option -ksp_view to
		print this info to the screen at the conclusion of KSPSolve().
	*/

//	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		Check solution and clean up
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

	/*
		Check the error
	*/
	//ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = VecGetArray(x, &solution); CHKERRQ(ierr);

	//cout<<endl<<"X\tPhi\t\tJ"<<endl;
	for(i = 0; i < n ; i++) {

		phi[i] = 0.5*(solution[4*i]+solution[4*i+2]);
		current[i] = 0.5*(solution[4*i+1]+solution[4*i+3]);
		cout<<dx/2.0+i*dx<<"\t"<<phi[i]<<"\t"<<current[i]<<endl;
	}
	ierr = VecRestoreArray(x, &solution); CHKERRQ(ierr);


	/*
		Free work space. All PETSc objects should be destroyed when they
		are no longer needed.
	*/

	ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
	ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
	ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

	/*
		Always call PetscFinalize() before exiting a program. This routine
		- finalizes the PETSc libraries as well as MPI
		- provides summary and diagnostic information if certain runtime
		options are chosen (e.g., -log_summary).
	*/

	ierr = PetscFinalize();

}
