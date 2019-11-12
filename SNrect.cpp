static char help[]	= "\n\n\n SN 2D Rectangle solve. \n\n\n";

/*==============================================================================

SOLVE 2D TRANSPORT EQUATION USING SOURCE ITERATION ALONE. THERE ARE
PRECONDITIONERS OR ACCELERATORS IN THIS CODE. WE WILL NOT EVEN FORM A MATRIX.

NOTE THAT WE DONT HAVE A PETSC IMPLEMENTATION OF QUADRATURE GENERATION SO
WE WILL GENERATE QUADRATURE IN MATLAB AND FEED IT HERE AS A PARAMETER.

1) DEFINE PROBLEM PARAMTERES.
2) INITIALIZE VECTOR OBJECTS, MATRIX OBJECTS, SOLVER OBJECTS AND PC OBJECTS
3) SWEEP
4) ITERATE

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

#include <petscksp.h>
#include <iostream>
#include <cmath>
#include <fstream>
using namespace std;

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **args) {


/*==========================================================================
   Every process in PETSc must begin with a PetscInitialize()
==========================================================================*/

PetscInitialize(&argc,&args,(char*)0,help);

// DECLARE VARIABLES

/* VECTOR OBJECTS */

Vec af_local, src_local;

/* MATRIX OBJECTS */

Mat mat_local;

/* PROBLEM PARAMETERS */

PetscInt i, j, k, ii, jj, kk, o, oo, order = 4, angles = order*order, nx = 100, ny = 100, its = 0;
PetscInt nv = 4;

PetscScalar add = 0.0, zr = 0.0, one = 1.0, *af_extract;

PetscReal L = 100, B = 100, xt = 1, xs = 0.99*xt, xa = xt - xs, tol = 1e-10, eps = 1;
PetscReal norm = 0, norm_o = 1, spr, rho, dx = L/nx, dy = B/ny, ext_src = 1;

/* REAL ARRAYS */

PetscReal fluxc[nx][ny], fluxe[nv][nx][ny], fluxo[nv][nx][ny], q0[nv][nx][ny], x_nod[nx+1];
PetscReal y_nod[ny+1], x_ctr[nx], y_ctr[ny], psi[nv*angles][nx][ny]; 
PetscReal scat_local[4] = {0, 0, 0, 0}, uw_local[4] = {0, 0, 0, 0}; // local scatter and upwind
PetscReal psi1L, psi1B, psi2B, psi2R, psi3R, psi3T, psi4T, psi4L; // needed for upwinding


PetscReal M[nv][nv] = { {4, 2, 1, 2}, {2, 4, 2, 1}, {1, 2, 4, 2}, {2, 1, 2, 4} };
PetscReal Lx[nv][nv] = { {2, 2, 1, 1}, {-2, -2, -1, -1}, {-1, -1, -2, -2}, {1, 1, 2, 2} };
PetscReal Ly[nv][nv] = { {2, 1, 1, 2}, {1, 2, 2, 1}, {-1, -2, -2, -1}, {-2, -1, -1, -2} };

PetscReal Amu_pos[nv][nv] = { {0, 0, 0, 0}, {0, 2, 1, 0}, {0, 1, 2, 0}, {0, 0, 0, 0} };
PetscReal Amu_neg[nv][nv] = { {-2, 0, 0, -1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {-1, 0, 0, -2} };
PetscReal Aeta_pos[nv][nv] = { {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 2, 1}, {0, 0, 1, 2} }; 
PetscReal Aeta_neg[nv][nv] =  { {-2, -1, 0, 0}, {-1, -2, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0} };

PetscReal A[nv][nv]; // we dont initialize this because we will do that in loops and it will overwritten
//PetscReal B[nv][nv]; // use this to generate values that go into mat_local 

/* NOW WE DEFINE WEIGHTS AND ANGLES (these come from the matlab quadrature code) */

PetscReal wt[angles] = {0.273204556499861, 0.273204556499861, 0.512193606897589, 0.512193606897589, 0.273204556499861, 0.273204556499861, 0.512193606897589, 0.512193606897589, 0.273204556499861, 0.273204556499861, 0.512193606897589, 0.512193606897589, 0.273204556499861, 0.273204556499861, 0.512193606897589, 0.512193606897589};
PetscReal mu[angles] = {0.861136311594053, 0.861136311594053, 0.339981043584856, 0.339981043584856, -0.861136311594053, -0.861136311594053, -0.339981043584856, -0.339981043584856, -0.861136311594053, -0.861136311594053, -0.339981043584856, -0.339981043584856, 0.861136311594053, 0.861136311594053, 0.339981043584856, 0.339981043584856};
PetscReal eta[angles] = {0.469676450658365, 0.194546355789953, 0.868846143426105, 0.359887856222652, 0.469676450658365, 0.194546355789953, 0.868846143426105, 0.359887856222652, -0.469676450658365, -0.194546355789953, -0.868846143426105, -0.359887856222652, -0.469676450658365, -0.194546355789953, -0.868846143426105, -0.359887856222652};

// PetscReal wt[angles] = {0.039752353243,0.039752353243,0.039752353243,0.039752353243,0.087328828017,0.087328828017,0.087328828017,0.087328828017,0.123192311759,0.123192311759,0.123192311759,0.123192311759,0.142425588680,0.142425588680,0.142425588680,0.142425588680,0.039752353243,0.039752353243,0.039752353243,0.039752353243,0.087328828017,0.087328828017,0.087328828017,0.087328828017,0.123192311759,0.123192311759,0.123192311759,0.123192311759,0.142425588680,0.142425588680,0.142425588680,0.142425588680,0.039752353243,0.039752353243,0.039752353243,0.039752353243,0.087328828017,0.087328828017,0.087328828017,0.087328828017,0.123192311759,0.123192311759,0.123192311759,0.123192311759,0.142425588680,0.142425588680,0.142425588680,0.142425588680,0.039752353243,0.039752353243,0.039752353243,0.039752353243,0.087328828017,0.087328828017,0.087328828017,0.087328828017,0.123192311759,0.123192311759,0.123192311759,0.123192311759,0.142425588680,0.142425588680,0.142425588680,0.142425588680};
// PetscReal mu[angles] = {0.960289856498,0.960289856498,0.960289856498,0.960289856498,0.796666477414,0.796666477414,0.796666477414,0.796666477414,0.525532409916,0.525532409916,0.525532409916,0.525532409916,0.183434642496,0.183434642496,0.183434642496,0.183434642496,-0.960289856498,-0.960289856498,-0.960289856498,-0.960289856498,-0.796666477414,-0.796666477414,-0.796666477414,-0.796666477414,-0.525532409916,-0.525532409916,-0.525532409916,-0.525532409916,-0.183434642496,-0.183434642496,-0.183434642496,-0.183434642496,-0.960289856498,-0.960289856498,-0.960289856498,-0.960289856498,-0.796666477414,-0.796666477414,-0.796666477414,-0.796666477414,-0.525532409916,-0.525532409916,-0.525532409916,-0.525532409916,-0.183434642496,-0.183434642496,-0.183434642496,-0.183434642496,0.960289856498,0.960289856498,0.960289856498,0.960289856498,0.796666477414,0.796666477414,0.796666477414,0.796666477414,0.525532409916,0.525532409916,0.525532409916,0.525532409916,0.183434642496,0.183434642496,0.183434642496,0.183434642496};
// PetscReal eta[angles] = {0.273643296705,0.231983585365,0.155006476088,0.054431035965,0.592805417586,0.502556166553,0.335797294845,0.117916329007,0.834426205201,0.707392379551,0.472664476643,0.165977691880,0.964143225427,0.817361159335,0.546143266133,0.191780011463,0.273643296705,0.231983585365,0.155006476088,0.054431035965,0.592805417586,0.502556166553,0.335797294845,0.117916329007,0.834426205201,0.707392379551,0.472664476643,0.165977691880,0.964143225427,0.817361159335,0.546143266133,0.191780011463,-0.273643296705,-0.231983585365,-0.155006476088,-0.054431035965,-0.592805417586,-0.502556166553,-0.335797294845,-0.117916329007,-0.834426205201,-0.707392379551,-0.472664476643,-0.165977691880,-0.964143225427,-0.817361159335,-0.546143266133,-0.191780011463,-0.273643296705,-0.231983585365,-0.155006476088,-0.054431035965,-0.592805417586,-0.502556166553,-0.335797294845,-0.117916329007,-0.834426205201,-0.707392379551,-0.472664476643,-0.165977691880,-0.964143225427,-0.817361159335,-0.546143266133,-0.191780011463};

/* SOLVER AND PRECONDITIONER CONTEXT */

KSP solver;
PC pc;

/* ERROR CHECKING */

PetscErrorCode ierr;

/* ALLOCATE NUMBERS TO ARRAYS */

x_nod[0] = 0.0;
y_nod[0] = 0.0;


for (i = 0; i < nx; i++) {

	x_nod[i+1] = x_nod[i] + dx;
	x_ctr[i] = (x_nod[i] + x_nod[i+1])/2;
	


	for(j = 0; j < ny; j++) {

		fluxc[i][j] = 0;

		y_nod[j+1] = y_nod[j] + dy;
		y_ctr[j] = (y_nod[j] + y_nod[j+1])/2;

		for(k = 0; k < nv; k++) {

			fluxe[k][i][j] = 0;
			fluxo[k][i][j] = 0;
			q0[k][i][j] = dx*dy*ext_src/(16*M_PI);
			
		}
	}
}

for(i = 0; i < nv; i++){

	for(j=0; j < nv; j++){
		
		// we initialized the array but did not multiply by leading factors (except angles) so we do it here

		M[i][j] = M[i][j]*dx*dy/36; 
		Lx[i][j] = Lx[i][j]*dy/12;
		Ly[i][j] = Ly[i][j]*dx/12;

		Amu_pos[i][j] = Amu_pos[i][j]*dy/6;
		Amu_neg[i][j] = Amu_neg[i][j]*dy/6;
		Aeta_pos[i][j] = Aeta_pos[i][j]*dx/6;
		Aeta_neg[i][j] = Aeta_neg[i][j]*dx/6;


	}

}


for(i = 0; i < nx; i++){

	for(j = 0; j < ny; j++) {

		for(k = 0; k < nv*angles; k++){

			psi[k][i][j] = 0;
		}
	}
}


/* NOW WE INITIALIZE LOCAL MATRIX, SOURCE, SOLUTION, SOLVER AND PRECONDITIONER */

// vecs

ierr = VecCreateSeq(PETSC_COMM_SELF, 4, &af_local); CHKERRQ(ierr);
ierr = VecDuplicate(af_local, &src_local); CHKERRQ(ierr);

//mat

ierr = MatCreate(PETSC_COMM_SELF, &mat_local); CHKERRQ(ierr);
ierr = MatSetSizes(mat_local, 4, 4, 4, 4); CHKERRQ(ierr);
ierr = MatSetFromOptions(mat_local); CHKERRQ(ierr);
ierr = MatSetUp(mat_local); CHKERRQ(ierr);

// solver

ierr = KSPCreate(PETSC_COMM_SELF, &solver); CHKERRQ(ierr);
ierr = KSPSetType(solver, KSPPREONLY); CHKERRQ(ierr);
ierr = KSPGetPC(solver, &pc); CHKERRQ(ierr);
ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);
ierr = KSPSetFromOptions(solver); CHKERRQ(ierr);

cout << " \n\n\n Begin source iteraition \n\n\n";

while(eps > tol){


	// cout << " \n\n\n Begin source iteraition - POS MU POS ETA \n\n\n";

	// angular loop

	for (o = 0; o < angles; o++)
	{

		if (mu[o] > 0) // POSITIVE MU - L TO R SWEEP
		{	
			// First define the A matrix outside loop

			for (i = 0; i < nv; i++)
			{
				for (j = 0; j < nv; j++)
				{
					A[i][j] = xt*M[i][j] + mu[o]*Lx[i][j];
				}
			}
			
			psi2R = 0.0;
			psi3R = 0.0;

			if (eta[o] > 0) // POSITIVE ETA AND MU - 1ST OCTANT - B TO T SWEEP
			{
				
				psi3T = 0.0;
				psi4T = 0.0;

				for (j = 0; j < ny; j++) 
				{
					for (i = 0; i < nx; i++)
					{
						// upwind terms

						if(j > 0){

							psi1B = psi[4*o + 3][i][j-1];
							psi2B = psi[4*o + 2][i][j-1];

						}
						else{

							psi1B = 0;
							psi2B = 0;
						}

						if(i > 0){

							psi1L = psi[4*o+1][i-1][j];
							psi4L = psi[4*o+2][i-1][j];

						}
						else{

							psi1L = 0;
							psi4L = 0;

						}

						// Now generate B = A + eta*Ly + mu*Amu_pos + eta*Aeta_pos - mat_local object

						for(ii = 0; ii < nv; ii++){

							for(jj = 0; jj < nv; jj++){

								add = A[ii][jj] + eta[o]*Ly[ii][jj] + mu[o]*Amu_pos[ii][jj] + eta[o]*Aeta_pos[ii][jj];
								ierr = MatSetValue(mat_local, ii, jj, add, INSERT_VALUES); CHKERRQ(ierr);
							}

						}

						ierr = MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
						ierr = MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


						// MatView(mat_local, PETSC_VIEWER_STDOUT_SELF);
						// exit(0);

						// now generate the source vector q = q0 + scat + mu_upwind + eta_upwind

						
						// first generate scat_local

						for(ii = 0; ii < nv; ii++){

							add = 0; // this gets overwritten every iteration

							for(jj = 0; jj < nv; jj++){

								add = add + M[ii][jj]*fluxe[jj][i][j];

							}

							scat_local[ii] = (xs/(4*M_PI))*add;

						}

						// now generate upwind_local

						uw_local[0] = (mu[o]*dy/6)*(2*psi1L + psi4L) + (eta[o]*dx/6)*(2*psi1B + psi2B);
						uw_local[1] = 0 + (eta[o]*dx/6)*(psi1B + 2*psi2B);
						uw_local[2] = 0 + 0;
						uw_local[3] = (mu[o]*dy/6)*(psi1L + 2*psi4L) + 0;

						// We already have the q0 array so now we just generate src_local now

						add = 0;  // probably don't need it here but keep it for safety

						for(kk = 0; kk < nv; kk++){

							add = scat_local[kk] + q0[kk][i][j] + uw_local[kk];
							ierr = VecSetValue(src_local, kk, add, INSERT_VALUES); CHKERRQ(ierr);

						}


						// now assemble matrix and vector

						ierr = VecAssemblyBegin(src_local); CHKERRQ(ierr);
						ierr = VecAssemblyEnd(src_local); CHKERRQ(ierr);


						// now setup operators and solve system

						ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
						ierr = KSPSolve(solver, src_local, af_local); CHKERRQ(ierr);

						// now that we have solved local system for i,j,o we extract soln and assign to ang_flux array

						ierr = VecGetArray(af_local, &af_extract); CHKERRQ(ierr);
						
						psi[4*o][i][j] = af_extract[0];
						psi[4*o + 1][i][j] = af_extract[1];
						psi[4*o + 2][i][j] = af_extract[2];
						psi[4*o + 3][i][j] = af_extract[3];

						ierr = VecRestoreArray(af_local, &af_extract); CHKERRQ(ierr);

					} // ENOD FOR LOOP OVER ELEMENT

				}

			}


			// cout << " \n\n\n Begin source iteraition - POS MU NEG ETA \n\n\n";


			if (eta[o] < 0) // NEGATIVE ETA, POSITIVE MU - 2ND OCTANT - T TO B SWEEP
			{
				
				
				psi1B = 0.0;
				psi2B = 0.0;

				for (j = ny-1; j >= 0; j--) 
				{
					for (i = 0; i < nx; i++)
					{
						// upwind terms

						if(j < ny-1){

							psi3T = psi[4*o + 1][i][j + 1];
							psi4T = psi[4*o + 0][i][j + 1];

						}
						else{

							psi3T = 0;
							psi4T = 0;
						}

						if(i > 0){

							psi1L = psi[4*o+1][i-1][j];
							psi4L = psi[4*o+2][i-1][j];

						}
						else{

							psi1L = 0;
							psi4L = 0;

						}

						// Now generate B = A + eta*Ly + mu*Amu_pos + eta*Aeta_pos - mat_local object

						for(ii = 0; ii < nv; ii++){

							for(jj = 0; jj < nv; jj++){

								add = A[ii][jj] + eta[o]*Ly[ii][jj] + mu[o]*Amu_pos[ii][jj] + eta[o]*Aeta_neg[ii][jj];
								ierr = MatSetValue(mat_local, ii, jj, add, INSERT_VALUES); CHKERRQ(ierr);
							}

						}


						// now generate the source vector q = q0 + scat + mu_upwind + eta_upwind

						
						// first generate scat_local

						for(ii = 0; ii < nv; ii++){

							add = 0; // this gets overwritten every iteration

							for(jj = 0; jj < nv; jj++){

								add = add + M[ii][jj]*fluxe[jj][i][j];

							}

							scat_local[ii] = (xs/(4*M_PI))*add;

						}

						// now generate upwind_local

						uw_local[0] = (mu[o]*dy/6)*(2*psi1L + psi4L) - 0;
						uw_local[1] = 0 - 0;
						uw_local[2] = 0 - (eta[o]*dx/6)*(2*psi3T + psi4T);
						uw_local[3] = (mu[o]*dy/6)*(psi1L + 2*psi4L) - (eta[o]*dx/6)*(psi3T + 2*psi4T);

						// We already have the q0 array so now we just generate src_local now

						add = 0;  // probably don't need it here but keep it for safety

						for(kk = 0; kk < nv; kk++){

							add = scat_local[kk] + q0[kk][i][j] + uw_local[kk];
							ierr = VecSetValue(src_local, kk, add, INSERT_VALUES); CHKERRQ(ierr);

						}


						// now assemble matrix and vector

						ierr = VecAssemblyBegin(src_local); CHKERRQ(ierr);
						ierr = VecAssemblyEnd(src_local); CHKERRQ(ierr);
						ierr = MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
						ierr = MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

						// now setup operators and solve system

						ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
						ierr = KSPSolve(solver, src_local, af_local); CHKERRQ(ierr);

						// now that we have solved local system for i,j,o we extract soln and assign to ang_flux array

						ierr = VecGetArray(af_local, &af_extract); CHKERRQ(ierr);
						
						psi[4*o][i][j] = af_extract[0];
						psi[4*o + 1][i][j] = af_extract[1];
						psi[4*o + 2][i][j] = af_extract[2];
						psi[4*o + 3][i][j] = af_extract[3];

						ierr = VecRestoreArray(af_local, &af_extract); CHKERRQ(ierr);

					} // ENOD FOR LOOP OVER ELEMENT
				}
			}


		} // END OF L TO R SWEEP - POSITIVE MU


		// cout << " \n\n\n Begin source iteraition - NEG MU POS ETA \n\n\n";



		if (mu[o] < 0) // NEGATIVE MU - R TO L SWEEP	
		{


			// First define the A matrix outside loop

			for (i = 0; i < nv; i++)
			{
				for (j = 0; j < nv; j++)
				{
					A[i][j] = xt*M[i][j] + mu[o]*Lx[i][j];
				}
			}

			psi1L = 0.0;
			psi4L = 0.0;

			
			if (eta[o] > 0) // POSITIVE ETA AND NEGATIVE MU - 3RD OCTANT - B TO T SWEEP
			{
				
				psi3T = 0.0;
				psi4T = 0.0;

				for (j = 0; j < ny; j++) 
				{
					for (i = nx-1; i >= 0; i--)
					{
						// upwind terms

						if(j > 0){

							psi1B = psi[4*o + 3][i][j-1];
							psi2B = psi[4*o + 2][i][j-1];

						}
						else{

							psi1B = 0;
							psi2B = 0;
						}

						if(i < nx-1){

							psi2R = psi[4*o + 0][i+1][j];
							psi3R = psi[4*o + 3][i+1][j];

						}
						else{

							psi2R = 0;
							psi3R = 0;

						}

						// Now generate B = A + eta*Ly + mu*Amu_pos + eta*Aeta_pos - mat_local object

						for(ii = 0; ii < nv; ii++){

							for(jj = 0; jj < nv; jj++){

								add = A[ii][jj] + eta[o]*Ly[ii][jj] + mu[o]*Amu_neg[ii][jj] + eta[o]*Aeta_pos[ii][jj];
								ierr = MatSetValue(mat_local, ii, jj, add, INSERT_VALUES); CHKERRQ(ierr);
							}

						}


						// now generate the source vector q = q0 + scat + mu_upwind + eta_upwind

						
						// first generate scat_local

						for(ii = 0; ii < nv; ii++){

							add = 0; // this gets overwritten every iteration

							for(jj = 0; jj < nv; jj++){

								add = add + M[ii][jj]*fluxe[jj][i][j];

							}

							scat_local[ii] = (xs/(4*M_PI))*add;


						}

						// now generate upwind_local

						uw_local[0] = -0 + (eta[o]*dx/6)*(2*psi1B + psi2B);
						uw_local[1] = -(mu[o]*dy/6)*(2*psi2R + psi3R) + (eta[o]*dx/6)*(psi1B + 2*psi2B);
						uw_local[2] = -(mu[o]*dy/6)*(psi2R + 2*psi3R) + 0;
						uw_local[3] = -0 + 0;

						// We already have the q0 array so now we just generate src_local now

						add = 0;  // probably don't need it here but keep it for safety

						for(kk = 0; kk < nv; kk++){

							add = scat_local[kk] + q0[kk][i][j] + uw_local[kk];
							ierr = VecSetValue(src_local, kk, add, INSERT_VALUES); CHKERRQ(ierr);

						}


						// now assemble matrix and vector

						ierr = VecAssemblyBegin(src_local); CHKERRQ(ierr);
						ierr = VecAssemblyEnd(src_local); CHKERRQ(ierr);
						ierr = MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
						ierr = MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

						// now setup operators and solve system

						ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
						ierr = KSPSolve(solver, src_local, af_local); CHKERRQ(ierr);

						// now that we have solved local system for i,j,o we extract soln and assign to ang_flux array

						ierr = VecGetArray(af_local, &af_extract); CHKERRQ(ierr);
						
						psi[4*o][i][j] = af_extract[0];
						psi[4*o + 1][i][j] = af_extract[1];
						psi[4*o + 2][i][j] = af_extract[2];
						psi[4*o + 3][i][j] = af_extract[3];

						ierr = VecRestoreArray(af_local, &af_extract); CHKERRQ(ierr);

					} // ENOD FOR LOOP OVER ELEMENT

				}

			}


			// cout << " \n\n\n Begin source iteraition - NEG MU NEG ETA \n\n\n";


			if (eta[o] < 0) // NEGATIVE ETA, NEGATIVE MU - 4TH OCTANT - T TO B SWEEP
			{
				

				psi1B = 0.0;
				psi2B = 0.0;

				for (j = ny-1; j >= 0; j--) 
				{
					for (i = nx-1; i >= 0; i--)
					{
						// upwind terms

						if(j < ny-1){

							psi3T = psi[4*o + 1][i][j + 1];
							psi4T = psi[4*o + 0][i][j + 1];

						}
						else{

							psi3T = 0;
							psi4T = 0;
						}

						if(i < nx-1){

							psi2R = psi[4*o + 0][i+1][j];
							psi3R = psi[4*o + 3][i+1][j];

						}
						else{

							psi2R = 0;
							psi3R = 0;

						}

						// Now generate B = A + eta*Ly + mu*Amu_pos + eta*Aeta_pos - mat_local object

						for(ii = 0; ii < nv; ii++){

							for(jj = 0; jj < nv; jj++){

								add = A[ii][jj] + eta[o]*Ly[ii][jj] + mu[o]*Amu_neg[ii][jj] + eta[o]*Aeta_neg[ii][jj];
								ierr = MatSetValue(mat_local, ii, jj, add, INSERT_VALUES); CHKERRQ(ierr);
							}

						}


						// now generate the source vector q = q0 + scat + mu_upwind + eta_upwind

						
						// first generate scat_local

						for(ii = 0; ii < nv; ii++){

							add = 0; // this gets overwritten every iteration

							for(jj = 0; jj < nv; jj++){

								add = add + M[ii][jj]*fluxe[jj][i][j];

							}

							scat_local[ii] = (xs/(4*M_PI))*add;

						}

						// now generate upwind_local

						uw_local[0] = 0;
						uw_local[1] = -(mu[o]*dy/6)*(2*psi2R + psi3R);
						uw_local[2] = -(mu[o]*dy/6)*(psi2R + 2*psi3R) - (eta[o]*dx/6)*(2*psi3T + psi4T);
						uw_local[3] = -(eta[o]*dx/6)*(psi3T + 2*psi4T);

						// We already have the q0 array so now we just generate src_local now

						add = 0;  // probably don't need it here but keep it for safety

						for(kk = 0; kk < nv; kk++){

							add = scat_local[kk] + q0[kk][i][j] + uw_local[kk];
							ierr = VecSetValue(src_local, kk, add, INSERT_VALUES); CHKERRQ(ierr);

						}


						// now assemble matrix and vector

						ierr = VecAssemblyBegin(src_local); CHKERRQ(ierr);
						ierr = VecAssemblyEnd(src_local); CHKERRQ(ierr);
						ierr = MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
						ierr = MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

						// now setup operators and solve system

						ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
						ierr = KSPSolve(solver, src_local, af_local); CHKERRQ(ierr);

						// now that we have solved local system for i,j,o we extract soln and assign to ang_flux array

						ierr = VecGetArray(af_local, &af_extract); CHKERRQ(ierr);
						
						psi[4*o][i][j] = af_extract[0];
						psi[4*o + 1][i][j] = af_extract[1];
						psi[4*o + 2][i][j] = af_extract[2];
						psi[4*o + 3][i][j] = af_extract[3];

						ierr = VecRestoreArray(af_local, &af_extract); CHKERRQ(ierr);

					} 

				} // END FOR LOOP OVER ELEMENT

			}


		} // END OF SWEEPS FOR NEGATIVE ANGLES - R TO L SWEEPS



	} // END OF SWEEPS OVER ALL ANGLES


	// cout << " \n\n\n " << its << "SWEEP \n\n\n";


	/* NOW THAT WE HAVE COMPLETED SWEEP FROM ALL 4 DIRECTIONS, WE WILL CALCULATE SCALAR FLUX
	   AND CHECK FOR CONVERGENCE */

	// first assign 0s to fluxe

	for(i = 0; i < nx; i++){

		for(j = 0; j < ny; j++) {

			for(k = 0; k < nv; k++){

				fluxe[k][i][j] = 0;
			}
		}
	}



	for (ii = 0; ii < nx; ii++){

		for (jj = 0; jj < ny; jj++){

			for (oo = 0; oo < angles; oo++){

				fluxe[0][ii][jj] = fluxe[0][ii][jj] + 2*wt[oo]*psi[4*oo][ii][jj];
				fluxe[1][ii][jj] = fluxe[1][ii][jj] + 2*wt[oo]*psi[4*oo + 1][ii][jj];
				fluxe[2][ii][jj] = fluxe[2][ii][jj] + 2*wt[oo]*psi[4*oo + 2][ii][jj];
				fluxe[3][ii][jj] = fluxe[3][ii][jj] + 2*wt[oo]*psi[4*oo + 3][ii][jj];
				
			}

		}
	}


	// Now calculate the norm 

	norm = 0;
	norm_o = 0;

	for (ii = 0; ii < nx; ii++){

		for (jj = 0; jj < ny; jj++){

			for (kk = 0; kk < nv; kk++){

				norm = norm + pow(fluxe[kk][ii][jj] - fluxo[kk][ii][jj], 2);
				norm_o = norm_o + pow(fluxo[kk][ii][jj], 2);
			}

		}
	}

	norm = sqrt(norm/(nx*ny));
	norm_o = sqrt(norm_o/(nx*ny));
	eps = norm/norm_o;
	cout << eps << "\n\n\n";

	// if(its > 1){

	// 	spr = norm/norm_o;
	// 	rho = 1 - spr;
	// 	eps = norm*(1-rho)/rho;
	// 	cout << eps;
	// }

	// norm_o = norm;


	// RESET OLD FLUX

	for (ii = 0; ii < nx; ii++){

		for (jj = 0; jj < ny; jj++){

			for (kk = 0; kk < nv; kk++){

				fluxo[kk][ii][jj] = fluxe[kk][ii][jj];

			}

		}
	}


for(i = 0; i < nx; i++){

	for(j = 0; j < ny; j++) {

		for(k = 0; k < nv*angles; k++){

			psi[k][i][j] = 0;
		}
	}
}



	its = its+1;

} // AFTER CONVERGENCE END OF WHILE LOOP

cout << "\n\n\n Total iterations is" << its << "\n\n\n";
// CALCULATE FINAL CELL CENTERED SCALAR FLUX

ofstream SN2Dsolve;
SN2Dsolve.open ("SN2Dsolve.txt");

for(i = 0; i < nx; i++){

	for(j = 0; j < ny; j++){

		for(k = 0; k < nv; k++){

			fluxc[i][j] = fluxc[i][j] + fluxe[k][i][j]; 

		}
	

		fluxc[i][j] = fluxc[i][j]/4;
		SN2Dsolve << "		" << i << "," << j << "	, " << fluxc[i][j] << " ;\n ";
		//cout << "		" << i << "," << j << "		" << fluxc[i][j] << " \n ";
	}

}

SN2Dsolve.close();

//cout << fluxe[0][0][0] << "	,  " << fluxe[1][0][0] << "	,  "  << fluxe[2][0][0] << "	,  " << fluxe[3][0][0] << "	,  " << "\n\n";


VecDestroy(&af_local);
VecDestroy(&src_local);
MatDestroy(&mat_local);
KSPDestroy(&solver);



/*============================================================================
 Every process in PETSc must end with a PetscFinalize()
============================================================================*/

ierr = PetscFinalize();
return 0;

}