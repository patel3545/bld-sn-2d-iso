static char help[] = "SN Slab Solve.\n\n";

/*==============================================================================

SOLVE SLAB GEOMETRY TRANSPORT EQUATION USING SOURCE ITERATION ALONE. THERE ARE
PRECONDITIONERS OR ACCELERATORS IN THIS CODE. WE WILL NOT EVEN FORM A MATRIX.
THIS CODE IS SIMPLY A TRANSLATION OF THE OLD MATLAB CODE INTO A PETSC CODE.

NOTE THAT WE DONT HAVE A PETSC IMPLEMENTATION OF QUADRATURE GENERATION SO
WE WILL GENERATE QUADRATURE IN MATLAB AND FEED IT HERE AS A PARAMETER.

1) DEFINE PROBLEM PARAMTERES.
2) INITIALIZE VECTOR OBJECTS, MATRIX OBJECTS, SOLVER OBJECTS AND PC OBJECTS
3) SWEEP
4) ITERATE

ALSO, NOTE THAT WE WILL BE USING THE INBUILD PETSC LU SOLVER TO SOLVE THE
2 BY 2 MATRIX OBTAINED FOR EACH CELL. WE WILL NOT BE USING AN EXTERNAL SOLVER
OR WRITE ONE. WE COULD ALSO USE RICHARDSON IF NEED BE. WE ARE SOLVING TINY
MATRICES SO IT SHOULD NOT BE EXPENSIVE WHICH METHOD WE USE

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

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char **args) {

/*==========================================================================
   Every process in PETSc must begin with a PetscInitialize()
==========================================================================*/

PetscInitialize(&argc,&args,(char*)0,help);

// DECLARE VARIABLES

/* VECTOR OBJECTS - 1) af_local 2) src_local - these two
 will keep getting overwritten for each cell and angle */

Vec af_local, src_local;
Vec norflx, norflx_o;

/* MATRIX OBJECTS - 1) mat_local */

Mat mat_local;

/* PROBLEM PARAMETERS */

PetscScalar L = 4, alpha = 0.25, beta = 0.5, add = 0.0, xt = 2, xs = 0.5*xt;
PetscScalar xa = xt - xs, zr = 0.0, one = 1.0, ext_src = 1.0;
PetscScalar *af_extract; /*  will use this to extract af_local before using it
                             to add to af matrix */

PetscInt i, o, j, order = 8, n_elem = 32, its = 0, index[2*n_elem];

PetscReal tol = 1e-6, eps = 1, norm, norm_o, dx = L/n_elem;

/* REAL ARRAYS */

PetscReal fluxc[n_elem], fluxe[2*n_elem], fluxe_o[2*n_elem], sigma_t[n_elem], sigma_s[n_elem], sigma_a[n_elem];
PetscReal q0[2*n_elem], q0dis[2*n_elem], x_nodal[n_elem+1], x_center[n_elem], deltax[n_elem];
PetscReal ang_flux[2*n_elem][order]; // angular flux storage matrix
PetscReal scat_local[2] = {0., 0.}; // local scattering source container
PetscReal upwind_local[2] = {0., 0.}; // local upwind term
PetscReal af_lb, af_rb; // needed for upwinding

PetscReal wt[8] = {0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362, 0.362683783378362, 0.3137066458778870, 0.222381034453374, 0.101228536290376};
PetscReal mu[8] = {-0.960289856497537, -0.796666477413627, -0.525532409916329, -0.183434642495650, 0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536};

/* DEFINE ANGLE AND WEIGHT WE WILL DO S8 FOR NOW */

/* SOLVER and PRECONDITIONER CONTEXT */

KSP solver;
PC pc;

/* ERROR CHECKING */

PetscErrorCode ierr;

/* ALLOCATE NUMBERS TO ARRAYS */

for(i = 0; i < n_elem; i++) {
  //flux
  fluxc[i] = 0;
  fluxe[2*i] = 0;
  fluxe[2*i+1] = 0;

  fluxe_o[2*i] = 0;
  fluxe_o[2*i+1] = 0;

  // xs
  sigma_t[i] = xt;
	sigma_s[i] = xs;
	sigma_a[i] = xa;

  // discretization
	deltax[i] = dx;

  // ext src
  q0[2*i] = ext_src;
  q0[2*i+1] = ext_src;
  q0dis[2*i] = q0[2*i]*deltax[i]/4;
  q0dis[2*i+1] = q0[2*i+1]*deltax[i]/4;

  // index
  index[2*i] = 2*i;
  index[2*i+1] = 2*i + 1;

}



// nodal x coordinates

x_nodal[0] = 0;

for(i = 1; i < n_elem+1; i++) {

  x_nodal[i] = x_nodal[i-1] + deltax[i];

}

// cell centers

for(i = 0; i < n_elem; i++) {

  x_center[i] = (x_nodal[i] + x_nodal[i+1])/2;

}

// NOW WE INITIALIZE LOCAL MATRIX, SOURCE AND SOLUTION, SOLVER and PRECONDITIONER

ierr = VecCreateSeq(PETSC_COMM_SELF, 2*n_elem, &norflx); CHKERRQ(ierr);
ierr = VecDuplicate(norflx, &norflx_o); CHKERRQ(ierr);

VecSet(norflx, zr);
VecSet(norflx_o, zr);

ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &af_local); CHKERRQ(ierr);
ierr = VecDuplicate(af_local, &src_local); CHKERRQ(ierr);

// VecSet(af_local, one);
// VecSet(src_local, one);
//
ierr = MatCreate(PETSC_COMM_SELF, &mat_local); CHKERRQ(ierr);
ierr = MatSetSizes(mat_local, 2, 2, 2, 2); CHKERRQ(ierr);
ierr = MatSetFromOptions(mat_local); CHKERRQ(ierr);
ierr = MatSetUp(mat_local); CHKERRQ(ierr);

ierr = KSPCreate(PETSC_COMM_SELF, &solver); CHKERRQ(ierr);
//ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
ierr = KSPSetType(solver, KSPGMRES); CHKERRQ(ierr);
ierr = KSPGetPC(solver, &pc); CHKERRQ(ierr);
ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
ierr = KSPSetFromOptions(solver); CHKERRQ(ierr);



while(eps > tol) {

  for (o = 0; o < order; o++) {

    if (mu[o] > 0) {

      // ierr = PetscPrintf(PETSC_COMM_SELF, " inside if \n\n\n\n"); CHKERRQ(ierr);

      // START WITH LEFT BOUNDARY AND 0TH CELL

      // first we generate the src_local vector
      // src_local = scat_local + q0dis (already calculated) + upwind_local

      scat_local[0] = (sigma_s[0]*deltax[0]/6)*fluxe[2*0] + (sigma_s[0]*deltax[0]/12)*fluxe[2*0 + 1];
      scat_local[1] = (sigma_s[0]*deltax[0]/12)*fluxe[2*0] + (sigma_s[0]*deltax[0]/6)*fluxe[2*0 + 1];

      upwind_local[0] = 0.;
      upwind_local[1] = 0.;

      // Now generate the src_local vec

      ierr = VecSetValue(src_local, 0, scat_local[0] + upwind_local[0] + q0dis[0], INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecSetValue(src_local, 1, scat_local[1] + upwind_local[1] + q0dis[1], INSERT_VALUES); CHKERRQ(ierr);

      // Now oing it out athat we have src_local, we generate mat_local
      add = (mu[o]/2) + (sigma_t[0]*deltax[0]/3);
      ierr = MatSetValue(mat_local, 0 , 0, add, INSERT_VALUES);

      add = (mu[o]/2) + (sigma_t[0]*deltax[0]/6);
      ierr = MatSetValue(mat_local, 0, 1, add, INSERT_VALUES);

      add = (-mu[o]/2) + (sigma_t[0]*deltax[0]/6);
      ierr = MatSetValue(mat_local, 1, 0, add, INSERT_VALUES);

      add = (mu[o]/2) + (sigma_t[0]*deltax[0]/3);
      ierr = MatSetValue(mat_local, 1, 1, add, INSERT_VALUES);

      // Assemble vector and matrix

      VecAssemblyBegin(src_local);
      VecAssemblyEnd(src_local);
      MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY);

      // ierr = VecView(src_local, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
      // ierr = MatView(mat_local, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);

      // now do matrix assembly here idk if needed here though - will do later if i get ERROR

      // SET UP OPERATORS

      // ierr = KSPCreate(PETSC_COMM_SELF, &solver); CHKERRQ(ierr);
      ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
      // ierr = KSPSetType(solver, KSPPREONLY); CHKERRQ(ierr);
      // ierr = KSPGetPC(solver, &pc); CHKERRQ(ierr);
      // ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);
      // ierr = KSPSetFromOptions(solver); CHKERRQ(ierr);
      ierr = KSPSolve(solver, src_local, af_local); CHKERRQ(ierr);

      // PetscPrintf(PETSC_COMM_SELF, "\n \n \n \n af_local \n\n");
      // ierr = VecView(af_local, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);



      // NOW EXTRACT THE SOLVE INTO REAL ARRAY AND ASSIGN THAT TO THE ang_flux ARRAY
      ierr = VecGetArray(af_local, &af_extract); CHKERRQ(ierr);
      ang_flux[0][o] = af_extract[0];
      ang_flux[1][o] = af_extract[1];
      af_lb = mu[o]*af_extract[1]; // for upwinding next cell
      ierr = VecRestoreArray(af_local, &af_extract); CHKERRQ(ierr);
      // ierr = KSPDestroy(&solver); CHKERRQ(ierr);



      // This constitutes solve for one cell. We now move on to interior cells

      for (i = 1; i < n_elem; i++) {

      //  PetscPrintf(PETSC_COMM_SELF, "\n\n inside for %d \n\n", i);

        // first we generate the src_local vector
        // src_local = scat_local + q0dis (already calculated) + upwind_local

        scat_local[0] = (sigma_s[i]*deltax[i]/6)*fluxe[2*i] + (sigma_s[i]*deltax[i]/12)*fluxe[2*i + 1];
        scat_local[1] = (sigma_s[i]*deltax[i]/12)*fluxe[2*i] + (sigma_s[i]*deltax[i]/6)*fluxe[2*i + 1];

        upwind_local[0] = af_lb;
        upwind_local[1] = 0.;

        // Now generate the src_local vec

        ierr = VecSetValue(src_local, 0, scat_local[0] + upwind_local[0] + q0dis[2*i], INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecSetValue(src_local, 1, scat_local[1] + upwind_local[1] + q0dis[2*i + 1], INSERT_VALUES); CHKERRQ(ierr);
        VecAssemblyBegin(src_local);
        VecAssemblyEnd(src_local);

        // PetscPrintf(PETSC_COMM_SELF, "\n\n new src vector cell %d angle %d \n\n", i, o);
        // VecView(src_local, PETSC_VIEWER_STDOUT_SELF);

        // Now that we have src_local, we generate mat_local
        add = (mu[o]/2) + (sigma_t[i]*deltax[i]/3);
        ierr = MatSetValue(mat_local, 0 , 0, add, INSERT_VALUES);

        add = (mu[o]/2) + (sigma_t[i]*deltax[i]/6);
        ierr = MatSetValue(mat_local, 0, 1, add, INSERT_VALUES);

        add = (-mu[o]/2) + (sigma_t[i]*deltax[i]/6);
        ierr = MatSetValue(mat_local, 1, 0, add, INSERT_VALUES);

        add = (mu[o]/2) + (sigma_t[i]*deltax[i]/3);
        ierr = MatSetValue(mat_local, 1, 1, add, INSERT_VALUES);

        // Assemble vector and matrix
         MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY);
         MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY);

        //  PetscPrintf(PETSC_COMM_SELF, "\n\n new mat cell %d \n\n", i);
        //  MatView(mat_local, PETSC_VIEWER_STDOUT_SELF);

        // now do matrix assembly here idk if needed here t MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY);
        //); CHKERRQ(ierr);

      //  ierr = KSPCreate(PETSC_COMM_SELF, &solver); CHKERRQ(ierr);
        ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
        // ierr = KSPSetType(solver, KSPPREONLY); CHKERRQ(ierr);
        // ierr = KSPGetPC(solver, &pc); CHKERRQ(ierr);
        // ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);
        // ierr = KSPSetFromOptions(solver); CHKERRQ(ierr);
        ierr = KSPSolve(solver, src_local, af_local); CHKERRQ(ierr);

        // if (o == 4) {
        //
        //   PetscPrintf(PETSC_COMM_SELF, "af_local \n\n");
        //   ierr = VecView(af_local, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
        //
        //
        // }



        // NOW EXTRACT THE SOLVE INTO REAL ARRAY AND ASSIGN THAT TO THE ang_flux ARRAY
        ierr = VecGetArray(af_local, &af_extract); CHKERRQ(ierr);
        ang_flux[2*i][o] = af_extract[0];
        ang_flux[2*i+1][o] = af_extract[1];
        af_lb = mu[o]*af_extract[1]; // for upwinding next cell
        ierr = VecRestoreArray(af_local, &af_extract); CHKERRQ(ierr);
      //   ierr = KSPDestroy(&solver); CHKERRQ(ierr);
    } // end for over elem
  } // end if

  // PetscPrintf(PETSC_COMM_SELF, "\n\n OUTSIDE POSITIVE", i);

  if (mu[o] < 0) {

  //  PetscPrintf(PETSC_COMM_SELF, "\n\n inside - o");
    // NOW WE SWEEP RIGHT TO LEFT

    // FIRST WE DO LAST CELL

    // src_local = scat_local + q0dis (already calculated) + upwind_local

    scat_local[0] = (sigma_s[n_elem-1]*deltax[n_elem-1]/6)*fluxe[2*(n_elem-1)] + (sigma_s[n_elem-1]*deltax[n_elem-1]/12)*fluxe[2*(n_elem-1) + 1];
    scat_local[1] = (sigma_s[n_elem-1]*deltax[n_elem-1]/12)*fluxe[2*(n_elem-1)] + (sigma_s[n_elem-1]*deltax[n_elem-1]/6)*fluxe[2*(n_elem-1) + 1];

    upwind_local[0] = 0.;
    upwind_local[1] = 0.;

    // Now generate the src_local vec

    ierr = VecSetValue(src_local, 0, scat_local[0] + upwind_local[0] + q0dis[2*(n_elem-1)], INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValue(src_local, 1, scat_local[1] + upwind_local[1] + q0dis[2*(n_elem-1) + 1], INSERT_VALUES); CHKERRQ(ierr);

    // Now that we have src_local, we generate mat_local
    add = (-mu[o]/2) + (sigma_t[n_elem-1]*deltax[n_elem-1]/3);
    ierr = MatSetValue(mat_local, 0 , 0, add, INSERT_VALUES);

    add = (mu[o]/2) + (sigma_t[n_elem-1]*deltax[n_elem-1]/6);
    ierr = MatSetValue(mat_local, 0, 1, add, INSERT_VALUES);

    add = (-mu[o]/2) + (sigma_t[n_elem-1]*deltax[n_elem-1]/6);
    ierr = MatSetValue(mat_local, 1, 0, add, INSERT_VALUES);

    add = (-mu[o]/2) + (sigma_t[n_elem-1]*deltax[n_elem-1]/3);
    ierr = MatSetValue(mat_local, 1, 1, add, INSERT_VALUES);

    // Assemble vector and matrix

     VecAssemblyBegin(src_local);
     VecAssemblyEnd(src_local);

  //   PetscPrintf(PETSC_COMM_SELF, " src local going neg");
  //   VecView(src_local, PETSC_VIEWER_STDOUT_SELF);

     MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY);

  //   PetscPrintf(PETSC_COMM_SELF, " mat local going neg");
  //   MatView(mat_local, PETSC_VIEWER_STDOUT_SELF);


    // now do matrix assembly here idk if needed here though - will do later if i get ERROR

    // SET UP OPERATORS

  //  ierr = KSPCreate(PETSC_COMM_SELF, &solver); CHKERRQ(ierr);
    ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
    // ierr = KSPSetType(solver, KSPPREONLY); CHKERRQ(ierr);
    // ierr = KSPGetPC(solver, &pc); CHKERRQ(ierr);
    // ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);
    // ierr = KSPSetFromOptions(solver); CHKERRQ(ierr);
    ierr = KSPSolve(solver, src_local, af_local); CHKERRQ(ierr);

  //  PetscPrintf(PETSC_COMM_SELF, "\n\n\n af local going neg");
  //  VecView(af_local, PETSC_VIEWER_STDOUT_SELF);


    // NOW EXTRACT THE SOLVE INTO REAL ARRAY AND ASSIGN THAT TO THE ang_flux ARRAY
    ierr = VecGetArray(af_local, &af_extract); CHKERRQ(ierr);
    ang_flux[2*(n_elem-1)][o] = af_extract[0];
    ang_flux[2*(n_elem-1)+1][o] = af_extract[1];
    af_rb = -mu[o]*af_extract[0]; // for upwinding with next cell
    ierr = VecRestoreArray(af_local, &af_extract); CHKERRQ(ierr);
  //  ierr = KSPDestroy(&solver); CHKERRQ(ierr);

    // Now do interior cells

    for (i = n_elem-2; i >= 0; i--) {

  //    PetscPrintf(PETSC_COMM_SELF, "\n\n inside - o %d %d", i, o);

      // first we generate the src_local vector
      // src_local = scat_local + q0dis (already calculated) + upwind_local

      scat_local[0] = (sigma_s[i]*deltax[i]/6)*fluxe[2*i] + (sigma_s[i]*deltax[i]/12)*fluxe[2*i + 1];
      scat_local[1] = (sigma_s[i]*deltax[i]/12)*fluxe[2*i] + (sigma_s[i]*deltax[i]/6)*fluxe[2*i + 1];

      upwind_local[0] = 0.;
      upwind_local[1] = af_rb;

      // Now generate the src_local vec

      ierr = VecSetValue(src_local, 0, scat_local[0] + upwind_local[0] + q0dis[2*i], INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecSetValue(src_local, 1, scat_local[1] + upwind_local[1] + q0dis[2*i + 1], INSERT_VALUES); CHKERRQ(ierr);

      // Now that we have src_local, we generate mat_local
      add = (-mu[o]/2) + (sigma_t[i]*deltax[i]/3);
      ierr = MatSetValue(mat_local, 0 , 0, add, INSERT_VALUES);

      add = (mu[o]/2) + (sigma_t[i]*deltax[i]/6);
      ierr = MatSetValue(mat_local, 0, 1, add, INSERT_VALUES);

      add = (-mu[o]/2) + (sigma_t[i]*deltax[i]/6);
      ierr = MatSetValue(mat_local, 1, 0, add, INSERT_VALUES);

      add = (-mu[o]/2) + (sigma_t[i]*deltax[i]/3);
      ierr = MatSetValue(mat_local, 1, 1, add, INSERT_VALUES);

      // now do matrix assembly here idk if needed here though - will do later if i get ERROR

      // Assemble vector and matrix

       VecAssemblyBegin(src_local);
       VecAssemblyEnd(src_local);
       MatAssemblyBegin(mat_local, MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(mat_local, MAT_FINAL_ASSEMBLY);


      // SET UP OPERATORS

    //  ierr = KSPCreate(PETSC_COMM_SELF, &solver); CHKERRQ(ierr);
      ierr = KSPSetOperators(solver, mat_local, mat_local); CHKERRQ(ierr);
      // ierr = KSPSetType(solver, KSPPREONLY); CHKERRQ(ierr);
      // ierr = KSPGetPC(solver, &pc); CHKERRQ(ierr);
      // ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);
      // ierr = KSPSetFromOptions(solver); CHKERRQ(ierr);
      ierr = KSPSolve(solver, src_local, af_local); CHKERRQ(ierr);

      // NOW EXTRACT THE SOLVE INTO REAL ARRAY AND ASSIGN THAT TO THE ang_flux ARRAY
      ierr = VecGetArray(af_local, &af_extract); CHKERRQ(ierr);
      ang_flux[2*i][o] = af_extract[0];
      ang_flux[2*i+1][o] = af_extract[1];
      af_rb = -mu[o]*af_extract[0]; // for upwinding next cell
      ierr = VecRestoreArray(af_local, &af_extract); CHKERRQ(ierr);
    //  ierr = KSPDestroy(&solver); CHKERRQ(ierr);

    }  // end for over elem

  } // end if


  } // end for over angle

//  PetscPrintf(PETSC_COMM_SELF, "\n\n \n outside the sweep \n\n\n\n\n");

// Now calculate scalar flux at edges - fluxe

  for (i = 0; i < n_elem; i++) {

    for (o = 0; o < order; o++) {

      fluxe[2*i] = 0;
      fluxe[2*i+1] = 0;

      }
    }



  for (i = 0; i < n_elem; i++) {

    for (o = 0; o < order; o++) {

      fluxe[2*i] = fluxe[2*i] + wt[o]*ang_flux[2*i][o];
      fluxe[2*i+1] = fluxe[2*i+1] + wt[o]*ang_flux[2*i+1][o];
      }
  }

//  PetscPrintf(PETSC_COMM_SELF, "\n\n\n\n\n calculated flux \n\n\n\n\n");

  // Now calculate norms

  for(i = 0; i < n_elem; i++){

    ierr = VecSetValue(norflx, 2*i, fluxe[2*i], INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValue(norflx_o, 2*i, fluxe_o[2*i], INSERT_VALUES); CHKERRQ(ierr);

    ierr = VecSetValue(norflx, 2*i+1, fluxe[2*i+1], INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecSetValue(norflx_o, 2*i+1, fluxe_o[2*i+1], INSERT_VALUES); CHKERRQ(ierr);

  }

  VecAssemblyBegin(norflx);
  VecAssemblyBegin(norflx_o);
  VecAssemblyEnd(norflx);
  VecAssemblyEnd(norflx_o);

  ierr = VecNorm(norflx_o, NORM_2, &norm_o); CHKERRQ(ierr);
  ierr = VecAYPX(norflx_o, -1, norflx); CHKERRQ(ierr);
  ierr = VecNorm(norflx_o, NORM_2, &norm); CHKERRQ(ierr);
  eps = norm/norm_o;
//  ierr = PetscPrintf(PETSC_COMM_SELF, "\n \n \n \n eps is %f \n \n \n", eps); CHKERRQ(ierr);
  // now set values for flux_o

  for (i = 0; i < n_elem; i++) {

    fluxe_o[2*i] = fluxe[2*i];

    //PetscPrintf(PETSC_COMM_SELF, "\n %f", fluxe_o[2*i]);

    fluxe_o[2*i+1] = fluxe[2*i+1];

    //PetscPrintf(PETSC_COMM_SELF, "\n %f", fluxe_o[2*i+1]);

  }


  its = its+1;
  // ierr = PetscPrintf(PETSC_COMM_SELF, " \n \n its are %d \n", its); CHKERRQ(ierr);

 } // end while

// NOW FIND CELL CENTERED FLUX

for (i = 0; i < n_elem; i++) {

  fluxc[i] = (fluxe[2*i] + fluxe[2*i+1])/2;
  ierr = PetscPrintf(PETSC_COMM_SELF, " %f \n", fluxc[i]); CHKERRQ(ierr);

 }

ierr = PetscPrintf(PETSC_COMM_SELF, " %d \n", its); CHKERRQ(ierr);

VecDestroy(&norflx);
VecDestroy(&norflx_o);
VecDestroy(&src_local);
MatDestroy(&mat_local);


/*============================================================================
 Every process in PETSc must end with a PetscFinalize()
============================================================================*/

ierr = PetscFinalize();
return 0;

}
