CFLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = Home/NetBeansProjects/petsc_ksp
EXAMPLESC        = ex1.c
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

vecs: vecs.o chkopts
	  -${CLINKER} -w -o vecs vecs.o  ${PETSC_KSP_LIB}
	   ${RM} vecs.o

P1slab_solvers:  P1slab_solvers.o chkopts
	  -${CLINKER} -o P1slab_solvers P1slab_solvers.o  ${PETSC_KSP_LIB}
	   ${RM} P1slab_solvers.o

P1slab: P1slab.o chkopts
	  -${CLINKER} -o P1slab P1slab.o  ${PETSC_KSP_LIB}
	   ${RM} P1slab.o

SNslab: SNslab.o chkopts
	-${CLINKER} -o SNslab SNslab.o  ${PETSC_KSP_LIB}
	 ${RM} SNslab.o

SNrect: SNrect.o chkopts
	-${CLINKER} -o SNrect SNrect.o  ${PETSC_KSP_LIB}
	 ${RM} SNrect.o

SNslabScratch: SNslabScratch.o chkopts
	-${CLINKER} -o SNslabScratch SNslabScratch.o  ${PETSC_KSP_LIB}
	 ${RM} SNslabScratch.o

SNslab_dsa: SNslab_dsa.o chkopts
	-${CLINKER} -o SNslab_dsa SNslab_dsa.o  ${PETSC_KSP_LIB}
	 ${RM} SNslab_dsa.o
