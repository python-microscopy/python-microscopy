This directory contains a matlab MEX interface to levmar. This interface
has been tested with Matlab v. 6.5 R13 under linux and v. 7.4 R2007 under Windows.
The following files are included:

levmar.c: C MEX-file for levmar
Makefile: UNIX makefile for compiling levmar.c using mex
Makefile.w32: Windows makefile for compiling levmar.c using mex
levmar.m: Documentation for the MEX interface
lmdemo.m: Demonstration of using the MEX interface; run as matlab < lmdemo.m

*.m: Matlab functions implementing various objective functions and their Jacobians.
     For instance, meyer.m implements the objective function for Meyer's (reformulated)
     problem and jacmeyer.m implements its Jacobian.
