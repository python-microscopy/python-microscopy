#ifndef _drawTriang_h_
#define _drawTriang_h_

#include "qhull/qhull_a.h"
#include <math.h>

#define BOOL unsigned int

void drawTriangle (double* pImage, int sizeX, int sizeY, double x0, double y0, double x1, double y1, double x2, double y2, float val);

void drawTetrahedron (double* pImage, int sizeX, int sizeY, int sizeZ, double x0,
        double y0, double z0, double x1, double y1, double z1, double x2, double y2,
        double z2, double x3, double y3, double z3, float val);

int tetAndDraw(coordT *points, int numpoints, double *pImage, int sizeX, int sizeY, int sizeZ, BOOL calc_area);

#endif