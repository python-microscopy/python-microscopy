#include "qhull/qhull_a.h"
#include <math.h>

#define BOOL unsigned int

int calcLHood (coordT *points, int dim, int numpoints, double *lsum);
int calcLHood2D (coordT *points, int numpoints, double *lsum);
int tetAndDraw(coordT *points, int numpoints, double *pImage, int sizeX, int sizeY, int sizeZ, BOOL calc_area);