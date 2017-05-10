/*
##################
# triangRend.c
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
 */

//#include "triangLhood.h"
#include "drawTriang.h"

#define DIST_(a, b, dim)\
    dist=0;for (dim_num=0;dim_num < dim; dim_num++){dist +=(a[dim_num] - b[dim_num])*(a[dim_num] - b[dim_num]);}


int tetAndDraw(coordT *points, int numpoints, double *pImage, int sizeX, int sizeY, int sizeZ, BOOL calc_area) {
  #define dim 3
  const char *flags = "qhull d Qbb Qt";//"qhull s d Tcv ";          /* option flags for qhull, see qh_opt.htm */
  int exitcode;             /* 0 if no error from qhull */
  facetT *facet;            /* set by FORALLfacets */
  //realT *vertex;
  realT *vertices[dim + 1];
  double vertex_x[4];
  double vertex_y[4];
  double vertex_z[4];
  double lsum = 0;
  double dist = 0;

  //double lsum = 0;
  int i = 0, j=0, dim_num=0;
  int curlong, totlong;     /* memory remaining after qh_memfreeshort */
  
  exitcode= qh_new_qhull (dim, numpoints, points, False,
                      flags, NULL, stderr);
  if (!exitcode) {                  /* if no error */
    FORALLfacets {
       lsum = 0;
       for (i=0; i < (dim+1); i++) //for each vertex
       {
           vertices[i] = SETelemt_(facet->vertices, i, vertexT)->point;
           vertex_x[i] = vertices[i][0];
           vertex_y[i] = vertices[i][1];
           vertex_z[i] = vertices[i][2];

           if (!calc_area)
           {
               for (j =0; j < i;j++) //for each preceeding vertex
               {
                   DIST_(vertices[i], vertices[j], dim);
                   lsum += sqrt(dist);
               }
           }
       }

       if (calc_area)
       {
           lsum = qh_facetarea(facet);
       } else //approximate area as mean side length
       {
           lsum /= 6;
           lsum = lsum*lsum*lsum;
       }

       drawTetrahedron(pImage, sizeX, sizeY, sizeZ, vertex_x[0],
                    vertex_y[0], vertex_z[0], vertex_x[1], vertex_y[1], vertex_z[1], vertex_x[2], vertex_y[2],
                    vertex_z[2], vertex_x[3], vertex_y[3], vertex_z[3], 1.0 / lsum);
       
    } 
  }


  qh_freeqhull(!qh_ALL);                 /* free long memory */
  qh_memfreeshort (&curlong, &totlong);  /* free short memory and memory allocator */
  if (curlong || totlong)
    fprintf (stderr, "qhull internal warning (user_eg, #2): did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);

  
  return exitcode;
}



