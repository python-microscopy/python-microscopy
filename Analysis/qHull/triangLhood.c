/*
##################
# triangLhood.c
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
 */

#include "triangLhood.h"

int calcLHood (coordT *points, int dim, int numpoints, double *lsum) {
  const char *flags = "qhull d Qbb Qt";//"qhull s d Tcv ";          /* option flags for qhull, see qh_opt.htm */
  int exitcode;             /* 0 if no error from qhull */
  facetT *facet;            /* set by FORALLfacets */
  //double lsum = 0;
  int curlong, totlong;     /* memory remaining after qh_memfreeshort */

  *lsum = 0;
  
  exitcode= qh_new_qhull (dim, numpoints, points, False,
                      flags, NULL, stderr);
  if (!exitcode) {                  /* if no error */
    FORALLfacets {
        *lsum += log(sqrt(qh_facetarea(facet)));
    } 
  }

  *lsum /= numpoints;

  qh_freeqhull(!qh_ALL);                 /* free long memory */
  qh_memfreeshort (&curlong, &totlong);  /* free short memory and memory allocator */
  if (curlong || totlong)
    fprintf (stderr, "qhull internal warning (user_eg, #2): did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);

  
  return exitcode;
}

#define DIST_(a, b, dim)\
    dist=0;for (dim_num=0;dim_num < dim; dim_num++){dist +=(a[dim_num] - b[dim_num])*(a[dim_num] - b[dim_num]);}


int calcLHood2D (coordT *points, int numpoints, double *lsum) {
  #define dim 2
  const char *flags = "qhull d Qbb Qt";//"qhull s d Tcv ";          /* option flags for qhull, see qh_opt.htm */
  int exitcode;             /* 0 if no error from qhull */
  facetT *facet;            /* set by FORALLfacets */
  realT *vertices[dim + 1];
  //int  vertex_n, vertex_i;
  //double lsum = 0;
  int curlong, totlong;     /* memory remaining after qh_memfreeshort */
  int i, j, dim_num;
  double dist;

  *lsum = 0;

  exitcode= qh_new_qhull (dim, numpoints, points, False,
                      flags, NULL, stderr);
  if (!exitcode) {                  /* if no error */
   //qh_vertexneighbors();
   FORALLfacets {
        //*lsum += log(sqrt(qh_facetarea(facet)));
        //FOREACHridge_i_(facet->ridges){
        //    ridge->vertices;
        //}

       for (i=0; i < (dim+1); i++) //for each vertex
       {
           vertices[i] = SETelemt_(facet->vertices, i, vertexT)->point;
           
           for (j =0; j < i;j++) //for each preceeding vertex
           {
               DIST_(vertices[i], vertices[j], dim);
               *lsum += log(sqrt(dist));
           }
       }


        //printf("Num ridges: %d\n", qh_setsize(facet->ridges));
    }
  }

  *lsum /= (numpoints*3);

  qh_freeqhull(!qh_ALL);                 /* free long memory */
  qh_memfreeshort (&curlong, &totlong);  /* free short memory and memory allocator */
  if (curlong || totlong)
    fprintf (stderr, "qhull internal warning (user_eg, #2): did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);


  return exitcode;
}

int calcLHood3D (coordT *points, int numpoints, double *lsum) {
  #define dim 3
  const char *flags = "qhull d Qbb Qt";//"qhull s d Tcv ";          /* option flags for qhull, see qh_opt.htm */
  int exitcode;             /* 0 if no error from qhull */
  facetT *facet;            /* set by FORALLfacets */
  realT *vertices[dim + 1];
  //int  vertex_n, vertex_i;
  //double lsum = 0;
  int curlong, totlong;     /* memory remaining after qh_memfreeshort */
  int i, j, dim_num;
  double dist;

  *lsum = 0;

  exitcode= qh_new_qhull (dim, numpoints, points, False,
                      flags, NULL, stderr);
  if (!exitcode) {                  /* if no error */
   //qh_vertexneighbors();
   FORALLfacets {
        //*lsum += log(sqrt(qh_facetarea(facet)));
        //FOREACHridge_i_(facet->ridges){
        //    ridge->vertices;
        //}

       for (i=0; i < (dim+1); i++) //for each vertex
       {
           vertices[i] = SETelemt_(facet->vertices, i, vertexT)->point;

           for (j =0; j < i;j++) //for each preceeding vertex
           {
               DIST_(vertices[i], vertices[j], dim);
               *lsum += log(sqrt(dist));
           }
       }


        //printf("Num ridges: %d\n", qh_setsize(facet->ridges));
    }
  }

  *lsum /= (numpoints*6);

  qh_freeqhull(!qh_ALL);                 /* free long memory */
  qh_memfreeshort (&curlong, &totlong);  /* free short memory and memory allocator */
  if (curlong || totlong)
    fprintf (stderr, "qhull internal warning (user_eg, #2): did not free %d bytes of long memory (%d pieces)\n", totlong, curlong);


  return exitcode;
}
