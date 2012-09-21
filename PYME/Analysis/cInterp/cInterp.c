/*
##################
# cInterp.c
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
 */

#include "Python.h"
#include <complex.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>


static PyObject * Interpolate(PyObject *self, PyObject *args, PyObject *keywds)
{
    float *res = 0;
    
    npy_intp outDimensions[3];
    int sizeX, sizeY, sizeZ;
    int xi, yi, j;
    
    PyObject *omod =0;
    
    PyArrayObject* amod;
    
    PyArrayObject* out;

    //double *mod = 0;
    
    /*parameters*/
    float x0,y0,z0, dx, dy, dz;
    int nx, ny;

    /*End paramters*/

    float rx, ry, rz;
    float r000, r100, r010, r110, r001, r101, r011, r111;
    int fx, fy, fz;
    
    static char *kwlist[] = {"model", "x0","y0", "z0","nx","ny","dx", "dy", "dz", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Offfiifff", kwlist,
         &omod, &x0, &y0, &z0, &nx, &ny, &dx, &dy, &dz))
        return NULL; 

    /* Do the calculations */ 
        
    amod = (PyArrayObject *) PyArray_ContiguousFromObject(omod, PyArray_FLOAT, 3, 3);
    if (amod == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad model");
      return NULL;
    }
  
    
    //pmod = (double*)amod->data;

    sizeX = PyArray_DIM(amod, 0);
    sizeY = PyArray_DIM(amod, 1);
    sizeZ = PyArray_DIM(amod, 2);
    

    outDimensions[0] = nx;
    outDimensions[1] = ny;
    outDimensions[2] = 1;

    //printf("shp: %d, %d", nx, ny);
        
    out = (PyArrayObject*) PyArray_SimpleNew(3,outDimensions,PyArray_FLOAT);
    if (out == NULL)
    {
      Py_DECREF(amod);
      
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      return NULL;
    }
    
    
    res = (float*) out->data;

    //Initialise our histogram
    for (j =0; j < nx*ny; j++)
    {
        res[j] = 0.0;
    }

    fx = (int)(floorf(sizeX/2.0) + floorf(x0/dx));
    fy = (int)(floorf(sizeY/2.0) + floorf(y0/dy));
    fz = (int)(floorf(sizeZ/2.0) + floorf(z0/dz));

    ///avoid negatives by adding a chunk before taking the mod
    rx = fmodf(x0+973*dx,dx)/dx;
    ry = fmodf(y0+973*dy,dy)/dy;
    rz = fmodf(z0+973*dz,dz)/dz;

    //printf("%3.3f, %d, %3.3f\n", rz, fz, z0);

    r000 = ((1.0-rx)*(1.0-ry)*(1.0-rz));
    r100 = ((rx)*(1.0-ry)*(1.0-rz));
    r010 = ((1.0-rx)*(ry)*(1.0-rz));
    r110 = ((rx)*(ry)*(1.0-rz));
    r001 = ((1.0-rx)*(1.0-ry)*(rz));
    r101 = ((rx)*(1.0-ry)*(rz));
    r011 = ((1.0-rx)*(ry)*(rz));
    r111 = ((rx)*(ry)*(rz));        

    for (xi = fx; xi < (fx+nx); xi++)
      {            
	for (yi = fy; yi < (fy +ny); yi++)
        {
            *res  = r000 * *(float*)PyArray_GETPTR3(amod, xi,   yi,   fz);
            *res += r100 * *(float*)PyArray_GETPTR3(amod, xi+1, yi,   fz);
            *res += r010 * *(float*)PyArray_GETPTR3(amod, xi,   yi+1, fz);
            *res += r110 * *(float*)PyArray_GETPTR3(amod, xi+1, yi+1, fz);
            *res += r001 * *(float*)PyArray_GETPTR3(amod, xi,   yi,   fz+1);
            *res += r101 * *(float*)PyArray_GETPTR3(amod, xi+1, yi,   fz+1);
            *res += r011 * *(float*)PyArray_GETPTR3(amod, xi,   yi+1, fz+1);
            *res += r111 * *(float*)PyArray_GETPTR3(amod, xi+1, yi+1, fz+1);

            res ++;
	  }
       
      }
    
    
    Py_DECREF(amod);
    
    return (PyObject*) out;
}




static PyMethodDef cInterpMethods[] = {
    {"Interpolate",  Interpolate, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initcInterp(void)
{
    PyObject *m;

    m = Py_InitModule("cInterp", cInterpMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
