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
//#include <complex.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

#define MIN(a, b) ((a<b) ? a : b) 
#define MAX(a, b) ((a>b) ? a : b)


static PyObject * Interpolate(PyObject *self, PyObject *args, PyObject *keywds)
{
    float *res = 0;
    float tmp = 0;
    
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
    
    Py_BEGIN_ALLOW_THREADS;
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
                tmp  = r000 * (*(float*)PyArray_GETPTR3(amod, xi,   yi,   fz));
                tmp += r100 * (*(float*)PyArray_GETPTR3(amod, xi+1, yi,   fz));
                tmp += r010 * (*(float*)PyArray_GETPTR3(amod, xi,   yi+1, fz));
                tmp += r110 * (*(float*)PyArray_GETPTR3(amod, xi+1, yi+1, fz));
                tmp += r001 * (*(float*)PyArray_GETPTR3(amod, xi,   yi,   fz+1));
                tmp += r101 * (*(float*)PyArray_GETPTR3(amod, xi+1, yi,   fz+1));
                tmp += r011 * (*(float*)PyArray_GETPTR3(amod, xi,   yi+1, fz+1));
                tmp += r111 * (*(float*)PyArray_GETPTR3(amod, xi+1, yi+1, fz+1));

                res[0] = tmp;

                res ++;
          }
       
      }
    
    Py_END_ALLOW_THREADS;
    Py_DECREF(amod);
    
    return (PyObject*) out;
}


void splCoeff(float r, float *coeffs)
{
    //calculate spline coefficients    
    int i = 0;
    float y = 0;

    for (i = 0; i <= 3; i++)
    {
        y = fabs(-1 - r + i);

        if (y < 1)
        {
            coeffs[i] = (y*y*(y-2.0)*3.0 + 4.)/6.;
        }
        else if (y < 2.0)
        {
            y = 2.0 - y;
            coeffs[i] = y*y*y/6.0;
        } else 
        {
            coeffs[i] = 0;
        }
    }
}

static PyObject * InterpolateCS(PyObject *self, PyObject *args, PyObject *keywds)
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
    //float r000, r100, r010, r110, r001, r101, r011, r111;
    int fx, fy, fz;
    int xj, yj, zj;

    float cx[4], cy[4], cz[4];
    
    //printf("parsing python parameters\n");

    static char *kwlist[] = {"model", "x0","y0", "z0","nx","ny","dx", "dy", "dz", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Offfiifff", kwlist,
         &omod, &x0, &y0, &z0, &nx, &ny, &dx, &dy, &dz))
        return NULL; 

        
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
    rx = fmodf(x0+973*dx,dx)/dx;// - 0.5;
    ry = fmodf(y0+973*dy,dy)/dy;// - 0.5;
    rz = fmodf(z0+973*dz,dz)/dz;// - 0.5;

    //printf("calculating spline coefficients\n");
    //calculate the spline coefficients
    splCoeff(rx, cx);
    splCoeff(ry, cy);
    splCoeff(rz, cz);

    //printf("calculated spline coefficients\n");

    if ((fx < 1) || ((fx + nx + 3) > sizeX)){
        //x out of bounds
        fprintf(stderr, "x out of bounds\n");

        Py_DECREF(amod);
        Py_DECREF(out);

        PyErr_Format(PyExc_RuntimeError, "X coordinates out of range - fx = %d", fx);
        return NULL;
    }

    if ((fy < 1) || ((fy + ny + 3) > sizeY)){
        //x out of bounds
        fprintf(stderr, "y out of bounds\n");

        Py_DECREF(amod);
        Py_DECREF(out);

        PyErr_Format(PyExc_RuntimeError, "Y coordinates out of range - fy = %d", fy);
        return NULL;
    }

    if ((fz < 1) || (fz +3 > sizeZ)){
        //x out of bounds
        fprintf(stderr, "z out of bounds\n");

        Py_DECREF(amod);
        Py_DECREF(out);

        PyErr_Format(PyExc_RuntimeError, "Z coordinates out of range - fz = %d", fz);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;

    //fprintf(stderr, "done interp prep work\n");
    //fprintf(stderr, "done interp prep work\n");

    for (xi = fx; xi < (fx+nx); xi++)
      {            
	for (yi = fy; yi < (fy +ny); yi++)
        {
            *res  = 0;
            
            for (xj=0; xj <= 3; xj ++)
            {
                for (yj=0; yj <= 3; yj ++)
                {
                    for (zj=0; zj <= 3; zj ++)
                        {
                            *res += cx[xj]*cy[yj]*cz[zj] * *(float*)PyArray_GETPTR3(amod, xi + xj - 1, yi + yj - 1,   fz + zj - 1);
                        }
                }
            }
            

            res ++;
	  }
       
      }
      //printf("done interpolation\n");

    Py_END_ALLOW_THREADS;
    Py_DECREF(amod);
    
    return (PyObject*) out;
}

static PyObject * InterpolateInplace(PyObject *self, PyObject *args, PyObject *keywds)
{
    float *res = 0;
    
    //npy_intp outDimensions[3];
    int sizeX, sizeY, sizeZ;
    int oSizeX, oSizeY;
    int xi, yi;//, j;
    
    PyObject *amod =0;
    
    //PyArrayObject* amod;
    
    PyObject* out=0;

    //double *mod = 0;
    
    /*parameters*/
    float x0,y0,z0, dx, dy, dz, A;
    int nx, ny;

    /*End paramters*/

    float rx, ry, rz;
    float r000, r100, r010, r110, r001, r101, r011, r111;
    //int nx, ny;
    int fx, fy, fz, cx, cy;
    int xo, yo;//, zo;
    
    static char *kwlist[] = {"model", "output", "x0","y0", "z0","nx", "ny", "dx", "dy", "dz", "A", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOfffiiffff", kwlist,
         &amod, &out, &x0, &y0, &z0, &nx, &ny, &dx, &dy, &dz, &A))
        return NULL; 

    /* Do the calculations */ 
        
    if (PyArray_TYPE(amod) != PyArray_FLOAT)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad model");
      return NULL;
    }

    if (PyArray_TYPE(out) != PyArray_FLOAT)
    {
        PyErr_Format(PyExc_RuntimeError, "bad output array");
        return NULL;
    }
  
    
    //pmod = (double*)amod->data;

    sizeX = PyArray_DIM(amod, 0);
    sizeY = PyArray_DIM(amod, 1);
    sizeZ = PyArray_DIM(amod, 2);

    oSizeX = PyArray_DIM(out, 0) - 1;
    oSizeY = PyArray_DIM(out, 1) - 1;
    
    
    Py_BEGIN_ALLOW_THREADS;

       

    fx = (int)(floorf(x0/dx));
    fy = (int)(floorf(y0/dy));
    fz = (int)(floorf(sizeZ/2.0) + floorf(z0/dz));

    cx = (int)(floorf(sizeX/2.0)) - fx;//+ floorf(x0/dx));
    cy = (int)(floorf(sizeY/2.0)) - fy; //+ floorf(y0/dy)); 

    ///avoid negatives by adding a chunk before taking the mod
    rx = fmodf(x0+973*dx,dx)/dx;
    ry = fmodf(y0+973*dy,dy)/dy;
    rz = fmodf(z0+973*dz,dz)/dz;

    //printf("%3.3f, %d, %3.3f\n", rz, fz, z0);

    r000 = A*((1.0-rx)*(1.0-ry)*(1.0-rz));
    r100 = A*((rx)*(1.0-ry)*(1.0-rz));
    r010 = A*((1.0-rx)*(ry)*(1.0-rz));
    r110 = A*((rx)*(ry)*(1.0-rz));
    r001 = A*((1.0-rx)*(1.0-ry)*(rz));
    r101 = A*((rx)*(1.0-ry)*(rz));
    r011 = A*((1.0-rx)*(ry)*(rz));
    r111 = A*((rx)*(ry)*(rz));        

    for (xo = MAX(fx - nx, 0); xo < MIN(fx + nx + 1, oSizeX); xo++)
      {            
        xi = xo + cx;

	for (yo = MAX(fy -ny, 0); yo < MIN(fy + ny + 1, oSizeY); yo++)
        {
            yi = yo + cy;
            
            res = (float*)PyArray_GETPTR2(out, xo,   yo); 
           
            *res += r000 * *(float*)PyArray_GETPTR3(amod, xi,   yi,   fz);
            *res += r100 * *(float*)PyArray_GETPTR3(amod, xi+1, yi,   fz);
            *res += r010 * *(float*)PyArray_GETPTR3(amod, xi,   yi+1, fz);
            *res += r110 * *(float*)PyArray_GETPTR3(amod, xi+1, yi+1, fz);
            *res += r001 * *(float*)PyArray_GETPTR3(amod, xi,   yi,   fz+1);
            *res += r101 * *(float*)PyArray_GETPTR3(amod, xi+1, yi,   fz+1);
            *res += r011 * *(float*)PyArray_GETPTR3(amod, xi,   yi+1, fz+1);
            *res += r111 * *(float*)PyArray_GETPTR3(amod, xi+1, yi+1, fz+1);

	  }
       
      }
    
    Py_END_ALLOW_THREADS;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * InterpolateInplaceM(PyObject *self, PyObject *args, PyObject *keywds)
{
    float *res = 0;
    
    //npy_intp outDimensions[3];
    int sizeX, sizeY, sizeZ;
    int oSizeX, oSizeY;
    int xi, yi;//, j;
    
    PyObject *amod =0;
    
    //PyArrayObject* amod;
    
    PyObject* out=0;
    
    PyObject* xv=0;
    PyObject* yv=0;
    PyObject* zv=0;
    PyObject* Av=0;
    PyObject* nv=0;

    //double *mod = 0;
    
    /*parameters*/
    float x0,y0,z0, dx, dy, dz, A;
    int nx;//, ny;

    /*End paramters*/

    float rx, ry, rz;
    float r000, r100, r010, r110, r001, r101, r011, r111;
    //int nx, ny;
    int fx, fy, fz, cx, cy;
    int xo, yo;//, zo;
    int i, npts;
    
    static char *kwlist[] = {"model", "output", "x0","y0", "z0", "Av","nx", "dx", "dy", "dz", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOOOfff", kwlist,
         &amod, &out, &xv, &yv, &zv, &Av, &nv, &dx, &dy, &dz))
        return NULL; 

    /* Do the calculations */ 
        
    if (PyArray_TYPE(amod) != PyArray_FLOAT)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad model");
      return NULL;
    }

    if (PyArray_TYPE(out) != PyArray_FLOAT)
    {
        PyErr_Format(PyExc_RuntimeError, "bad output array");
        return NULL;
    }

    if (PyArray_TYPE(xv) != PyArray_FLOAT)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x");
      return NULL;
    }

    npts = PyArray_DIM(xv, 0);
    //printf("n: %d\n", npts);

    if (PyArray_TYPE(yv) != PyArray_FLOAT || PyArray_DIM(yv, 0) != npts)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad y");
      return NULL;
    }

    if (PyArray_TYPE(zv) != PyArray_FLOAT || PyArray_DIM(zv, 0) != npts)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad z");
      return NULL;
    }

    if (PyArray_TYPE(Av) != PyArray_FLOAT || PyArray_DIM(Av, 0) != npts)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad A");
      return NULL;
    }

    if (PyArray_TYPE(nv) != PyArray_INT || PyArray_DIM(nv, 0) != npts)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad nx");
      return NULL;
    }
  
    
    //pmod = (double*)amod->data;

    sizeX = PyArray_DIM(amod, 0);
    sizeY = PyArray_DIM(amod, 1);
    sizeZ = PyArray_DIM(amod, 2);

    oSizeX = PyArray_DIM(out, 0) - 1;
    oSizeY = PyArray_DIM(out, 1) - 1;

    //npts = PyArray_DIM(xv, 0);
    
    
    Py_BEGIN_ALLOW_THREADS;

    for (i=0; i < npts; i++)
    {
        x0 = *(float*)PyArray_GETPTR1(xv, i);
        y0 = *(float*)PyArray_GETPTR1(yv, i);
        z0 = *(float*)PyArray_GETPTR1(zv, i);
        A = *(float*)PyArray_GETPTR1(Av, i);
        nx = *(int*)PyArray_GETPTR1(nv, i);

        //printf("p: %d\t", nx);
        
       

    fx = (int)(floorf(x0/dx));
    fy = (int)(floorf(y0/dy));
    fz = MIN(sizeZ - 2, MAX(0, (int)(floorf(sizeZ/2.0) + floorf(z0/dz))));

    cx = (int)(floorf(sizeX/2.0)) - fx;//+ floorf(x0/dx));
    cy = (int)(floorf(sizeY/2.0)) - fy; //+ floorf(y0/dy)); 

    ///avoid negatives by adding a chunk before taking the mod
    rx = 1.0 - fmodf(x0+973*dx,dx)/dx;
    ry = 1.0 - fmodf(y0+973*dy,dy)/dy;
    rz = 1.0 - fmodf(z0+973*dz,dz)/dz;

    //printf("%3.3f, %d, %3.3f\n", rz, fz, z0);

    r000 = A*((1.0-rx)*(1.0-ry)*(1.0-rz));
    r100 = A*((rx)*(1.0-ry)*(1.0-rz));
    r010 = A*((1.0-rx)*(ry)*(1.0-rz));
    r110 = A*((rx)*(ry)*(1.0-rz));
    r001 = A*((1.0-rx)*(1.0-ry)*(rz));
    r101 = A*((rx)*(1.0-ry)*(rz));
    r011 = A*((1.0-rx)*(ry)*(rz));
    r111 = A*((rx)*(ry)*(rz));        

    for (xo = MAX(fx - nx, 0); xo < MIN(fx + nx + 1, oSizeX); xo++)
      {            
        xi = xo + cx;

	for (yo = MAX(fy -nx, 0); yo < MIN(fy + nx + 1, oSizeY); yo++)
        {
            yi = yo + cy;
            
            res = (float*)PyArray_GETPTR2(out, xo,   yo); 
           
            *res += r000 * *(float*)PyArray_GETPTR3(amod, xi,   yi,   fz);
            *res += r100 * *(float*)PyArray_GETPTR3(amod, xi+1, yi,   fz);
            *res += r010 * *(float*)PyArray_GETPTR3(amod, xi,   yi+1, fz);
            *res += r110 * *(float*)PyArray_GETPTR3(amod, xi+1, yi+1, fz);
            *res += r001 * *(float*)PyArray_GETPTR3(amod, xi,   yi,   fz+1);
            *res += r101 * *(float*)PyArray_GETPTR3(amod, xi+1, yi,   fz+1);
            *res += r011 * *(float*)PyArray_GETPTR3(amod, xi,   yi+1, fz+1);
            *res += r111 * *(float*)PyArray_GETPTR3(amod, xi+1, yi+1, fz+1);

	  }
       
      }
    }
    Py_END_ALLOW_THREADS;

    Py_INCREF(Py_None);
    return Py_None;
}


static PyMethodDef cInterpMethods[] = {
    {"Interpolate",  (PyCFunction)Interpolate, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
    {"InterpolateCS",  (PyCFunction)InterpolateCS, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
    {"InterpolateInplace",  (PyCFunction)InterpolateInplace, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
    {"InterpolateInplaceM",  (PyCFunction)InterpolateInplaceM, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},


    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cInterp",     /* m_name */
        "fast interpolation",  /* m_doc */
        -1,                  /* m_size */
        cInterpMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_cInterp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else
PyMODINIT_FUNC initcInterp(void)
{
    PyObject *m;

    m = Py_InitModule("cInterp", cInterpMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
#endif