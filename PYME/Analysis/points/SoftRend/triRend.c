/*
##################
# triRend.c
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
#include "drawTriang.h"

/*
void drawTriangle (double* pImage, int sizeX, int sizeY, double x0, double y0, double x1, double y1, double x2, double y2, float val)
{
    double tmp;
    double y01, y02, y12;
    double m01, m02, m12;
    int x, y;

    // Sort the points so that x0 <= x1 <= x2
    if (x0 > x1) { tmp=x0; x0=x1; x1=tmp; tmp=y0; y0=y1; y1=tmp;}
    if (x0 > x2) { tmp=x0; x0=x2; x2=tmp; tmp=y0; y0=y2; y2=tmp;}
    if (x1 > x2) { tmp=x1; x1=x2; x2=tmp; tmp=y1; y1=y2; y2=tmp;}

    if ((x0 < 0.0) || (x1 < 0.0) || (x2 < 0.0) || (y0 < 0.0) || (y1 < 0.0) || (y2 < 0.0)
            || (x0 >= (double)sizeX) || (x1 >= (double)sizeX) || (x2 >= (double)sizeX)
            || (y0 >= (double)sizeY) || (y1 >= (double)sizeY) || (y2 >= (double)sizeY)
            )
    {
        return; //drop any triangles which extend over the boundaries
    }

    
    //calculate gradient
    m01 = (y1-y0)/(x1-x0);
    m02 = (y2-y0)/(x2-x0);
    m12 = (y2-y1)/(x2-x1);

    y01 = y0;
    y02 = y0;
    y12 = y1;

    // Draw vertical segments
    for (x = (int)x0; x < (int)x1; x++)
    {
        if (y01 < y02)
        {
            for (y = (int)y01; y < (int)y02; y++)
                pImage[sizeY*x + y] += val;
        }
        else
        {
            for (y = (int)y02; y < (int)y01; y++)
                pImage[sizeY*x + y] += val;
        }

        y01 += m01;
        y02 += m02;

    }
    

    for (x = (int)x1; x < (int)x2; x++)
    {
        if (y12 < y02)
        {
            for (y = (int)y12; y < (int)y02; y++)
                pImage[sizeY*x + y] += val;
        }
        else
        {
            for (y = (int)y02; y < (int)y12; y++)
                pImage[sizeY*x + y] += val;
        }

        y12 += m12;
        y02 += m02;

    }
            
}

void drawTetrahedron (double* pImage, int sizeX, int sizeY, int sizeZ, double x0,
        double y0, double z0, double x1, double y1, double z1, double x2, double y2,
        double z2, double x3, double y3, double z3, float val)
{
    double tmp;
    double y01, y02, y03, y12, y13, y23;
    double x01, x02, x03, x12, x13, x23;
    double m01x, m02x, m03x, m12x, m13x, m23x;
    double m01y, m02y, m03y, m12y, m13y, m23y;
    int z;

    // Sort the points so that z0 <= z1 <= z2 <= z3
    if (z0 > z1) { tmp=x0; x0=x1; x1=tmp; tmp=y0; y0=y1; y1=tmp; tmp=z0; z0=z1; z1=tmp;}
    if (z0 > z2) { tmp=x0; x0=x2; x2=tmp; tmp=y0; y0=y2; y2=tmp; tmp=z0; z0=z2; z2=tmp;}
    if (z0 > z3) { tmp=x0; x0=x3; x3=tmp; tmp=y0; y0=y3; y3=tmp; tmp=z0; z0=z3; z3=tmp;}
    if (z1 > z2) { tmp=x1; x1=x2; x2=tmp; tmp=y1; y1=y2; y2=tmp; tmp=z1; z1=z2; z2=tmp;}
    if (z1 > z3) { tmp=x1; x1=x3; x3=tmp; tmp=y1; y1=y3; y3=tmp; tmp=z1; z1=z3; z3=tmp;}
    if (z2 > z3) { tmp=x2; x2=x3; x3=tmp; tmp=y2; y2=y3; y3=tmp; tmp=z2; z2=z3; z3=tmp;}


    if (//(x0 < 0.0) || (x1 < 0.0) || (x2 < 0.0) || (x3 < 0.0)
            //|| (y0 < 0.0) || (y1 < 0.0) || (y2 < 0.0) || (y3 < 0.0)
            (z0 < 0.0) || (z1 < 0.0) || (z2 < 0.0) || (z3 < 0.0)
            //|| (x0 >= (double)sizeX) || (x1 >= (double)sizeX) || (x2 >= (double)sizeX) || (x3 >= (double)sizeX)
            //|| (y0 >= (double)sizeY) || (y1 >= (double)sizeY) || (y2 >= (double)sizeY) || (y3 >= (double)sizeY)
            || (z0 >= (double)sizeZ) || (z1 >= (double)sizeZ) || (z2 >= (double)sizeZ) || (z3 >= (double)sizeZ)
            )
    {
        //printf("drop: %f, %f, %f, %f\n", z0, z1, z2, z3);
        return; //drop any triangles which extend over the boundaries
    }


    //calculate gradient
    m01x = (x1-x0)/(z1-z0);
    m01y = (y1-y0)/(z1-z0);
    m02x = (x2-x0)/(z2-z0);
    m02y = (y2-y0)/(z2-z0);
    m03x = (x3-x0)/(z3-z0);
    m03y = (y3-y0)/(z3-z0);
    m12x = (x2-x1)/(z2-z1);
    m12y = (y2-y1)/(z2-z1);
    m13x = (x3-x1)/(z3-z1);
    m13y = (y3-y1)/(z3-z1);
    m23x = (x3-x2)/(z3-z2);
    m23y = (y3-y2)/(z3-z2);

    y01 = y0;
    x01 = x0;
    y02 = y0;
    x02 = x0;
    y03 = y0;
    x03 = x0;
    y12 = y1;
    x12 = x1;
    y13 = y1;
    x13 = x1;
    y23 = y2;
    x23 = x2;

    //printf("z; %f, %f, %f, %f\n", z0, z1, z2, z3);

    // Draw triangles
    for (z = (int)z0; z < (int)z1; z++)
    {
        drawTriangle(&(pImage[sizeX*sizeY*z]), sizeX, sizeY, x01, y01, x02, y02, x03, y03, val);

        //printf("%f, %f, %f\n", x01, x02, x03);

        y01 += m01y;
        x01 += m01x;
        y02 += m02y;
        x02 += m02x;
        y03 += m03y;
        x03 += m03x;
    }


    for (z = (int)z1; z < (int)z2; z++)
    {
        drawTriangle(&(pImage[sizeX*sizeY*z]), sizeX, sizeY, x12, y12, x02, y02, x13, y13, val);
        drawTriangle(&(pImage[sizeX*sizeY*z]), sizeX, sizeY, x13, y13, x02, y02, x03, y03, val);

        //printf("%f, %f, %f, %f\n", x12, x13, x02, x03);

        y02 += m02y;
        x02 += m02x;
        y03 += m03y;
        x03 += m03x;
        y12 += m12y;
        x12 += m12x;
        y13 += m13y;
        x13 += m13x;
    }

    for (z = (int)z2; z < (int)z3; z++)
    {
        drawTriangle(&(pImage[sizeX*sizeY*z]), sizeX, sizeY, x13, y13, x23, y23, x03, y03, val);
        //printf("%f, %f, %f\n", x13, x23, x03);

        y13 += m13y;
        x13 += m13x;
        y23 += m23y;
        x23 += m23x;
        y03 += m03y;
        x03 += m03x;
    }

}
*/


static PyObject * drawTriang(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *data = 0;
    
    PyObject *odata =0;
    
    //PyArrayObject* adata;
    
    double x0;
    double y0;
    double x1;
    double y1;
    double x2;
    double y2;
    double val;

    //int * dims;
    int sizeX;
    int sizeY;
    
    static char *kwlist[] = {"data", "x0", "y0", "x1", "y1","x2", "y2", "val", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oddddddd", kwlist,
         &odata, &x0, &y0, &x1, &y1, &x2, &y2, &val))
        return NULL; 

    /* Do the calculations */ 
        
/*
    adata = (PyArrayObject *) PyArray_ContiguousFromObject(odata, PyArray_DOUBLE, 0, 1);
    if (adata == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad data");
      return NULL;
    }
*/

    if (!PyArray_Check(odata) | !PyArray_ISCONTIGUOUS(odata))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array");
        return NULL;
    }

    if (PyArray_NDIM(odata) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        return NULL;
    }

    //dims = PyArray_DIMS(odata);

    sizeX = PyArray_DIM(odata, 0);
    sizeY = PyArray_DIM(odata, 1);
    
    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];
    //printf('Dims: %d, %d', dims[0], dims[1]);
    
    data = (double*) PyArray_DATA(odata);

    drawTriangle(data, sizeX, sizeY, x0, y0, x1, y1, x2, y2, val);
    
    
    //Py_DECREF(adata)
    //return (PyObject*) adata;
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject * drawTriangles(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *data = 0;
    double *xs = 0;
    double *xs1 = 0;
    double *xs2 = 0;
    double *ys = 0;
    double *ys1 = 0;
    double *ys2 = 0;
    double *vals = 0;

    PyObject *odata =0;
    PyObject *oxs =0;
    PyObject *oys =0;
    PyObject *ovals =0;

    PyArrayObject *axs=0;
    PyArrayObject *ays=0;
    PyArrayObject *avals=0;

    //int * dims;
    int sizeX;
    int sizeY;
    int N;
    int i;

    static char *kwlist[] = {"data", "xs", "ys", "vals", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO", kwlist,
         &odata, &oxs, &oys, &ovals))
        return NULL;

    /* Do the calculations */

/*
    adata = (PyArrayObject *) PyArray_ContiguousFromObject(odata, PyArray_DOUBLE, 0, 1);
    if (adata == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad data");
      return NULL;
    }
*/

    if (!PyArray_Check(odata) || !PyArray_ISCONTIGUOUS(odata))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array");
        return NULL;
    }

    if (PyArray_NDIM(odata) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        return NULL;
    }

    axs = (PyArrayObject *)PyArray_FROM_OTF(oxs, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (axs == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad xs");
        return NULL;
    }

    ays = (PyArrayObject *)PyArray_FROM_OTF(oys, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (ays == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad ys");
        Py_DECREF(axs);
        return NULL;
    }

    avals = (PyArrayObject *)PyArray_FROM_OTF(ovals, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (avals == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad vals");
        Py_DECREF(axs);
        Py_DECREF(ays);
        return NULL;
    }

    //dims = PyArray_DIMS(odata);

    sizeX = PyArray_DIM(odata, 0);
    sizeY = PyArray_DIM(odata, 1);

    N = PyArray_DIM(axs, 0);

    //printf("dims: %d, %d\n", N, PyArray_DIM(axs, 1));

    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];
    //printf('Dims: %d, %d', dims[0], dims[1]);

    data = (double*) PyArray_DATA(odata);
    xs = (double*) PyArray_DATA(axs);
    ys = (double*) PyArray_DATA(ays);
    vals = (double*) PyArray_DATA(avals);

    xs1 = xs + N;
    xs2 = xs1 + N;
    ys1 = ys + N;
    ys2 = ys1 + N;

    Py_BEGIN_ALLOW_THREADS;

    for (i=0;i < N; i++)
    {
        //printf("i: %d\n", i);
        drawTriangle(data, sizeX, sizeY, xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], *vals);
        xs +=3;
        ys +=3;
/*
        xs1 ++;
        ys1 ++;
        xs2 ++;
        ys2 ++;
*/
        vals ++;
        //drawTriangle(data, sizeX, sizeY, *(double*)PyArray_GETPTR2(axs, i, 0), *(double*)PyArray_GETPTR2(ays, i, 0), *(double*)PyArray_GETPTR2(axs, i, 1), *(double*)PyArray_GETPTR2(ays, i, 1), *(double*)PyArray_GETPTR2(axs, i, 2), *(double*)PyArray_GETPTR2(ays, i, 2), vals[i]);
    }

    Py_END_ALLOW_THREADS;


    //Py_DECREF(adata)
    //return (PyObject*) adata;
    Py_DECREF(axs);
    Py_DECREF(ays);
    Py_DECREF(avals);
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject * drawTetrahedra(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *data = 0;
    double *xs = 0;
    //double *xs1 = 0;
    //double *xs2 = 0;
    //double *xs3 = 0;
    double *ys = 0;
    //double *ys1 = 0;
    //double *ys2 = 0;
    //double *ys3 = 0;
    double *zs = 0;
    //double *zs1 = 0;
    //double *zs2 = 0;
    //double *zs3 = 0;
    double *vals = 0;

    PyObject *odata =0;
    PyObject *oxs =0;
    PyObject *oys =0;
    PyObject *ozs =0;
    PyObject *ovals =0;

    PyArrayObject *axs=0;
    PyArrayObject *ays=0;
    PyArrayObject *azs=0;
    PyArrayObject *avals=0;

    //int * dims;
    int sizeX;
    int sizeY;
    int sizeZ;
    int N;
    int i;

    static char *kwlist[] = {"data", "xs", "ys", "zs", "vals", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOO", kwlist,
         &odata, &oxs, &oys, &ozs, &ovals))
        return NULL;

    /* Do the calculations */

/*
    adata = (PyArrayObject *) PyArray_ContiguousFromObject(odata, PyArray_DOUBLE, 0, 1);
    if (adata == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad data");
      return NULL;
    }
*/

    if (!PyArray_Check(odata) || !PyArray_ISFORTRAN(odata))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a fortran contiguous numpy array");
        return NULL;
    }

    if (PyArray_NDIM(odata) != 3)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 3 dimensional array");
        return NULL;
    }

    axs = (PyArrayObject *)PyArray_FROM_OTF(oxs, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (axs == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad xs");
        return NULL;
    }

    ays = (PyArrayObject *)PyArray_FROM_OTF(oys, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (ays == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad ys");
        Py_DECREF(axs);
        return NULL;
    }

    azs = (PyArrayObject *)PyArray_FROM_OTF(ozs, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (azs == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad zs");
        Py_DECREF(axs);
        Py_DECREF(ays);
        return NULL;
    }

    avals = (PyArrayObject *)PyArray_FROM_OTF(ovals, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (avals == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad vals");
        Py_DECREF(axs);
        Py_DECREF(ays);
        Py_DECREF(azs);
        return NULL;
    }

    //dims = PyArray_DIMS(odata);

    sizeX = PyArray_DIM(odata, 1);
    sizeY = PyArray_DIM(odata, 0);
    sizeZ = PyArray_DIM(odata, 2);

    N = PyArray_DIM(axs, 0);

    //printf("size: %d, %d, %d\n", sizeX, sizeY,sizeZ);

    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];
    //printf('Dims: %d, %d', dims[0], dims[1]);

    data = (double*) PyArray_DATA(odata);
    xs = (double*) PyArray_DATA(axs);
    ys = (double*) PyArray_DATA(ays);
    zs = (double*) PyArray_DATA(azs);
    vals = (double*) PyArray_DATA(avals);

    //xs1 = xs + N;
    //xs2 = xs1 + N;
    //ys1 = ys + N;
    //ys2 = ys1 + N;

    Py_BEGIN_ALLOW_THREADS;

    for (i=0;i < N; i++)
    {
        //printf("i: %d\n", i);
        drawTetrahedron(data, sizeX, sizeY, sizeZ, xs[0], ys[0], zs[0], xs[1], ys[1], zs[1],
                xs[2], ys[2], zs[2], xs[3], ys[3], zs[3], *vals);
        xs +=4;
        ys +=4;
        zs +=4;
/*
        xs1 ++;
        ys1 ++;
        xs2 ++;
        ys2 ++;
*/
        vals ++;
        //drawTriangle(data, sizeX, sizeY, *(double*)PyArray_GETPTR2(axs, i, 0), *(double*)PyArray_GETPTR2(ays, i, 0), *(double*)PyArray_GETPTR2(axs, i, 1), *(double*)PyArray_GETPTR2(ays, i, 1), *(double*)PyArray_GETPTR2(axs, i, 2), *(double*)PyArray_GETPTR2(ays, i, 2), vals[i]);
    }

    Py_END_ALLOW_THREADS;


    //Py_DECREF(adata)
    //return (PyObject*) adata;
    Py_DECREF(axs);
    Py_DECREF(ays);
    Py_DECREF(azs);
    Py_DECREF(avals);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * PyTetAndDraw(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *oPositions =0;
    PyArrayObject* aPositions;

    PyObject *odata =0;

    double *data;

    int dim, nPositions, sizeX, sizeY, sizeZ;
    int iErr;
    BOOL calc_area;

    static char *kwlist[] = {"positions", "data", "calcArea", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|I", kwlist,
         &oPositions, &odata, &calc_area))
        return NULL;

    if (!PyArray_Check(odata) || !PyArray_ISFORTRAN(odata))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a fortran contiguous numpy array");
        return NULL;
    }

    if (PyArray_NDIM(odata) != 3)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 3 dimensional array");
        return NULL;
    }

    aPositions = (PyArrayObject *) PyArray_ContiguousFromObject(oPositions, PyArray_DOUBLE, 2, 2);
    if (aPositions == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad position data");
      return NULL;
    }


    dim = PyArray_DIM(aPositions, 1);
    nPositions = PyArray_DIM(aPositions, 0);

    if (dim != 3)
    {
      PyErr_Format(PyExc_RuntimeError, "Expecting an Nx3 array of point positions");
      Py_DECREF(aPositions);
      return NULL;
    }

    sizeX = PyArray_DIM(odata, 1);
    sizeY = PyArray_DIM(odata, 0);
    sizeZ = PyArray_DIM(odata, 2);

    data = (double*) PyArray_DATA(odata);

    iErr = tetAndDraw((coordT *)PyArray_DATA(aPositions), nPositions, data, sizeX, sizeY, sizeZ, calc_area);
    if (iErr)
    {
      PyErr_Format(PyExc_RuntimeError, "QHull error");
      Py_DECREF(aPositions);
      return NULL;
    }

    Py_DECREF(aPositions);

    return Py_None;
}




static PyMethodDef triRendMethods[] = {
    {"drawTriang",  (PyCFunction)drawTriang, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"drawTriangles",  (PyCFunction)drawTriangles, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"drawTetrahedra",  (PyCFunction)drawTetrahedra, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"RenderTetrahedra",  (PyCFunction)PyTetAndDraw, METH_VARARGS | METH_KEYWORDS,
    ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};



#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "triRend",     /* m_name */
        "Render triangles (or tetrahedra)",  /* m_doc */
        -1,                  /* m_size */
        triRendMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_triRend(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else
PyMODINIT_FUNC inittriRend(void)
{
    PyObject *m;

    m = Py_InitModule("triRend", triRendMethods);
    import_array();
}
#endif
