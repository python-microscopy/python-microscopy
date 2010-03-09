#include "Python.h"
#include <complex.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

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


static PyObject * drawTriang(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *data = 0;
    
    PyObject *odata =0;
    
    PyArrayObject* adata;
    
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

    axs = PyArray_FROM_OTF(oxs, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (axs == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad xs");
        return NULL;
    }

    ays = PyArray_FROM_OTF(oys, NPY_DOUBLE, NPY_CONTIGUOUS);
    if (ays == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "bad ys");
        Py_DECREF(axs);
        return NULL;
    }

    avals = PyArray_FROM_OTF(ovals, NPY_DOUBLE, NPY_CONTIGUOUS);
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




static PyMethodDef triRendMethods[] = {
    {"drawTriang",  drawTriang, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
    {"drawTriangles",  drawTriangles, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC inittriRend(void)
{
    PyObject *m;

    m = Py_InitModule("triRend", triRendMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
