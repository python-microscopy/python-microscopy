#include "Python.h"
#include <complex.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>


static PyObject * distanceHistogram(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *res = 0;
    int i1,i2;
    //int size[2];

    int x1_len;
    int x2_len;
    int outDimensions[1];
    int id, j;
    
    PyObject *ox1 =0;
    PyObject *oy1=0;
    PyObject *ox2 =0;
    PyObject *oy2=0;

    
    PyArrayObject* ax1;
    PyArrayObject* ay1;
    PyArrayObject* ax2;
    PyArrayObject* ay2;
    
    PyArrayObject* out;
    
    double *px1;
    double *px2;
    double *py1;
    double *py2;
    double *px2o;
    double *py2o;

    double d, dx, dy;
    
    /*parameters*/
    int nBins = 1000;
    double binSize = 1;

    /*End paramters*/

    
      
    
    static char *kwlist[] = {"x1", "y1","x2", "y2","nBins","binSize", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|id", kwlist,
         &ox1, &oy1, &ox2, &oy2, &nBins, &binSize))
        return NULL; 

    /* Do the calculations */ 
        
    ax1 = (PyArrayObject *) PyArray_ContiguousFromObject(ox1, PyArray_DOUBLE, 0, 1);
    if (ax1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x1");
      return NULL;
    }

    ay1 = (PyArrayObject *) PyArray_ContiguousFromObject(oy1, PyArray_DOUBLE, 0, 1);
    if (ay1 == NULL)
    {
      Py_DECREF(ax1);
      PyErr_Format(PyExc_RuntimeError, "Bad y1");
      return NULL;
    }

    ax2 = (PyArrayObject *) PyArray_ContiguousFromObject(ox2, PyArray_DOUBLE, 0, 1);
    if (ax2 == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      PyErr_Format(PyExc_RuntimeError, "Bad x2");
      return NULL;
    }

    ay2 = (PyArrayObject *) PyArray_ContiguousFromObject(oy2, PyArray_DOUBLE, 0, 1);
    if (ay2 == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      Py_DECREF(ax2);
      PyErr_Format(PyExc_RuntimeError, "Bad y2");
      return NULL;
    }
      
    
    px1 = (double*)ax1->data;
    py1 = (double*)ay1->data;
    px2 = (double*)ax2->data;
    py2 = (double*)ay2->data;
    
    
    x1_len = PyArray_Size((PyObject*)ax1);
    x2_len = PyArray_Size((PyObject*)ax2);

    outDimensions[0] = nBins;
        
    out = (PyArrayObject*) PyArray_FromDims(1,outDimensions,PyArray_DOUBLE);
    if (out == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      Py_DECREF(ax2);
      Py_DECREF(ay2);
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      return NULL;
    }
    
    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];
    
    res = (double*) out->data;

    //Initialise our histogram
    for (j =0; j < nBins; j++)
    {
        res[j] = 0;
    }
    
        
    px2o = px2;
    py2o = py2;

    for (i1 = 0; i1 < x1_len; i1++)
      {            
	for (i2 = 0; i2 < x2_len; i2++)
	  {
            dx = *px1 - *px2;
            dy = *py1 - *py2;
            d = sqrt(dx*dx + dy*dy);

            id = (int)floor(d/binSize);

            if (id < nBins) res[id] += 1;

            px2++;
            py2++;
            
	  }
        px1++;
        py1++;

        //reset inner pointers
        px2 = px2o;
        py2 = py2o;
      }
    
    
    Py_DECREF(ax1);
    Py_DECREF(ay1);
    Py_DECREF(ax2);
    Py_DECREF(ay2);
    
    return (PyObject*) out;
}




static PyMethodDef distHistMethods[] = {
    {"distanceHistogram",  distanceHistogram, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initdistHist(void)
{
    PyObject *m;

    m = Py_InitModule("distHist", distHistMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
