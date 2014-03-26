#include "gapp.h"


static PyObject * NRFilter(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *res = 0;  
    int i,j,lenx, numP, lut_size; 
    npy_intp size[1];
    
    PyObject *oX =0;
    PyObject *oY=0;
    PyObject *oI=0;
    PyObject *oLUT=0;
    
    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    PyArrayObject* Ivals;
    PyArrayObject* LUTvals;
    
    PyArrayObject* out;
    
    int *pXvals;
    int *pYvals;
    double *pIvals;
    double *pLUTvals;

    int xi, yi, dx, dy, r2;
      
    
    static char *kwlist[] = {"X", "Y", "I","LUT", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOO", kwlist, 
         &oX, &oY, &oI, &oLUT))
        return NULL; 

    /* Get values into a cormat we understand */ 
        
    Xvals = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_INT, 0, 1);
    if (Xvals == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Bad X");   
      goto abort;
    }
    
    Yvals = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_INT, 0, 1);
    if (Yvals == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Bad Y");
        goto abort;
    }
    
    Ivals = (PyArrayObject *) PyArray_ContiguousFromObject(oI, PyArray_DOUBLE, 0, 1);
    if (Ivals == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Bad I");
        goto abort;
    }

    LUTvals = (PyArrayObject *) PyArray_ContiguousFromObject(oLUT, PyArray_DOUBLE, 0, 1);
    if (LUTvals == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Bad LUT");
        goto abort;
    }    
    
    pXvals = (int*)Xvals->data;
    pYvals = (int*)Yvals->data;
    pIvals = (double*)Ivals->data;
    pLUTvals = (double*)LUTvals->data;
    
    
    size[0] = PyArray_Size((PyObject*)Xvals);
    lenx = size[0];

    lut_size = PyArray_Size((PyObject*)LUTvals);
    
    /* Allocate memory for result */
    out = (PyArrayObject*) PyArray_New(&PyArray_Type, 1,size,NPY_DOUBLE, NULL, NULL, 0, 1, NULL);
    if (out == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to allocate memory");
        goto abort;    
    }
    
    
    res = (double*) PyArray_DATA(out);
        
    for (i = 0; i < lenx; i++)
      {
      	*res = 0;
      	xi = pXvals[i];
      	yi = pYvals[i];            
	
		for (j = 0; j < numP; j++)
	  	{
	  		dx = xi - pXvals[j];
          	dy = yi = pYvals[j];

          	r2 = dx*dx + dy*dy;
          	r2 = MIN(r2, (lut_size-1));
        	
	    	*res += pLUTvals[r2]*pIvals[j];	    
	  	}
        
        res++;
      }
    
    
    Py_XDECREF(Xvals);
    Py_XDECREF(Yvals);
    Py_XDECREF(Ivals);
    Py_XDECREF(LUTvals);
    
    return (PyObject*) out;

abort:
    Py_XDECREF(Xvals);
    Py_XDECREF(Yvals);
    Py_XDECREF(Ivals);
    Py_XDECREF(LUTvals);
    
    return NULL;
}