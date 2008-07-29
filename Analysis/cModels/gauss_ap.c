#include "Python.h"
#include <complex.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

#define MIN(a, b) ((a<b) ? a : b) 
#define MAX(a, b) ((a>b) ? a : b)

#define LITTLEENDIAN


//Fast approximation to exponential
static union
{
  double d;
  struct
  {

#ifdef LITTLEENDIAN
    int j, i;
#else
    int i, j;
#endif
  } n;
} eco;

#define EXP_A (1048576/M_LN2) /* use 1512775 for integer version */
#define EXP_C 60801 /* see text for choice of c values */
#define EXP(y) (eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), eco.d)

//end eponential approx

static PyObject * genGauss(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *res = 0;  
    int ix,iy; 
    int size[2];
    
    PyObject *oX =0;
    PyObject *oY=0;
    
    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    
    PyArrayObject* out;
    
    double *pXvals;
    double *pYvals;
    
    /*parameters*/
    double A = 1;
    double x0 = 0;
    double y0 = 0;
    double sigma = 1;
    double b = 0;
    double b_x = 0;
    double b_y = 0;

    /*End paramters*/

    double ts2;
    double byY;

      
    
    static char *kwlist[] = {"X", "Y", "A","x0", "y0","sigma","b","b_x","b_y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|ddddddd", kwlist, 
         &oX, &oY, &A, &x0, &y0, &sigma, &b, &b_x, &b_y))
        return NULL; 

    /* Do the calculations */ 
        
    Xvals = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_DOUBLE, 0, 1);
    if (Xvals == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Bad X");   
      return NULL;
    }
    
    Yvals = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_DOUBLE, 0, 1);
    if (Yvals == NULL)
    {
        Py_DECREF(Xvals);
        PyErr_Format(PyExc_RuntimeError, "Bad Y");
        return NULL;
    }
    
    
    
    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    
    
    size[0] = PyArray_Size((PyObject*)Xvals);
    size[1] = PyArray_Size((PyObject*)Yvals);
        
    out = (PyArrayObject*) PyArray_FromDims(2,size,PyArray_DOUBLE);
    
    //fix strides
    out->strides[0] = sizeof(double);
    out->strides[1] = sizeof(double)*size[0];
    
    res = (double*) out->data;
    
    ts2 = 2*sigma*sigma;
        
    for (iy = 0; iy < size[1]; iy++)
      {            
	byY = b_y*pYvals[iy] + b;
	for (ix = 0; ix < size[0]; ix++)
	  {
	    *res = A*exp(-(((pXvals[ix] - x0) * (pXvals[ix] - x0)) + ((pYvals[iy]-y0) * (pYvals[iy]-y0)))/ts2) + b_x*pXvals[ix] + byY;
	    //*res = 1.0;
	    res++;
            
	  }
        
      }
    
    
    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    
    return (PyObject*) out;
}


//same as above but using dodgy exponential
static PyObject * genGaussF(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *res = 0;  
    int ix,iy; 
    int size[2];
    
    PyObject *oX =0;
    PyObject *oY=0;
    
    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    
    PyArrayObject* out;
    
    double *pXvals;
    double *pYvals;
    
    /*parameters*/
    double A = 1;
    double x0 = 0;
    double y0 = 0;
    double sigma = 1;
    double b = 0;
    double b_x = 0;
    double b_y = 0;

    /*End paramters*/

    double ts2;
    double byY;

    //double g_;

      
    
    static char *kwlist[] = {"X", "Y", "A","x0", "y0","sigma","b","b_x","b_y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|ddddddd", kwlist, 
         &oX, &oY, &A, &x0, &y0, &sigma, &b, &b_x, &b_y))
        return NULL; 

    /* Do the calculations */ 
        
    Xvals = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_DOUBLE, 0, 1);
    if (Xvals == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Bad X");   
      return NULL;
    }
    
    Yvals = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_DOUBLE, 0, 1);
    if (Yvals == NULL)
    {
        Py_DECREF(Xvals);
        PyErr_Format(PyExc_RuntimeError, "Bad Y");
        return NULL;
    }
    
    
    
    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    
    
    size[0] = PyArray_Size((PyObject*)Xvals);
    size[1] = PyArray_Size((PyObject*)Yvals);
        
    out = (PyArrayObject*) PyArray_FromDims(2,size,PyArray_DOUBLE);
    
    //fix strides
    out->strides[0] = sizeof(double);
    out->strides[1] = sizeof(double)*size[0];
    
    res = (double*) out->data;
    
    ts2 = 2*sigma*sigma;
        
    for (iy = 0; iy < size[1]; iy++)
      {            
	byY = b_y*pYvals[iy] + b;
	for (ix = 0; ix < size[0]; ix++)
	  {
	    *res = A*EXP(-(((pXvals[ix] - x0) * (pXvals[ix] - x0)) + ((pYvals[iy]-y0) * (pYvals[iy]-y0)))/ts2) + b_x*pXvals[ix] + byY;
	    //*res = 1.0;
	    res++;
            
	  }
        
      }
    
    
    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    
    return (PyObject*) out;
}


//generate jacobian
static PyObject * genGaussFJac(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *res = 0;  
    int ix,iy; 
    int size[3];
    
    PyObject *oX =0;
    PyObject *oY=0;
    
    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    
    PyArrayObject* out;
    
    double *pXvals;
    double *pYvals;
    
    /*parameters*/
    double A = 1;
    double x0 = 0;
    double y0 = 0;
    double sigma = 1;
    double b = 0;
    double b_x = 0;
    double b_y = 0;

    /*End paramters*/

    double ts2;
    double byY;
    double A_s2;
    double g_;

      
    
    static char *kwlist[] = {"X", "Y", "A","x0", "y0","sigma","b","b_x","b_y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|ddddddd", kwlist, 
         &oX, &oY, &A, &x0, &y0, &sigma, &b, &b_x, &b_y))
        return NULL; 

    /* Do the calculations */ 
        
    Xvals = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_DOUBLE, 0, 1);
    if (Xvals == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Bad X");   
      return NULL;
    }
    
    Yvals = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_DOUBLE, 0, 1);
    if (Yvals == NULL)
    {
        Py_DECREF(Xvals);
        PyErr_Format(PyExc_RuntimeError, "Bad Y");
        return NULL;
    }
    
    
    
    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    
    size[1] = PyArray_Size((PyObject*)Xvals);
    size[2] = PyArray_Size((PyObject*)Yvals);
    size[0] = 7;
        
    out = (PyArrayObject*) PyArray_FromDims(3,size,PyArray_DOUBLE);
    
    //fix strides
    out->strides[0] = sizeof(double);
    out->strides[1] = sizeof(double)*size[0];
    out->strides[2] = sizeof(double)*size[0]*size[1];
    
    res = (double*) out->data;
    
    ts2 = 1/(2*sigma*sigma);
    A_s2 = A/(sigma*sigma);
        
    for (ix = 0; ix < size[1]; ix++)
      {            
	//byY = b_y*pYvals[iy] + b;
	for (iy = 0; iy < size[2]; iy++)
	  {
	    g_ = EXP(-(((pXvals[ix] - x0) * (pXvals[ix] - x0)) + ((pYvals[iy]-y0) * (pYvals[iy]-y0)))*ts2);
	    *res = g_; // d/dA
	    res++;
	    g_ *= A_s2;
	    *res = (pXvals[ix] - x0)*g_; // d/dx0
	    res++;
	    *res = (pYvals[iy] - y0)*g_; // d/dx0
	    res++;
	    *res = (((pXvals[ix] - x0) * (pXvals[ix] - x0)) + ((pYvals[iy]-y0) * (pYvals[iy]-y0)))*g_/sigma; // d/dsigma
	    res++;
	    *res = 1.0;
	    res++;
	    *res = pXvals[ix];
	    res++;
	    *res = pYvals[iy];
	    //*res = 1.0;
	    res++;
            
	  }
        
      }
    
    
    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    
    return (PyObject*) out;
}



//Double Gaussian
static PyObject * genGaussA(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *res = 0;  
    int ix,iy; 
    int size[2];
    
    PyObject *oX =0;
    PyObject *oY=0;
    
    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    
    PyArrayObject* out;
    
    double *pXvals;
    double *pYvals;
    
    /*parameters*/
    double A = 1;
    double x0 = 0;
    double y0 = 0;
    double sigma_x = 1;
    double sigma_y = 1;
    double b = 0;
    double b_x = 0;
    double b_y = 0;

    /*End paramters*/

    double tsx2;
    double tsy2;
    double byY;

      
    
    static char *kwlist[] = {"X", "Y", "A","x0", "y0","sigma_x", "sigma_y","b","b_x","b_y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|dddddddd", kwlist, 
         &oX, &oY, &A, &x0, &y0, &sigma_x,  &sigma_y,&b, &b_x, &b_y))
        return NULL; 

    /* Do the calculations */ 
        
    Xvals = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_DOUBLE, 0, 1);
    if (Xvals == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Bad X");   
      return NULL;
    }
    
    Yvals = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_DOUBLE, 0, 1);
    if (Yvals == NULL)
    {
        Py_DECREF(Xvals);
        PyErr_Format(PyExc_RuntimeError, "Bad Y");
        return NULL;
    }
    
    
    
    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    
    
    size[0] = PyArray_Size((PyObject*)Xvals);
    size[1] = PyArray_Size((PyObject*)Yvals);
        
    out = (PyArrayObject*) PyArray_FromDims(2,size,PyArray_DOUBLE);
    
    //fix strides
    out->strides[0] = sizeof(double);
    out->strides[1] = sizeof(double)*size[0];
    
    res = (double*) out->data;
    
    tsx2 = 2*sigma_x*sigma_x;
    tsy2 = 2*sigma_y*sigma_y;
        
    for (iy = 0; iy < size[1]; iy++)
      {            
	byY = b_y*pYvals[iy] + b;
	for (ix = 0; ix < size[0]; ix++)
	  {
	    *res = A*exp(-(((pXvals[ix] - x0) * (pXvals[ix] - x0))/tsx2 + ((pYvals[iy]-y0) * (pYvals[iy]-y0))/tsy2)) + b_x*pXvals[ix] + byY;
	    //*res = 1.0;
	    res++;
            
	  }
        
      }
    
    
    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    
    return (PyObject*) out;
}


//same as above but using dodgy exponential
static PyObject * genGaussAF(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *res = 0;  
    int ix,iy; 
    int size[2];
    
    PyObject *oX =0;
    PyObject *oY=0;
    
    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    
    PyArrayObject* out;
    
    double *pXvals;
    double *pYvals;
    
    /*parameters*/
    double A = 1;
    double x0 = 0;
    double y0 = 0;
    double sigma_x = 1;
    double sigma_y = 1;
    double b = 0;
    double b_x = 0;
    double b_y = 0;

    /*End paramters*/

    double tsx2;
    double tsy2;
    double byY;

      
    
    static char *kwlist[] = {"X", "Y", "A","x0", "y0","sigma_x", "sigma_y","b","b_x","b_y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|dddddddd", kwlist, 
         &oX, &oY, &A, &x0, &y0, &sigma_x,  &sigma_y,&b, &b_x, &b_y))
        return NULL; 

    /* Do the calculations */ 
        
    Xvals = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_DOUBLE, 0, 1);
    if (Xvals == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Bad X");   
      return NULL;
    }
    
    Yvals = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_DOUBLE, 0, 1);
    if (Yvals == NULL)
    {
        Py_DECREF(Xvals);
        PyErr_Format(PyExc_RuntimeError, "Bad Y");
        return NULL;
    }
    
    
    
    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    
    
    size[0] = PyArray_Size((PyObject*)Xvals);
    size[1] = PyArray_Size((PyObject*)Yvals);
        
    out = (PyArrayObject*) PyArray_FromDims(2,size,PyArray_DOUBLE);
    
    //fix strides
    out->strides[0] = sizeof(double);
    out->strides[1] = sizeof(double)*size[0];
    
    res = (double*) out->data;
    
    tsx2 = 2*sigma_x*sigma_x;
    tsy2 = 2*sigma_y*sigma_y;
    
        
    for (iy = 0; iy < size[1]; iy++)
      {            
	byY = b_y*pYvals[iy] + b;
	for (ix = 0; ix < size[0]; ix++)
	  {
	    *res = A*EXP(-(((pXvals[ix] - x0) * (pXvals[ix] - x0))/tsx2 + ((pYvals[iy]-y0) * (pYvals[iy]-y0))/tsy2)) + b_x*pXvals[ix] + byY;
	    //*res = 1.0;
	    res++;
            
	  }
        
      }
    
    
    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    
    return (PyObject*) out;
}

static PyMethodDef gauss_appMethods[] = {
    {"genGauss",  genGauss, METH_VARARGS | METH_KEYWORDS,
    "Generate a (fast) Gaussian.\n. Arguments are: 'X', 'Y', 'A'=1,'x0'=0, 'y0'=0,sigma=0,b=0,b_x=0,b_y=0"},
    {"genGaussF",  genGaussF, METH_VARARGS | METH_KEYWORDS,
    "Generate a (fast) Gaussian using dodgy exponential approx.\n. Arguments are: 'X', 'Y', 'A'=1,'x0'=0, 'y0'=0,sigma=0,b=0,b_x=0,b_y=0"},
    {"genGaussFJac",  genGaussFJac, METH_VARARGS | METH_KEYWORDS,
    "Generate jacobian for Gaussian using dodgy exponential approx.\n. Arguments are: 'X', 'Y', 'A'=1,'x0'=0, 'y0'=0,sigma=0,b=0,b_x=0,b_y=0"},
    {"genGaussA",  genGaussA, METH_VARARGS | METH_KEYWORDS,
    "Generate a (fast) astigmatic Gaussian.\n. Arguments are: 'X', 'Y', 'A'=1,'x0'=0, 'y0'=0,sigma_x=1, sigma_y = 1,b=0,b_x=0,b_y=0"},
    {"genGaussAF",  genGaussAF, METH_VARARGS | METH_KEYWORDS,
    "Generate a (fast) astigmatic Gaussian using dodgy exponential approx.\n. Arguments are: 'X', 'Y', 'A'=1,'x0'=0, 'y0'=0,'sigma_x'=1, 'sigma_y'=1,b=0,b_x=0,b_y=0"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initgauss_app(void)
{
    PyObject *m;

    m = Py_InitModule("gauss_app", gauss_appMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
