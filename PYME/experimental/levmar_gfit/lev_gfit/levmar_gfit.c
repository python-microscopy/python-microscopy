#include "Python.h"
//#include <complex.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include "lm.h"
#include <stdio.h>
#include <stdlib.h>


typedef struct gFitData
{
  int sx; //size of x array
  int sy; //size of y array

  double *x;
  double *y;

  double *imageData;
  double *weights;
} gFitData;


struct params
{
  double A;
  double x0;
  double y0;
  double sigma;
  double b;
  double b_x;
  double b_y;
};

typedef union parameters
{
  double pvals[7];
  struct params p;
} parameters;


/*parameters
#define A p[0]
#define x0 p[1]
#define y0 p[2]
#define sigma p[3]
#define b p[4]
#define b_x p[5]
#define b_y p[6]

End paramters*/


static void modGauss(double *p, double *out, int m, int n, void *data) 
{
    gFitData *fdata; 
    parameters *p_;

    double ts2;
    double byY;
    double dy2;
    
    double *imData;
    double *weights;
    
    int ix,iy;

    p_ = (parameters*)p;
    fdata = (gFitData *)data;
    imData = fdata->imageData;
    weights = fdata->weights;

    //printf("in modGauss\n");

    ts2 = 2*p_->p.sigma*p_->p.sigma;
        
    for (iy = 0; iy < fdata->sy; iy++)
      {            
	byY =p_->p.b_y*(fdata->y[iy]- p_->p.y0) + p_->p.b;
	dy2 = ((fdata->y[iy]-p_->p.y0) * (fdata->y[iy]-p_->p.y0));
	for (ix = 0; ix < fdata->sx; ix++)
	  {
	    *out = *weights*(*imData - (p_->p.A*exp(-(((fdata->x[ix] - p_->p.x0) * (fdata->x[ix] - p_->p.x0)) + dy2)/ts2) + p_->p.b_x*(fdata->x[ix] - p_->p.x0) + byY));
	    //*res = 1.0;
	    out++;
	    imData++;
	    weights++;
            
	  }
        
      }
    //printf("end modGauss\n");
}

static void jacGauss(double *p, double *res, int m, int n, void *data) 
{
    gFitData *fdata;
    parameters *p_;
    
    //double dy2;

     double ts2;
    //double byY;
    double A_s2;
    double g_;

    double *weights;
    
    int ix,iy;

    fdata = (gFitData *)data;
    p_ = (parameters*)p;

    weights = fdata->weights;
    
    //printf("in jacGauss\n");

        ts2 = 1.0/(2*p_->p.sigma*p_->p.sigma);
    A_s2 = p_->p.A/(p_->p.sigma*p_->p.sigma);
        
    for (iy = 0; iy < fdata->sy ; iy++)
      {            
	//byY = b_y*pYvals[iy] + b;
	for (ix = 0; ix < fdata->sx; ix++)
	  {
	    g_ = -*weights*exp(-(((fdata->x[ix] - p_->p.x0) * (fdata->x[ix] -p_->p.x0)) + ((fdata->y[iy]-p_->p.y0) * (fdata->y[iy]-p_->p.y0)))*ts2);
	    *res = g_; // d/dA
	    res++;
	    g_ *= A_s2;
	    *res = (fdata->x[ix] - p_->p.x0)*g_; // d/dx0
	    res++;
	    *res = (fdata->y[iy] - p_->p.y0)*g_; // d/dx0
	    res++;
	    *res = (((fdata->x[ix] - p_->p.x0) * (fdata->x[ix] - p_->p.x0)) + ((fdata->y[iy]-p_->p.y0) * (fdata->y[iy]-p_->p.y0)))*g_/p_->p.sigma; // d/dsigma
	    res++;
	    *res = -*weights;
	    res++;
	    *res = -*weights*fdata->x[ix];
	    res++;
	    *res = -*weights*fdata->y[iy];
	    //*res = 1.0;
	    res++;
	    weights ++;
            
	  }
        
      }
        
    //printf("end jacGauss\n");
}



static PyObject * fitGauss(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *x_ = 0;  
    npy_intp m[1];
    int n=0;
    int i;
    npy_intp covSize[2];
        
    PyObject *oX =0;
    PyObject *oY=0;
    PyObject *oD=0;
    PyObject *oP=0;
    PyObject *oW=0;
    
    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    PyArrayObject* Data;
    PyArrayObject* p_;
    PyArrayObject* Weights;

    PyArrayObject* outp;
    PyArrayObject* outcov;

    parameters *p;
    gFitData fitData;
        
    int ret;
    
    double opts[LM_OPTS_SZ], info[LM_INFO_SZ];

    opts[0]=LM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-20;
    opts[4]=LM_DIFF_DELTA; // relevant only if the finite difference jacobian version is used

    m[0] = 7;
    outp = (PyArrayObject*) PyArray_SimpleNew(1,m,PyArray_DOUBLE);
    p = (parameters*)outp->data;
    
    
    //static char *kwlist[] = {"X", "Y", "D", "A","x0", "y0","sigma","b","b_x","b_y", NULL};
    static char *kwlist[] = {"p","X", "Y", "D", "W", NULL};
    
    //if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO", kwlist, 
    //				     &oP,&oX, &oY, &oD, &p->p.A, &p->p.x0, &p->p.y0, &p->p.sigma, &p->p.b, &p->p.b_x, &p->p.b_y))
    //    return NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOO", kwlist, &oP,&oX, &oY, &oD, &oW))
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
    
    Data = (PyArrayObject *) PyArray_ContiguousFromObject(oD, PyArray_DOUBLE, 0, 1);
    if (Data == NULL)
    {
        Py_DECREF(Xvals);
	Py_DECREF(Yvals);
        PyErr_Format(PyExc_RuntimeError, "Bad data");
        return NULL;
    }

    Weights = (PyArrayObject *) PyArray_ContiguousFromObject(oW, PyArray_DOUBLE, 0, 1);
    if (Weights == NULL)
    {
        Py_DECREF(Xvals);
	Py_DECREF(Yvals);
	Py_DECREF(Data);
        PyErr_Format(PyExc_RuntimeError, "Bad weights");
        return NULL;
    }

    p_ = (PyArrayObject *) PyArray_ContiguousFromObject(oP, PyArray_DOUBLE, 0, 1);
    if (Data == NULL)
    {
        Py_DECREF(Xvals);
	Py_DECREF(Yvals);
	Py_DECREF(Data);
	Py_DECREF(Weights);
        PyErr_Format(PyExc_RuntimeError, "Bad parameters");
        return NULL;
    }

    if (PyArray_Size((PyObject*)p_) != 7)
    {
        Py_DECREF(Xvals);
	Py_DECREF(Yvals);
	Py_DECREF(Data);
	Py_DECREF(Weights);
	Py_DECREF(p_);
        PyErr_Format(PyExc_RuntimeError, "size of parameters is not 7");
        return NULL;
    }

    memcpy(p, p_->data, 7*sizeof(double));
    //printf("pStart: %f\n", p->p.A);

    fitData.x = (double*)Xvals->data;
    fitData.y = (double*)Yvals->data;
    fitData.imageData = (double*)Data->data;
    fitData.weights = (double*)Weights->data;
    
    fitData.sx = PyArray_Size((PyObject*)Xvals);
    fitData.sy = PyArray_Size((PyObject*)Yvals);
    
    n = fitData.sx*fitData.sy;
    

    if (PyArray_Size((PyObject*)Data) != n)
    {
        Py_DECREF(Xvals);
	Py_DECREF(Yvals);
	Py_DECREF(Data);
	Py_DECREF(Weights);
	Py_DECREF(p_);
        PyErr_Format(PyExc_RuntimeError, "size of data does not match that of coordinates");
        return NULL;
    }

    if (PyArray_Size((PyObject*)Weights) != n)
    {
        Py_DECREF(Xvals);
	Py_DECREF(Yvals);
	Py_DECREF(Data);
	Py_DECREF(Weights);
	Py_DECREF(p_);
        PyErr_Format(PyExc_RuntimeError, "size of weights does not match that of coordinates");
        return NULL;
    }

    covSize[0]=m[0];covSize[1]=m[0];
    outcov = (PyArrayObject*) PyArray_SimpleNew(2,covSize,PyArray_DOUBLE);
        
    x_ = PyMem_Malloc(n*sizeof(double));
    if (x_ == 0)
    {
        Py_DECREF(Xvals);
	Py_DECREF(Yvals);
	Py_DECREF(Data);
	Py_DECREF(Weights);
	Py_DECREF(p_);
        PyErr_Format(PyExc_RuntimeError, "error allocating memory");
        return NULL;
    }

    for (i = 0; i < n; i++) x_[i] = 0.0;
    
    ret=dlevmar_der(modGauss, jacGauss, p->pvals, x_, m, n, 1000, opts, info, NULL, (double*)outcov->data, &fitData); // with analytic jacobian
    //ret=dlevmar_dif(modGauss, p->pvals, x_, m, n, 1000, opts, info, NULL, NULL, &fitData); // with numeric jacobian
    
    PyMem_Free(x_);
    
    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    Py_DECREF(Data);
    Py_DECREF(Weights);
    Py_DECREF(p_);
    
    
    return Py_BuildValue("OiOff",outp,ret,outcov, info[5], info[6]);
}


static PyMethodDef levmar_gfitMethods[] = {
    {"fitGauss",  fitGauss, METH_VARARGS | METH_KEYWORDS,
    "fit a 2D Gaussian using the levmar solver: 'X', 'Y', 'Data','A'=1,'x0'=0, 'y0'=0,sigma=0,b=0,b_x=0,b_y=0"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};







PyMODINIT_FUNC initlevmar_gfit(void)
{
    PyObject *m;

    m = Py_InitModule("levmar_gfit", levmar_gfitMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}

void MAIN__(void){};
