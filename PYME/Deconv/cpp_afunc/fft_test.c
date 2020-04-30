#include "Python.h"
//#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include "Numeric/arrayobject.h"
#include <stdio.h>

#define MIN(a, b) ((a<b) ? a : b) 
#define MAX(a, b) ((a>b) ? a : b)

fftwf_complex *F, *tmp, *tmp2, *tmp3;
fftwf_plan p, p1, p2, p3;

int prepared = 0;

static PyObject * cDec_fw_map(PyObject *self, PyObject *args) 
{ 
    PyObject *input1, *input2, *input3, *input4, *input5, *input6, *input7; 
    PyArrayObject *f, *alpha, *H, *He1, *He2, *e1, *e2, *result;
    
    //fftwf_complex *F, *tmp, *tmp2, *tmp3;
    //fftwf_plan p, p1, p2, p3;
     
    //int dimensions[1]; 
     
    //long dim0[1], dim1[1]; 
    int k,l,m;
    
    if (prepared == 0) return NULL;
    
    //extern dgemv_(char *trans, long *m, long *n, double *alpha, double *a, long *lda, double *x, long *incx, double *beta, double *Y, long *incy); 
    //printf("debug_0");
    
    if (!PyArg_ParseTuple(args, "OOOOOOO", &input1, &input2, &input3, &input4, &input5, &input6, &input7)) return NULL; 
    
    //printf("debug_1");
    
    f = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_CFLOAT, 1, 1); 
    if (f == NULL) return NULL; 
    
    //printf("debug_2");
    
    alpha = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_CFLOAT, 3, 3); 
    if (alpha == NULL)
    {
        Py_DECREF(f);
        return NULL;
    }
    
    //printf("strides (%d,%d,%d)", alpha->strides[0], alpha->strides[1], alpha->strides[2]);
    //printf("debug_3");
    
    H = (PyArrayObject *) PyArray_ContiguousFromObject(input3, PyArray_CFLOAT, 3, 3); 
    if (H == NULL)
    {
        Py_DECREF(f);
        Py_DECREF(alpha);
        return NULL;
    }
    
    He1 = (PyArrayObject *) PyArray_ContiguousFromObject(input4, PyArray_CFLOAT, 3, 3); 
    if (He1 == NULL) 
    {
        Py_DECREF(f);
        Py_DECREF(alpha);
        Py_DECREF(H);
        return NULL;
    }
    
    He2 = (PyArrayObject *) PyArray_ContiguousFromObject(input5, PyArray_CFLOAT, 3, 3); 
    if (He2 == NULL)  
    {
        Py_DECREF(f);
        Py_DECREF(alpha);
        Py_DECREF(H);
        Py_DECREF(He1);
        return NULL;
    } 
    
    e1 = (PyArrayObject *) PyArray_ContiguousFromObject(input6, PyArray_CFLOAT, 3, 3); 
    if (e1 == NULL)   
    {
        Py_DECREF(f);
        Py_DECREF(alpha);
        Py_DECREF(H);
        Py_DECREF(He1);
        Py_DECREF(He2);
        return NULL;
    }  
    
    e2 = (PyArrayObject *) PyArray_ContiguousFromObject(input7, PyArray_CFLOAT, 3, 3); 
    if (e2 == NULL)    
    {
        Py_DECREF(f);
        Py_DECREF(alpha);
        Py_DECREF(H);
        Py_DECREF(He1);
        Py_DECREF(He2);
        Py_DECREF(e1);
        return NULL;
    } 
    
    result = (PyArrayObject *)PyArray_SimpleNew(1, f->dimensions, PyArray_FLOAT); 
    if (result == NULL)    
    {
        Py_DECREF(f);
        Py_DECREF(alpha);
        Py_DECREF(H);
        Py_DECREF(He1);
        Py_DECREF(He2);
        Py_DECREF(e1);
        Py_DECREF(e2);
        return NULL;
    } 
    
    //printf("debug_4");
    
    
    
    //printf("debug_5");
    
    p = fftwf_plan_dft_3d(H->dimensions[0],H->dimensions[1],H->dimensions[2], ((fftwf_complex*)f->data), F, FFTW_FORWARD, FFTW_ESTIMATE);
    
     //printf("debug_6");
    
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    
    
    
     //printf("debug_7");
    
    for (k = 0; k < f->dimensions[0]; k++)
    {
        tmp[k] = F[k]*((fftwf_complex*)H->data)[k];
        tmp2[k] = F[k]*((fftwf_complex*)He1->data)[k];
        tmp3[k] = F[k]*((fftwf_complex*)He2->data)[k];
    }
    
    
    fftwf_execute(p1);
    
    fftwf_execute(p2);
    
    fftwf_execute(p3);
    
    
    for (k = 0; k < f->dimensions[0]; k++)
    {
        //for (l = 0; l < f->dimensions[0]; l++)
        //{
        //    for (m = 0; m < f->dimensions[0]; m++)
        //    {
                ((float*)result->data)[k] = (float)((1.5*crealf(tmp[k]) +  2*crealf(tmp2[k]*((fftwf_complex*)e1->data)[k]) +  0.5*crealf(tmp2[k]*((fftwf_complex*)e2->data)[k]))/f->dimensions[0]);
        //    }
        //}
    }
    
    /*if (matrix->dimensions[1] != vector->dimensions[0]) 
    { 
        PyErr_SetString(PyExc_ValueError, "array dimensions are not compatible"); 
        return NULL; 
    } */
    
    
    //dimensions[0] = matrix->dimensions[0]; 
    
    
    
    
    //dim0[0] = (long)matrix->dimensions[0];  
    //dim1[0] = (long)matrix->dimensions[1]; 
    
    //dgemv_("T", dim1, dim0, factor, (double *)matrix->data, dim1, (double *)vector->data, int_one, real_zero, (double *)result->data, int_one); 
    
    Py_DECREF(f);
    Py_DECREF(alpha);
    Py_DECREF(H);
    Py_DECREF(He1);
    Py_DECREF(He2);
    Py_DECREF(e1);
    Py_DECREF(e2);
    
    return PyArray_Return(result); 
}


static PyObject * cDec_L_func(PyObject *self, PyObject *args) 
{ 
    PyObject *input1; 
    PyArrayObject *f, *result;
    int x_siz,y_siz, z_siz;
    int k;
        
    if (!PyArg_ParseTuple(args, "O(iii)", &input1, &x_siz, &y_siz, &z_siz)) return NULL; 
      
    f = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_FLOAT, 1, 1); 
    if (f == NULL) return NULL; 
    
    result = (PyArrayObject *)PyArray_SimpleNew(1, f->dimensions, PyArray_FLOAT); 
    if (result == NULL) return NULL; 
    
    for (k = 0; k < f->dimensions[0];k++)
    {
        ((float*)result->data)[k] = -6*((float*)f->data)[k];
        ((float*)result->data)[k] += ((float*)f->data)[MIN(k + 1, f->dimensions[0])];
        ((float*)result->data)[k] += ((float*)f->data)[MAX(k - 1, 0)];
        ((float*)result->data)[k] += ((float*)f->data)[MIN(k + x_siz, f->dimensions[0])];
        ((float*)result->data)[k] += ((float*)f->data)[MAX(k - x_siz, 0)];
        ((float*)result->data)[k] += ((float*)f->data)[MIN(k + x_siz*y_siz, f->dimensions[0])];
        ((float*)result->data)[k] += ((float*)f->data)[MAX(k - x_siz*y_siz, 0)];
    }
    
    Py_DECREF(f);
     
    return PyArray_Return(result); 
}

static PyObject * cDec_prepare(PyObject *self, PyObject *args) 
{
    int x_siz,y_siz, z_siz;
    int k;
    
    if (prepared == 1) return NULL;
        
    if (!PyArg_ParseTuple(args, "(iii)", &x_siz, &y_siz, &z_siz)) return NULL;
    
    tmp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * x_siz*y_siz*z_siz);
    if (tmp == NULL) return NULL;
    tmp2 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * x_siz*y_siz*z_siz);
    if (tmp2 == NULL) return NULL;
    tmp3 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * x_siz*y_siz*z_siz);
    if (tmp3 == NULL) return NULL;
    F = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * x_siz*y_siz*z_siz);
    if (F == NULL) return NULL;
    
    p1 = fftwf_plan_dft_3d(x_siz,y_siz,z_siz, tmp, tmp, FFTW_BACKWARD, FFTW_MEASURE);
    p2 = fftwf_plan_dft_3d(x_siz,y_siz,z_siz, tmp2, tmp2, FFTW_BACKWARD, FFTW_ESTIMATE);
    p3 = fftwf_plan_dft_3d(x_siz,y_siz,z_siz, tmp3, tmp3, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    prepared = 1;
    
    return Py_BuildValue("i",prepared);
}

static PyObject * cDec_cleanup(PyObject *self, PyObject *args) 
{
    fftwf_destroy_plan(p1);
    fftwf_destroy_plan(p2);
    fftwf_destroy_plan(p3);
    
    fftwf_free(tmp); fftwf_free(tmp2);fftwf_free(tmp3);fftwf_free(F);
    
    prepared = 0;
    
    return Py_BuildValue("i",prepared);
}

static PyMethodDef cDecMethods[] = {
    {"fw_map",  cDec_fw_map, METH_VARARGS,
     "Perform forward mapping."},
     {"Lfunc",  cDec_L_func, METH_VARARGS,
     "Likelihood fcn."},
     {"prepare",  cDec_prepare, METH_VARARGS,
     "Do some allocations, compute fftw plans etc."},
     {"cleanup",  cDec_cleanup, METH_VARARGS,
     "deallocate buffers & fftw plans."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initcDec(void)
{
    PyObject *m;

    m = Py_InitModule("cDec", cDecMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}






