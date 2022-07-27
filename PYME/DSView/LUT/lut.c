/*
##################
# lut.c
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
 */

#include "Python.h"

// Define error handling stuff that works on both python 2 and 3
/////////////////////////
struct module_state {
    PyObject * error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
static struct module_state _state;
#define GETSTATE(m) (&_state)
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}
// End py 3 error handling code


//#include <complex.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

static PyObject * applyLUTuint16(PyObject *self, PyObject *args, PyObject *keywds)
{
    unsigned short *data = 0;
    unsigned char *LUTR = 0;
    unsigned char *LUTG = 0;
    unsigned char *LUTB = 0;
    unsigned char *out = 0;
    float gain = 0;
    float offset = 0;
    //float d = 0;

    int tmp = 0;

    PyArrayObject *odata =0;
    PyArrayObject *adata =0;
    PyArrayObject *oLUT =0;
    PyArrayObject *oout =0;

    int sizeX;
    int sizeY;
    int N, N1;
    int i,j;

    static char *kwlist[] = {"data", "gain", "offest", "LUT", "output", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OffOO", kwlist,
         &odata, &gain, &offset, &oLUT, &oout))
        return NULL;

    /* Do the calculations */

    adata = PyArray_GETCONTIGUOUS(odata);


    if (!PyArray_Check(adata)  || !PyArray_ISCONTIGUOUS(adata))
    {
        PyErr_Format(PyExc_RuntimeError, "data - Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(adata) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }

    if (!PyArray_Check(oLUT) || !PyArray_ISCONTIGUOUS(oLUT))
    {
        PyErr_Format(PyExc_RuntimeError, "lut - Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(oLUT) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }

    if (!PyArray_Check(oout) || !PyArray_ISCONTIGUOUS(oout))
    {
        PyErr_Format(PyExc_RuntimeError, "out - Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(oout) != 3)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 3 dimensional output array");
        Py_DECREF(adata);
        return NULL;
    }

    sizeX = PyArray_DIM(odata, 0);
    sizeY = PyArray_DIM(odata, 1);

    N = PyArray_DIM(oLUT, 1);


    if ((PyArray_NDIM(oout) != 3) || (PyArray_DIM(oout, 0) != sizeX)|| (PyArray_DIM(oout, 1) != sizeY)|| (PyArray_DIM(oout, 2) != 3))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a data.shape[0] x data.shape[1] x 3 array");
        Py_DECREF(adata);
        return NULL;
    }

    data = (unsigned short*) PyArray_DATA(adata);

    LUTR = (unsigned char*) PyArray_DATA(oLUT);
    LUTG = LUTR + N;
    LUTB = LUTG + N;
    out = (unsigned char*) PyArray_DATA(oout);

    N1 = N - 1;

    gain = gain*(float)N1;

    //printf("%d\n", N);


    Py_BEGIN_ALLOW_THREADS;

    for (i=0;i < sizeX; i++)
    {
        for (j=0;j< sizeY;j++)
        {
            //d = (float)(*(unsigned short *)PyArray_GETPTR2(odata, i, j));
            tmp =  (int)MAX(MIN((gain*(((float) *data) - offset)), 255), 0);
            //tmp =  (int)(((float)(N-1))*gain*(d - offset));
            //printf("%d", tmp);
            //tmp = MIN(tmp, (N1));
            //tmp = MAX(tmp, 0);
            *out += LUTR[tmp];
            out++;
            *out += LUTG[tmp];
            out++;
            *out += LUTB[tmp];
            out++;
            data ++;
        }
    }

    Py_END_ALLOW_THREADS;

    Py_DECREF(adata);

    Py_INCREF(Py_None);
    return Py_None;
}


/*
minmax_uint16(np.ndarray[:,:])

Compute both the minumum and maximum of a uint16 array in one pass. Used to allow real-time scaling
on a full camera frame in acquisition.
Between 2 and 10 times faster than calling ndarray.min(), ndarray.max().
*/
static PyObject * minmax_uint16(PyObject *self, PyObject *args, PyObject *keywds)
{
    unsigned short *data = 0;
    unsigned short _max = 0;
    unsigned short _min = (2<<15) - 1;

    PyArrayObject *odata =0;
    PyArrayObject *adata =0;

    //int sizeX;
    //int sizeY;
    int size;
    int i;

    static char *kwlist[] = {"data", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist,
         &odata))
        return NULL;

    /* Do the calculations */

    adata = PyArray_GETCONTIGUOUS(odata);


    if (!PyArray_Check(adata)  || !PyArray_ISCONTIGUOUS(adata))
    {
        PyErr_Format(PyExc_RuntimeError, "data - Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    /*if (PyArray_NDIM(adata) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }
    */
    //sizeX = PyArray_DIM(adata, 0);
    //sizeY = PyArray_DIM(adata, 1);
    size = PyArray_SIZE(adata);

    data = (unsigned short*) PyArray_DATA(adata);

    Py_BEGIN_ALLOW_THREADS;

    for (i=0;i < size; i++)
    {
        _min = MIN(_min, *data);
        _max = MAX(_max, *data);
        data ++;
    }

    Py_END_ALLOW_THREADS;

    Py_DECREF(adata);

    //Py_INCREF(Py_None);
    return Py_BuildValue("H,H", _min, _max);
}

static PyObject * applyLUTuint8(PyObject *self, PyObject *args, PyObject *keywds)
{
    unsigned char *data = 0;
    unsigned char *LUTR = 0;
    unsigned char *LUTG = 0;
    unsigned char *LUTB = 0;
    unsigned char *out = 0;
    float gain = 0;
    float offset = 0;
    //float d = 0;

    int tmp = 0;

    PyArrayObject *odata =0;
    PyArrayObject *adata =0;
    PyArrayObject *oLUT =0;
    PyArrayObject *oout =0;

    int sizeX;
    int sizeY;
    int N, N1;
    int i,j;

    static char *kwlist[] = {"data", "gain", "offest", "LUT", "output", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OffOO", kwlist,
         &odata, &gain, &offset, &oLUT, &oout))
        return NULL;

    /* Do the calculations */

    adata = PyArray_GETCONTIGUOUS(odata);


    if (!PyArray_Check(adata)  || !PyArray_ISCONTIGUOUS(adata))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(adata) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }

    if (!PyArray_Check(oLUT) || !PyArray_ISCONTIGUOUS(oLUT))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(oLUT) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }

    if (!PyArray_Check(oout) || !PyArray_ISCONTIGUOUS(oout))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(oout) != 3)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }

    sizeX = PyArray_DIM(odata, 0);
    sizeY = PyArray_DIM(odata, 1);

    N = PyArray_DIM(oLUT, 1);


    if ((PyArray_NDIM(oout) != 3) || (PyArray_DIM(oout, 0) != sizeX)|| (PyArray_DIM(oout, 1) != sizeY)|| (PyArray_DIM(oout, 2) != 3))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a data.shape[0] x data.shape[1] x 3 array");
        Py_DECREF(adata);
        return NULL;
    }

    data = (unsigned char*) PyArray_DATA(adata);

    LUTR = (unsigned char*) PyArray_DATA(oLUT);
    LUTG = LUTR + N;
    LUTB = LUTG + N;
    out = (unsigned char*) PyArray_DATA(oout);

    N1 = N - 1;

    gain = gain*(float)N1;

    //printf("%d\n", N);


    Py_BEGIN_ALLOW_THREADS;

    for (i=0;i < sizeX; i++)
    {
        for (j=0;j< sizeY;j++)
        {
            //d = (float)(*(unsigned short *)PyArray_GETPTR2(odata, i, j));
            //tmp =  (int)(gain*(((float) *data) - offset));
            tmp =  (int)MAX(MIN((gain*(((float) *data) - offset)), 255), 0);
            //tmp =  (int)(((float)(N-1))*gain*(d - offset));
            //printf("%d", tmp);
            //tmp = MIN(tmp, N1);
            //tmp = MAX(tmp, 0);
            *out += LUTR[tmp];
            out++;
            *out += LUTG[tmp];
            out++;
            *out += LUTB[tmp];
            out++;
            data ++;
        }
    }

    Py_END_ALLOW_THREADS;

    Py_DECREF(adata);

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject * applyLUTfloat(PyObject *self, PyObject *args, PyObject *keywds)
{
    float *data = 0;
    unsigned char *LUTR = 0;
    unsigned char *LUTG = 0;
    unsigned char *LUTB = 0;
    unsigned char *out = 0;
    float gain = 0;
    float offset = 0;
    //float d = 0;

    int tmp = 0;

    PyArrayObject *odata =0;
    PyArrayObject *adata =0;
    PyArrayObject *oLUT =0;
    PyArrayObject *oout =0;

    int sizeX;
    int sizeY;
    int N, N1;
    int i,j;

    static char *kwlist[] = {"data", "gain", "offest", "LUT", "output", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OffOO", kwlist,
         &odata, &gain, &offset, &oLUT, &oout))
        return NULL;

    /* Do the calculations */

    adata = PyArray_GETCONTIGUOUS(odata);


    if (!PyArray_Check(adata)  || !PyArray_ISCONTIGUOUS(adata))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(adata) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }

    if (!PyArray_Check(oLUT) || !PyArray_ISCONTIGUOUS(oLUT))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(oLUT) != 2)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }

    if (!PyArray_Check(oout) || !PyArray_ISCONTIGUOUS(oout))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array");
        Py_DECREF(adata);
        return NULL;
    }

    if (PyArray_NDIM(oout) != 3)
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a 2 dimensional array");
        Py_DECREF(adata);
        return NULL;
    }

    sizeX = PyArray_DIM(odata, 0);
    sizeY = PyArray_DIM(odata, 1);

    N = PyArray_DIM(oLUT, 1);


    if ((PyArray_NDIM(oout) != 3) || (PyArray_DIM(oout, 0) != sizeX)|| (PyArray_DIM(oout, 1) != sizeY)|| (PyArray_DIM(oout, 2) != 3))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a data.shape[0] x data.shape[1] x 3 array");
        Py_DECREF(adata);
        return NULL;
    }

    data = (float*) PyArray_DATA(adata);

    LUTR = (unsigned char*) PyArray_DATA(oLUT);
    LUTG = LUTR + N;
    LUTB = LUTG + N;
    out = (unsigned char*) PyArray_DATA(oout);

    N1 = N - 1;

    gain = gain*(float)N1;

    //printf("%d\n", N);

    Py_BEGIN_ALLOW_THREADS;

    for (i=0;i < sizeX; i++)
    {
        for (j=0;j< sizeY;j++)
        {
            //d = (float)(*(unsigned short *)PyArray_GETPTR2(odata, i, j));
            //tmp =  (int)(gain*(((float) *data) - offset));
            tmp =  (int)MAX(MIN((gain*(((float) *data) - offset)), 255), 0);
            //tmp =  (int)(((float)(N-1))*gain*(d - offset));
            //printf("%d", tmp);
            //tmp = MIN(tmp, N1);
            //tmp = MAX(tmp, 0);
            *out = MIN(*out + LUTR[tmp], 255);
            //*out += LUTR[tmp];
            out++;
            //*out += LUTG[tmp];
            *out = MIN(*out + LUTG[tmp], 255);
            out++;
            //*out += LUTB[tmp];
            *out = MIN(*out + LUTB[tmp], 255);
            out++;
            data ++;
        }
    }

    Py_END_ALLOW_THREADS;

    Py_DECREF(adata);

    Py_INCREF(Py_None);
    return Py_None;
}



//Begin module initialization stuff. Try to make compatible with python 2 and 3


static PyMethodDef lutMethods[] = {
    {"applyLUTu16",  (PyCFunction)applyLUTuint16, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"applyLUTu8",  (PyCFunction)applyLUTuint8, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"applyLUTf",  (PyCFunction)applyLUTfloat, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"minmax_u16",  (PyCFunction)minmax_uint16, METH_VARARGS | METH_KEYWORDS,
    ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3

static int lut_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int lut_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "lut",
        NULL,
        sizeof(struct module_state),
        lutMethods,
        NULL,
        lut_traverse,
        lut_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC PyInit_lut(void)
#else
#define INITERROR return

PyMODINIT_FUNC initlut(void)
#endif
{
    struct module_state *st;
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("lut", lutMethods);
#endif

    import_array();

    if (module == NULL)
        INITERROR;

    st = GETSTATE(module);

    st->error = PyErr_NewException("lut.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}



//PyMODINIT_FUNC initlut(void)
//{
//    PyObject *m;
//
//    m = Py_InitModule("lut", lutMethods);
//    import_array()
//
//    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
//    //Py_INCREF(SpamError);
//    //PyModule_AddObject(m, "error", SpamError);
//}
