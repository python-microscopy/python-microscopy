#include "Python.h"
//#include <complex.h>
#include <stdlib.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

#define MIN(a, b) ((a<b) ? a : b)
#define MAX(a, b) ((a>b) ? a : b)

#define MAX_DIMS 4
static PyObject * astiglookup(PyObject *self, PyObject *args, PyObject *keywds)
{
    npy_intp outDimensions[1];
    int nPts=0;
    int nViews = 0;
    int nCalPts = 0;

    int i, j, k;
    float err_i, err_k, errX, errY;
    float w_k;
    int k_i;

    float s_xi[MAX_DIMS];
    float s_yi[MAX_DIMS];
    float w_xi[MAX_DIMS];
    float w_yi[MAX_DIMS];

    float *p_out_err;
    int *p_out_z;

    PyObject *owX =0;
    PyObject *owY=0;
    PyObject *osX =0;
    PyObject *osY=0;
    PyObject *os_calX =0;
    PyObject *os_calY=0;

    PyArrayObject* out_z=NULL;
    PyArrayObject* out_err=NULL;

    static char *kwlist[] = {"sigCalX", "sigCalY", "sX", "sY", "wX", "wY", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOO|", kwlist,
         &os_calX, &os_calY, &osX, &osY, &owX, &owY))
        return NULL;

    //TODO - add checks on input dimensionality
    nPts = PyArray_DIM(osX, 0);
    nViews = PyArray_DIM(osX, 1);

    nCalPts = PyArray_DIM(os_calX, 0);

    outDimensions[0] = nPts;

    // Allocate output arrays
    out_z = (PyArrayObject*) PyArray_SimpleNew(1,outDimensions,PyArray_INT);
    if (out_z == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      goto fail;
    }

    out_err = (PyArrayObject*) PyArray_SimpleNew(1,outDimensions,PyArray_FLOAT);
    if (out_err == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      goto fail;
    }

    p_out_z = PyArray_DATA(out_z);
    p_out_err = PyArray_DATA(out_err);

    //Do the calculations
    for (i=0; i < nPts; i++)
    {
        for (j=0; j < nViews; j++)
        {
            s_xi[j] = * (float*) PyArray_GETPTR2(osX, i, j);
            s_yi[j] = * (float*) PyArray_GETPTR2(osY, i, j);
            w_xi[j] = * (float*) PyArray_GETPTR2(owX, i, j);
            w_yi[j] = * (float*) PyArray_GETPTR2(owY, i, j);
        }

        err_i = 1e9;
        k_i = -1;

        for (k=0; k < nCalPts; k++)
        {
            err_k = 0;
            w_k = 0;

            for (j=0; j < nViews; j++)
            {
                errX = (s_xi[j] - * (float *) PyArray_GETPTR2(os_calX, k, j));
                errY = (s_yi[j] - * (float *) PyArray_GETPTR2(os_calY, k, j));

                err_k += w_xi[j]*errX*errX + w_yi[j]*errY*errY;
                w_k += w_xi[j] + w_yi[j];
            }

            err_k /= w_k;

            if (err_k < err_i)
            {
                k_i = k;
                err_i = err_k;
            }
        }

        p_out_err[i] = err_i;
        p_out_z[i] = k_i;
    }

    return Py_BuildValue("OO", (PyObject*) out_z, (PyObject*) out_err);

fail:

    Py_XDECREF(out_z);
    Py_XDECREF(out_err);

    return NULL;
}

#define N_VIEWS 4
static PyObject * astiglookup4(PyObject *self, PyObject *args, PyObject *keywds)
{
    npy_intp outDimensions[1];
    int nPts=0;
    int nViews = 0;
    int nCalPts = 0;

    int i, j, k;
    float err_i, err_k, errX, errY;
    float w_k;
    int k_i;

    float s_xi[N_VIEWS];
    float s_yi[N_VIEWS];
    float w_xi[N_VIEWS];
    float w_yi[N_VIEWS];

    float *p_out_err;
    int *p_out_z;

    PyObject *owX =0;
    PyObject *owY=0;
    PyObject *osX =0;
    PyObject *osY=0;
    PyObject *os_calX =0;
    PyObject *os_calY=0;

    PyArrayObject* out_z=NULL;
    PyArrayObject* out_err=NULL;

    static char *kwlist[] = {"sigCalX", "sigCalY", "sX", "sY", "wX", "wY", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOO|", kwlist,
         &os_calX, &os_calY, &osX, &osY, &owX, &owY))
        return NULL;

    //TODO - add checks on input dimensionality
    nPts = PyArray_DIM(osX, 0);
    nViews = PyArray_DIM(osX, 1);

    nCalPts = PyArray_DIM(os_calX, 0);

    outDimensions[0] = nPts;

    // Allocate output arrays
    out_z = (PyArrayObject*) PyArray_SimpleNew(1,outDimensions,PyArray_INT);
    if (out_z == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      goto fail;
    }

    out_err = (PyArrayObject*) PyArray_SimpleNew(1,outDimensions,PyArray_FLOAT);
    if (out_err == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      goto fail;
    }

    p_out_z = PyArray_DATA(out_z);
    p_out_err = PyArray_DATA(out_err);

    //Do the calculations
    for (i=0; i < nPts; i++)
    {
        for (j=0; j < N_VIEWS; j++)
        {
            s_xi[j] = * (float*) PyArray_GETPTR2(osX, i, j);
            s_yi[j] = * (float*) PyArray_GETPTR2(osY, i, j);
            w_xi[j] = * (float*) PyArray_GETPTR2(owX, i, j);
            w_yi[j] = * (float*) PyArray_GETPTR2(owY, i, j);
        }

        err_i = 1e9;
        k_i = -1;

        //coarse search
        for (k=0; k < nCalPts; k += 100)
        {
            err_k = 0;
            w_k = 0;

            for (j=0; j < N_VIEWS; j++)
            {
                errX = (s_xi[j] - * (float *) PyArray_GETPTR2(os_calX, k, j));
                errY = (s_yi[j] - * (float *) PyArray_GETPTR2(os_calY, k, j));

                err_k += w_xi[j]*errX*errX + w_yi[j]*errY*errY;
                w_k += w_xi[j] + w_yi[j];
            }

            err_k /= w_k;

            if (err_k < err_i)
            {
                k_i = k;
                err_i = err_k;
            }
        }

        //fine search
        for (k= MAX(k_i - 100, 0); k < MIN(k_i + 100, nCalPts); k++)
        {
            err_k = 0;
            w_k = 0;

            for (j=0; j < N_VIEWS; j++)
            {
                errX = (s_xi[j] - * (float *) PyArray_GETPTR2(os_calX, k, j));
                errY = (s_yi[j] - * (float *) PyArray_GETPTR2(os_calY, k, j));

                err_k += w_xi[j]*errX*errX + w_yi[j]*errY*errY;
                w_k += w_xi[j] + w_yi[j];
            }

            err_k /= w_k;

            if (err_k < err_i)
            {
                k_i = k;
                err_i = err_k;
            }
        }

        p_out_err[i] = err_i;
        p_out_z[i] = k_i;
    }

    return Py_BuildValue("OO", (PyObject*) out_z, (PyObject*) out_err);

fail:

    Py_XDECREF(out_z);
    Py_XDECREF(out_err);

    return NULL;
}



static PyMethodDef astiglookupMethods[] = {
    {"astig_lookup",  (PyCFunction)astiglookup4, METH_VARARGS | METH_KEYWORDS,
    ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "astiglookup",     /* m_name */
        "major refactoring of the Analysis tree",  /* m_doc */
        -1,                  /* m_size */
        astiglookupMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_astiglookup(void)
{
	PyObject *m;
    // m = PyModule_Create("edgeDB", edgeDBMethods);
    m = PyModule_Create(&moduledef);
    import_array()
    return m;
}
#else
PyMODINIT_FUNC initastiglookup(void)
{
    PyObject *m;

    m = Py_InitModule("astiglookup", astiglookupMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
#endif
