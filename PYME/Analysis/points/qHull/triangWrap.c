/*
##################
# triangRend.c
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
 */

#include "Python.h"
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>
#include "triangLhood.h"

#define MODE_AREA 0
#define MODE_SIDELENGTH 1

static PyObject * PyCalcLHood(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *oPositions =0;
    PyArrayObject* aPositions;

    int dim, nPositions;
    int mode = 0;
    int iErr;
    double lhood;

    static char *kwlist[] = {"positions", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|i", kwlist,
         &oPositions, &mode))
        return NULL;

    /* Do the calculations */

    aPositions = (PyArrayObject *) PyArray_ContiguousFromObject(oPositions, PyArray_DOUBLE, 2, 2);
    if (aPositions == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad position data");
      return NULL;
    }


    dim = PyArray_DIM(aPositions, 1);
    nPositions = PyArray_DIM(aPositions, 0);

    if (mode == MODE_AREA)
    {
        iErr = calcLHood ((coordT *)PyArray_DATA(aPositions), dim,  nPositions, &lhood);
        if (iErr)
        {
          PyErr_Format(PyExc_RuntimeError, "QHull error");
          Py_DECREF(aPositions);
          return NULL;
        }
    } else
    {

        if (dim == 2)
        {
            iErr = calcLHood2D((coordT *)PyArray_DATA(aPositions),  nPositions, &lhood);
            if (iErr)
            {
              PyErr_Format(PyExc_RuntimeError, "QHull error");
              Py_DECREF(aPositions);
              return NULL;
            }
        }
        else if (dim == 3)
        {
            iErr = calcLHood3D((coordT *)PyArray_DATA(aPositions),  nPositions, &lhood);
            if (iErr)
            {
              PyErr_Format(PyExc_RuntimeError, "QHull error");
              Py_DECREF(aPositions);
              return NULL;
            }
        }
        else
        {
           PyErr_Format(PyExc_RuntimeError, "Expecting a list of 2D, or 3D points");
           Py_DECREF(aPositions);
           return NULL;
        }
    }

    Py_DECREF(aPositions);

    return Py_BuildValue("d", lhood);
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






static PyMethodDef triangWrapMethods[] = {
    {"CalcLHood",  PyCalcLHood, METH_VARARGS | METH_KEYWORDS,
    "Calculate the likelihood for a give set of point positions:\n Arguments: positions (an NxD array), mode"},
    {"RenderTetrahedra",  PyTetAndDraw, METH_VARARGS | METH_KEYWORDS,
    "Generate a tetrahedra from a set of points and render them into an image.\n Arguments: positions (an Nx3 array), image (a WxHxD 3D array with Fortran byte order)"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC inittriangWrap(void)
{
    PyObject *m;

    m = Py_InitModule("triangWrap", triangWrapMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}