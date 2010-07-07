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

        if (dim != 2)
        {
           PyErr_Format(PyExc_RuntimeError, "Expecting a list of 2D points");
           Py_DECREF(aPositions);
           return NULL;
        }

        iErr = calcLHood2D((coordT *)PyArray_DATA(aPositions),  nPositions, &lhood);
        if (iErr)
        {
          PyErr_Format(PyExc_RuntimeError, "QHull error");
          Py_DECREF(aPositions);
          return NULL;
        }
    }

    Py_DECREF(aPositions);

    return Py_BuildValue("d", lhood);
}




static PyMethodDef triangWrapMethods[] = {
    {"CalcLHood",  PyCalcLHood, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
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