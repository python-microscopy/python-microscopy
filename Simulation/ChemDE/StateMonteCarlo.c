/*
##################
# edgeDB.c
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
 */


#include "Python.h"
//#include <complex.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>
#include <stdlib.h>



static PyObject * StateMC(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *TransitionMatrix =0;
    PyObject *Iterates = 0;

    npy_int32 *res = 0;

    int nSteps;
    //int startState=0;
    int state = 0;
    int nStates;
    npy_intp dims[2];

    int i=0;
    double r;
    const double IRMAX = 1.0/RAND_MAX;


    static char *kwlist[] = {"TransitionMatrix", "NSteps", "StartState", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oi|i", kwlist,
         &TransitionMatrix, &nSteps, &state))
        return NULL;


    if (!PyArray_Check(TransitionMatrix) || !PyArray_ISCONTIGUOUS(TransitionMatrix))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the transition matrix");
        return NULL;
    }

    nStates = PyArray_DIM(TransitionMatrix, 0);
    if (PyArray_DIM(TransitionMatrix, 0) != nStates)
    {
        PyErr_Format(PyExc_RuntimeError, "Transition matrix should be square");
        return NULL;
    }

    

    dims[0] = nSteps;
    dims[1] = 1;
    Iterates = PyArray_SimpleNew(1, dims, PyArray_INT32);
    if (!Iterates)
    {
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for steps");
        return NULL;
    }

    res = (npy_int32*)PyArray_DATA(Iterates);

    while (nSteps > 0)
    {
        r = ((double)random())*IRMAX;

        i = 0;
        while (i < nStates)
        {
            if (r < *(double*)PyArray_GETPTR2(TransitionMatrix, state, i))
            {
                state = i;
                break;
            } else i ++;

        }

        *res = (npy_int32) state;

        res ++;
        nSteps --;
    }

    //Py_INCREF(Iterates);
    return (PyObject*) Iterates;
}







static PyMethodDef StateMCMethods[] = {
    {"StateMC",  StateMC, METH_VARARGS | METH_KEYWORDS,
    ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initStateMC(void)
{
    PyObject *m;

    m = Py_InitModule("StateMC", StateMCMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
