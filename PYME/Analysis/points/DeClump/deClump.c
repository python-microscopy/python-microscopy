/*
##################
# deClump.c
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

#define MAX_RECURSION_DEPTH 10000

#define MIN(a, b) ((a<b) ? a : b)
#define MAX(a, b) ((a>b) ? a : b)

int findConnected(int i, int nPts, int *t, float *x, float *y,float *delta_x, int *frameIndices, int *assigned, int clumpNum, int nFrames, int *recDepth)
{
    //float dis;
    float dx;
    float dy;
    int j;

    //printf("bar\n");
    (*recDepth)++;
    if (*recDepth > MAX_RECURSION_DEPTH)
    {
        //PyErr_Format(PyExc_RuntimeError, "Exceeded max recursion depth");
        printf("Warning: max recursion depth reached - objects might be artificially divided\n");
        return -1;
    }

    

    //printf("%d, ", MIN(frameIndices[t[i] + nFrames], nPts) - (i+1));
    for (j = i+1; j < MIN(frameIndices[t[i] + nFrames], nPts); j++)
    {      
        //if (i < 200) printf("b %d, %d, %d, %d\t", i, j,  MIN(frameIndices[t[i] + nFrames], nPts), clumpNum);
        if (assigned[j]==0)
        {
            dx = x[j] - x[i];
            dy = y[j] - y[i];
            //printf("d %f, %f\t", dx, dy);

            if ((dx*dx + dy*dy) < (4*delta_x[i]*delta_x[i]))
            {
                //printf("bar %d, %d, %d, %d\t", i, j,  MIN(frameIndices[t[i] + nFrames], nPts), clumpNum);
                assigned[j] = clumpNum;

                findConnected(j, nPts, t, x, y, delta_x, frameIndices, assigned, clumpNum, nFrames, recDepth);// == -1)
                //    return -1;
            }
        }
    }

}

int findConnectedN(int i, int nPts, int *t, float *x, float *y,float *delta_x, int *frameIndices, int *assigned, int clumpNum, int nFrames, int *recDepth)
{
    //float dis;
    float dx;
    float dy;
    int j;

    //printf("bar\n");
    /*(*recDepth)++;
    if (*recDepth > MAX_RECURSION_DEPTH)
    {
        //PyErr_Format(PyExc_RuntimeError, "Exceeded max recursion depth");
        printf("Warning: max recursion depth reached - objects might be artificially divided\n");
        return -1;
    }*/

    
    for (j = MAX(frameIndices[MAX(t[i] - nFrames, 0)], 0); j < i; j++)
    {      
        if (assigned[j]!=0)
        {
            dx = x[j] - x[i];
            dy = y[j] - y[i];

            if ((dx*dx + dy*dy) < (4*delta_x[i]*delta_x[i]))
            {
                return assigned[j];
            }
        }
    }

    return -1;

}


static PyObject * findClumps(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *tO = 0;
    PyObject *xO = 0;
    PyObject *yO = 0;
    PyObject *delta_xO = 0;

    PyArrayObject *tA = 0;
    PyArrayObject *xA = 0;
    PyArrayObject *yA = 0;
    PyArrayObject *delta_xA = 0;

    int *t = 0;
    float *x = 0;
    float *y = 0;
    float *delta_x = 0;

    PyObject * assignedA=0;


    int nPts = 0;
    //int nTimes = 0;
    int tMax = 0;

    int nFrames = 10;

    int *frameIndices = 0;

    int *assigned = 0;
    int clumpNum = 1;

    int dims[2];
    int i = 0;
    int j = 0;
    int t_last = 0;
    int t_i = 0;
    int recDepth = 0;

    static char *kwlist[] = {"t", "x", "y", "delta_x", "nFrames", NULL};

    dims[0] = 0;
    dims[1] = 0;


    

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|i", kwlist,
         &tO, &xO, &yO, &delta_xO, &nFrames))
        return NULL;

    tA = (PyArrayObject *) PyArray_ContiguousFromObject(tO, PyArray_INT, 0, 1);
    if (tA == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad t");
      return NULL;
    }

    nPts = PyArray_DIM(tA, 0);

    xA = (PyArrayObject *) PyArray_ContiguousFromObject(xO, PyArray_FLOAT, 0, 1);
    if ((xA == NULL) || (PyArray_DIM(xA, 0) != nPts))
    {
      Py_DECREF(tA);
      PyErr_Format(PyExc_RuntimeError, "Bad x");
      return NULL;
    }

    yA = (PyArrayObject *) PyArray_ContiguousFromObject(yO, PyArray_FLOAT, 0, 1);
    if ((yA == NULL) || (PyArray_DIM(xA, 0) != nPts))
    {
      Py_DECREF(tA);
      Py_DECREF(xA);
      PyErr_Format(PyExc_RuntimeError, "Bad y");
      return NULL;
    }

    delta_xA = (PyArrayObject *) PyArray_ContiguousFromObject(delta_xO, PyArray_FLOAT, 0, 1);
    if ((delta_xA == NULL) || (PyArray_DIM(xA, 0) != nPts))
    {
      Py_DECREF(tA);
      Py_DECREF(xA);
      Py_DECREF(yA);
      PyErr_Format(PyExc_RuntimeError, "Bad delta_x");
      return NULL;
    }


    t = (int*)PyArray_DATA(tA);
    x = (float*)PyArray_DATA(xA);
    y = (float*)PyArray_DATA(yA);
    delta_x = (float*)PyArray_DATA(delta_xA);


    dims[0] = nPts;
    printf("nPts = %d\n", nPts);
    assignedA = PyArray_SimpleNew(1, dims, PyArray_INT32);
    if (assignedA == NULL)
    {
        Py_DECREF(tA);
        Py_DECREF(xA);
        Py_DECREF(yA);
        Py_DECREF(delta_xA);
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for objects");
        return NULL;
    }

    assigned = (int*)PyArray_DATA(assignedA);

    for (i=0; i < nPts; i++)
    {
        assigned[i] = 0;
        tMax = MAX(tMax, t[i]);
    }


    
    frameIndices = malloc((tMax + 10)*sizeof(int));
    for (i=0; i < (tMax + 10); i++)
    {
        frameIndices[i] = (nPts + 2);
    }

    for (i=0; i < nPts; i++)
    {
        t_i = t[i];
        for (j= t_last; j < (t_i + 1); j++)
        {
            frameIndices[j] = i;
        }
        t_last = t_i;
    }

/*
    for (i=0; i < (tMax + 10); i++)
    {
        printf("%d, ", frameIndices[i]);
    }

    printf("\n");
*/

    //i = 0;

    for (i=0; i < nPts; i++)
    {
        //printf("foo %d\n", i);
        if (assigned[i] == 0)
        {
            assigned[i] = clumpNum;
            recDepth = 0;

            findConnected(i, nPts, t, x, y, delta_x, frameIndices, assigned, clumpNum, nFrames, &recDepth);

            clumpNum++;
        }
    }


    free(frameIndices);

    Py_DECREF(tA);
    Py_DECREF(xA);
    Py_DECREF(yA);
    Py_DECREF(delta_xA);

    return (PyObject*) assignedA;

fail:
    return NULL;
}

static PyObject * findClumpsN(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *tO = 0;
    PyObject *xO = 0;
    PyObject *yO = 0;
    PyObject *delta_xO = 0;

    PyArrayObject *tA = 0;
    PyArrayObject *xA = 0;
    PyArrayObject *yA = 0;
    PyArrayObject *delta_xA = 0;

    int *t = 0;
    float *x = 0;
    float *y = 0;
    float *delta_x = 0;

    PyObject * assignedA=0;


    int nPts = 0;
    //int nTimes = 0;
    int tMax = 0;

    int nFrames = 10;

    int *frameIndices = 0;

    int *assigned = 0;
    int clumpNum = 1;

    int clump = -1;

    int dims[2];
    int i = 0;
    int j = 0;
    int t_last = 0;
    int t_i = 0;
    int recDepth = 0;

    static char *kwlist[] = {"t", "x", "y", "delta_x", "nFrames", NULL};

    dims[0] = 0;
    dims[1] = 0;


    

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|i", kwlist,
         &tO, &xO, &yO, &delta_xO, &nFrames))
        return NULL;

    tA = (PyArrayObject *) PyArray_ContiguousFromObject(tO, PyArray_INT, 0, 1);
    if (tA == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad t");
      return NULL;
    }

    nPts = PyArray_DIM(tA, 0);

    xA = (PyArrayObject *) PyArray_ContiguousFromObject(xO, PyArray_FLOAT, 0, 1);
    if ((xA == NULL) || (PyArray_DIM(xA, 0) != nPts))
    {
      Py_DECREF(tA);
      PyErr_Format(PyExc_RuntimeError, "Bad x");
      return NULL;
    }

    yA = (PyArrayObject *) PyArray_ContiguousFromObject(yO, PyArray_FLOAT, 0, 1);
    if ((yA == NULL) || (PyArray_DIM(xA, 0) != nPts))
    {
      Py_DECREF(tA);
      Py_DECREF(xA);
      PyErr_Format(PyExc_RuntimeError, "Bad y");
      return NULL;
    }

    delta_xA = (PyArrayObject *) PyArray_ContiguousFromObject(delta_xO, PyArray_FLOAT, 0, 1);
    if ((delta_xA == NULL) || (PyArray_DIM(xA, 0) != nPts))
    {
      Py_DECREF(tA);
      Py_DECREF(xA);
      Py_DECREF(yA);
      PyErr_Format(PyExc_RuntimeError, "Bad delta_x");
      return NULL;
    }


    t = (int*)PyArray_DATA(tA);
    x = (float*)PyArray_DATA(xA);
    y = (float*)PyArray_DATA(yA);
    delta_x = (float*)PyArray_DATA(delta_xA);


    dims[0] = nPts;
    printf("nPts = %d\n", nPts);
    assignedA = PyArray_SimpleNew(1, dims, PyArray_INT32);
    if (assignedA == NULL)
    {
        Py_DECREF(tA);
        Py_DECREF(xA);
        Py_DECREF(yA);
        Py_DECREF(delta_xA);
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for objects");
        return NULL;
    }

    assigned = (int*)PyArray_DATA(assignedA);

    for (i=0; i < nPts; i++)
    {
        assigned[i] = 0;
        tMax = MAX(tMax, t[i]);
    }


    
    frameIndices = malloc((tMax + 10)*sizeof(int));
    for (i=0; i < (tMax + 10); i++)
    {
        frameIndices[i] = (nPts + 2);
    }

    for (i=0; i < nPts; i++)
    {
        t_i = t[i];
        for (j= t_last; j < (t_i + 1); j++)
        {
            frameIndices[j] = i;
        }
        t_last = t_i;
    }

/*
    for (i=0; i < (tMax + 10); i++)
    {
        printf("%d, ", frameIndices[i]);
    }

    printf("\n");
*/

    //i = 0;

    for (i=0; i < nPts; i++)
    {
        //printf("foo %d\n", i);
        if (assigned[i] == 0)
        {
            //assigned[i] = clumpNum;
            recDepth = 0;

            clump = findConnectedN(i, nPts, t, x, y, delta_x, frameIndices, assigned, clumpNum, nFrames, &recDepth);
            if (clump > 0)
            {
                assigned[i] = clump;
            }
            else
            {
                assigned[i] = clumpNum;
                clumpNum++;
            }
        }
    }


    free(frameIndices);

    Py_DECREF(tA);
    Py_DECREF(xA);
    Py_DECREF(yA);
    Py_DECREF(delta_xA);

    return (PyObject*) assignedA;

fail:
    return NULL;
}




static PyMethodDef deClumpMethods[] = {
    {"findClumps",  findClumps, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"findClumpsN",  findClumpsN, METH_VARARGS | METH_KEYWORDS,
    ""},
    
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initdeClump(void)
{
    PyObject *m;

    m = Py_InitModule("deClump", deClumpMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
