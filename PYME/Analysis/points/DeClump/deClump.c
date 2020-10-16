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

    return 0;

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

    //look backwards at already assigned frames
    //As frameIndices[t_i] gives you the first event *after* the timepoint t_i, nFrames=1 means look in current frame
    //etc ...
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

    npy_intp dims[2];
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

//fail:
//    return NULL;
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

    npy_intp dims[2];
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


    /*
    set frame indices up to point give the starting frame for each timePoint

    nb tlast starts at 0

    e.g. if t = [0,0,0,1,1,2,2,2,3,4,4,6,6] ,
    frameIndices = [3,5,8,9,11,11,12]
    */
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

//fail:
 //   return NULL;
}

/*
Aggregate data into clumps by taking a weighted mean. This assumes data has been sorted by clumpIndex
*/
static PyObject * aggregateWeightedMean(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *clumpIDO = 0;
    PyObject *varO = 0;
    PyObject *sigmaO = 0;

    PyObject* out = 0;

    PyArrayObject *clumpIDA = 0;
    PyArrayObject *varA = 0;
    PyArrayObject *sigmaA = 0;

    int *clumpIDs = 0;
    float *vars = 0;
    float *sigs = 0;


    PyObject * outVarA=0;
    PyObject * outSigA=0;


    int nPts = 0;
    int nClumps = 0;
    int currentClump = -1;

    int i=0;
    //int j=0;

    float *outVar = 0;
    float *outSig = 0;

    float iws, weight_sum, var_sum, w;

    npy_intp dims[2];

    static char *kwlist[] = {"nClumps", "clumpIDs", "var", "sig" , NULL};

    dims[0] = 0;
    dims[1] = 0;


    if (!PyArg_ParseTupleAndKeywords(args, keywds, "iOOO", kwlist,
         &nClumps, &clumpIDO, &varO, &sigmaO))
        return NULL;

    clumpIDA = (PyArrayObject *) PyArray_ContiguousFromObject(clumpIDO, PyArray_INT, 0, 1);
    if (clumpIDA == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad clumpIDs");
      return NULL;
    }

    nPts = PyArray_DIM(clumpIDA, 0);

    varA = (PyArrayObject *) PyArray_ContiguousFromObject(varO, PyArray_FLOAT, 0, 1);
    if ((varA == NULL) || (PyArray_DIM(varA, 0) != nPts))
    {
      Py_DECREF(clumpIDA);
      PyErr_Format(PyExc_RuntimeError, "Bad var");
      return NULL;
    }

    sigmaA = (PyArrayObject *) PyArray_ContiguousFromObject(sigmaO, PyArray_FLOAT, 0, 1);
    if ((sigmaA == NULL) || (PyArray_DIM(sigmaA, 0) != nPts))
    {
      Py_DECREF(clumpIDA);
      Py_DECREF(varA);
      PyErr_Format(PyExc_RuntimeError, "Bad sigma");
      return NULL;
    }

    //retrieve pointers to the various data arrays
    clumpIDs = (int*)PyArray_DATA(clumpIDA);
    vars = (float*)PyArray_DATA(varA);
    sigs = (float*)PyArray_DATA(sigmaA);

    dims[0] = nClumps;

    outVarA = PyArray_SimpleNew(1, dims, PyArray_FLOAT);
    if (outVarA == NULL)
    {
        Py_DECREF(clumpIDA);
        Py_DECREF(varA);
        Py_DECREF(sigmaA);
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for clumped output");
        return NULL;
    }

    outSigA = PyArray_SimpleNew(1, dims, PyArray_FLOAT);
    if (outSigA == NULL)
    {
        Py_DECREF(clumpIDA);
        Py_DECREF(varA);
        Py_DECREF(sigmaA);
        Py_DECREF(outVarA);
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for clumped errors");
        return NULL;
    }

    //initialise memory in output arrays
    PyArray_FILLWBYTE(outVarA, 0);
    PyArray_FILLWBYTE(outSigA, 0);

    outVar = (float*)PyArray_DATA(outVarA);
    outSig = (float*)PyArray_DATA(outSigA);

    /*
    for (i=0; i < nPts; i++)
    {
        outVar[i] = 0;
        outSig[i] = -1e4;
    }
    */


    //j = 0;
    for (i=0; i < nPts; i++)
    {
        if (currentClump != clumpIDs[i]){
            //We have moved on to the next clump
            if (currentClump >= 0)
            {
                if (weight_sum == 0){
                    outVar[currentClump] = 0;
                    outSig[currentClump] = -1e4;
                } else {
                    iws = 1.0/weight_sum;
                    outVar[currentClump] = var_sum*iws;
                    outSig[currentClump] = sqrtf(iws);
                }
            }

            weight_sum = 0;
            var_sum = 0;

            currentClump = clumpIDs[i];
        }

        w = 1.0/(sigs[i]*sigs[i]);
        weight_sum += w;
        var_sum += w*vars[i];
    }

    if (currentClump >= 0)
    {
        if (weight_sum == 0){
            outVar[currentClump] = 0;
            outSig[currentClump] = -1e4;
        } else {
            iws = 1.0/weight_sum;
            outVar[currentClump] = var_sum*iws;
            outSig[currentClump] = sqrtf(iws);
        }
    }


    Py_DECREF(clumpIDA);
    Py_DECREF(varA);
    Py_DECREF(sigmaA);

    out = Py_BuildValue("(O,O)", (PyObject*) outVarA, (PyObject*) outSigA);

    Py_XDECREF(outVarA);
    Py_XDECREF(outSigA);
    return out;

/*fail:
    Py_XDECREF(clumpIDA);
    Py_XDECREF(varA);
    Py_XDECREF(sigmaA);
    Py_XDECREF(outVarA);
    Py_XDECREF(outSigA);


    return NULL;*/
}

/*
Aggregate data into clumps by taking an unweighted mean. This assumes data has been sorted by clumpIndex
*/
static PyObject * aggregateMean(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *clumpIDO = 0;
    PyObject *varO = 0;
    //PyObject *sigmaO = 0;

    PyArrayObject *clumpIDA = 0;
    PyArrayObject *varA = 0;
    //PyArrayObject *sigmaA = 0;

    int *clumpIDs = 0;
    float *vars = 0;
    //float *sigs = 0;


    PyObject * outVarA=0;
    //PyObject * outSigA=0;


    int nPts = 0;
    int nClumps = 0;
    int currentClump = -1;

    int i=0;
    //int j=0;

    float *outVar = 0;
    //float *outSig = 0;

    float iws, weight_sum, var_sum, w;

    npy_intp dims[2];

    static char *kwlist[] = {"nClumps", "clumpIDs", "var", NULL};

    dims[0] = 0;
    dims[1] = 0;


    if (!PyArg_ParseTupleAndKeywords(args, keywds, "iOO", kwlist,
         &nClumps, &clumpIDO, &varO))
        return NULL;

    clumpIDA = (PyArrayObject *) PyArray_ContiguousFromObject(clumpIDO, PyArray_INT, 0, 1);
    if (clumpIDA == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad clumpIDs");
      return NULL;
    }

    nPts = PyArray_DIM(clumpIDA, 0);

    varA = (PyArrayObject *) PyArray_ContiguousFromObject(varO, PyArray_FLOAT, 0, 1);
    if ((varA == NULL) || (PyArray_DIM(varA, 0) != nPts))
    {
      Py_DECREF(clumpIDA);
      PyErr_Format(PyExc_RuntimeError, "Bad var");
      return NULL;
    }

    //retrieve pointers to the various data arrays
    clumpIDs = (int*)PyArray_DATA(clumpIDA);
    vars = (float*)PyArray_DATA(varA);

    dims[0] = nClumps;

    outVarA = PyArray_SimpleNew(1, dims, PyArray_FLOAT);
    if (outVarA == NULL)
    {
        Py_DECREF(clumpIDA);
        Py_DECREF(varA);
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for clumped output");
        return NULL;
    }

    outVar = (float*)PyArray_DATA(outVarA);

    //initialise memory in output arrays
    PyArray_FILLWBYTE(outVarA, 0);
    //PyArray_FILLWBYTE(outSigA, 0);

    /*
    for (i=0; i < nPts; i++)
    {
        outVar[i] = 0;
        outSig[i] = -1e4;
    }
    */


    //j = 0;
    for (i=0; i < nPts; i++)
    {
        if (currentClump != clumpIDs[i]){
            //We have moved on to the next clump
            if (currentClump >= 0)
            {
                iws = 1.0/weight_sum;
                outVar[currentClump] = var_sum*iws;
            }

            weight_sum = 0;
            var_sum = 0;

            currentClump = clumpIDs[i];
        }

        w = 1.0;
        weight_sum += w;
        var_sum += w*vars[i];
    }

    if (currentClump >= 0)
    {
        iws = 1.0/weight_sum;
        outVar[currentClump] = var_sum*iws;
    }


    Py_DECREF(clumpIDA);
    Py_DECREF(varA);

    return (PyObject*) outVarA;

/*fail:
    Py_XDECREF(clumpIDA);
    Py_XDECREF(varA);
    Py_XDECREF(sigmaA);
    Py_XDECREF(outVarA);
    Py_XDECREF(outSigA);


    return NULL;*/
}

/*
Aggregate data into clumps by taking the minimum. This assumes data has been sorted by clumpIndex
*/
static PyObject * aggregateMin(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *clumpIDO = 0;
    PyObject *varO = 0;
    //PyObject *sigmaO = 0;

    PyArrayObject *clumpIDA = 0;
    PyArrayObject *varA = 0;
    //PyArrayObject *sigmaA = 0;

    int *clumpIDs = 0;
    float *vars = 0;
    //float *sigs = 0;


    PyObject * outVarA=0;
    //PyObject * outSigA=0;


    int nPts = 0;
    int nClumps = 0;
    int currentClump = -1;

    int i=0;
    //int j=0;

    float *outVar = 0;
    //float *outSig = 0;

    float var_min;

    npy_intp dims[2];

    static char *kwlist[] = {"nClumps", "clumpIDs", "var", NULL};

    dims[0] = 0;
    dims[1] = 0;


    if (!PyArg_ParseTupleAndKeywords(args, keywds, "iOO", kwlist,
         &nClumps, &clumpIDO, &varO))
        return NULL;

    clumpIDA = (PyArrayObject *) PyArray_ContiguousFromObject(clumpIDO, PyArray_INT, 0, 1);
    if (clumpIDA == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad clumpIDs");
      return NULL;
    }

    nPts = PyArray_DIM(clumpIDA, 0);

    varA = (PyArrayObject *) PyArray_ContiguousFromObject(varO, PyArray_FLOAT, 0, 1);
    if ((varA == NULL) || (PyArray_DIM(varA, 0) != nPts))
    {
      Py_DECREF(clumpIDA);
      PyErr_Format(PyExc_RuntimeError, "Bad var");
      return NULL;
    }

    //retrieve pointers to the various data arrays
    clumpIDs = (int*)PyArray_DATA(clumpIDA);
    vars = (float*)PyArray_DATA(varA);

    dims[0] = nClumps;

    outVarA = PyArray_SimpleNew(1, dims, PyArray_FLOAT);
    if (outVarA == NULL)
    {
        Py_DECREF(clumpIDA);
        Py_DECREF(varA);
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for clumped output");
        return NULL;
    }

    outVar = (float*)PyArray_DATA(outVarA);

        //initialise memory in output arrays
    PyArray_FILLWBYTE(outVarA, 0);
    //PyArray_FILLWBYTE(outSigA, 0);

    /*
    for (i=0; i < nPts; i++)
    {
        outVar[i] = 0;
        outSig[i] = -1e4;
    }
    */


    //j = 0;
    for (i=0; i < nPts; i++)
    {
        if (currentClump != clumpIDs[i]){
            //We have moved on to the next clump
            if (currentClump >= 0)
            {
                outVar[currentClump] = var_min;
            }

            var_min = 1e9;

            currentClump = clumpIDs[i];
        }

        var_min = MIN(var_min, vars[i]);
    }

    if (currentClump >= 0)
    {
        outVar[currentClump] = var_min;
    }


    Py_DECREF(clumpIDA);
    Py_DECREF(varA);

    return (PyObject*) outVarA;

/*fail:
    Py_XDECREF(clumpIDA);
    Py_XDECREF(varA);
    Py_XDECREF(sigmaA);
    Py_XDECREF(outVarA);
    Py_XDECREF(outSigA);


    return NULL;*/
}

static PyObject * aggregateSum(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *clumpIDO = 0;
    PyObject *varO = 0;
    //PyObject *sigmaO = 0;

    PyArrayObject *clumpIDA = 0;
    PyArrayObject *varA = 0;
    //PyArrayObject *sigmaA = 0;

    int *clumpIDs = 0;
    float *vars = 0;
    //float *sigs = 0;


    PyObject * outVarA=0;
    //PyObject * outSigA=0;


    int nPts = 0;
    int nClumps = 0;
    int currentClump = -1;

    int i=0;
    //int j=0;

    float *outVar = 0;
    //float *outSig = 0;

    float var_sum;

    npy_intp dims[2];

    static char *kwlist[] = {"nClumps", "clumpIDs", "var", NULL};

    dims[0] = 0;
    dims[1] = 0;


    if (!PyArg_ParseTupleAndKeywords(args, keywds, "iOO", kwlist,
         &nClumps, &clumpIDO, &varO))
        return NULL;

    clumpIDA = (PyArrayObject *) PyArray_ContiguousFromObject(clumpIDO, PyArray_INT, 0, 1);
    if (clumpIDA == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad clumpIDs");
      return NULL;
    }

    nPts = PyArray_DIM(clumpIDA, 0);

    varA = (PyArrayObject *) PyArray_ContiguousFromObject(varO, PyArray_FLOAT, 0, 1);
    if ((varA == NULL) || (PyArray_DIM(varA, 0) != nPts))
    {
      Py_DECREF(clumpIDA);
      PyErr_Format(PyExc_RuntimeError, "Bad var");
      return NULL;
    }

    //retrieve pointers to the various data arrays
    clumpIDs = (int*)PyArray_DATA(clumpIDA);
    vars = (float*)PyArray_DATA(varA);

    dims[0] = nClumps;

    outVarA = PyArray_SimpleNew(1, dims, PyArray_FLOAT);
    if (outVarA == NULL)
    {
        Py_DECREF(clumpIDA);
        Py_DECREF(varA);
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for clumped output");
        return NULL;
    }

        //initialise memory in output arrays
    PyArray_FILLWBYTE(outVarA, 0);
    //PyArray_FILLWBYTE(outSigA, 0);

    outVar = (float*)PyArray_DATA(outVarA);

    /*
    for (i=0; i < nPts; i++)
    {
        outVar[i] = 0;
        outSig[i] = -1e4;
    }
    */


    //j = 0;
    for (i=0; i < nPts; i++)
    {
        if (currentClump != clumpIDs[i]){
            //We have moved on to the next clump
            if (currentClump >= 0)
            {
                outVar[currentClump] = var_sum;
            }

            var_sum = 0;

            currentClump = clumpIDs[i];
        }

        var_sum +=  vars[i];
    }

    if (currentClump >= 0)
    {
        outVar[currentClump] = var_sum;
    }


    Py_DECREF(clumpIDA);
    Py_DECREF(varA);

    return (PyObject*) outVarA;

/*fail:
    Py_XDECREF(clumpIDA);
    Py_XDECREF(varA);
    Py_XDECREF(sigmaA);
    Py_XDECREF(outVarA);
    Py_XDECREF(outSigA);


    return NULL;*/
}


static PyMethodDef deClumpMethods[] = {
    {"findClumps",  findClumps, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"findClumpsN",  findClumpsN, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"aggregateWeightedMean",  aggregateWeightedMean, METH_VARARGS | METH_KEYWORDS,
    "Aggregate data into clumps by taking a weighted mean. This assumes data has been sorted by clumpIndex."},
    {"aggregateMean",  aggregateMean, METH_VARARGS | METH_KEYWORDS,
    "Aggregate data into clumps by taking an unweighted mean. This assumes data has been sorted by clumpIndex."},
    {"aggregateMin",  aggregateMin, METH_VARARGS | METH_KEYWORDS,
    "Aggregate data into clumps by taking the minimum. This assumes data has been sorted by clumpIndex."},
    {"aggregateSum",  aggregateSum, METH_VARARGS | METH_KEYWORDS,
    "Aggregate data into clumps by taking the sum. This assumes data has been sorted by clumpIndex."},
    
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "deClump",     /* m_name */
        "clump localizations together)",  /* m_doc */
        -1,                  /* m_size */
        deClumpMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_deClump(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else

PyMODINIT_FUNC initdeClump(void)
{
    PyObject *m;

    m = Py_InitModule("deClump", deClumpMethods);
    import_array();

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
#endif