/*
##################
# distHist.c
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
#include <stdlib.h>
#define _USE_MATH_DEFINES //make M_PI available under msvcc
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

//#define SSE
#ifdef SSE
  #include <xmmintrin.h>
#endif



static PyObject * distanceHistogram(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *res = 0;
    int i1,i2;
    //int size[2];

    int x1_len;
    int x2_len;
    npy_intp outDimensions[1];
    int id, j;
    
    PyObject *ox1 =0;
    PyObject *oy1=0;
    PyObject *ox2 =0;
    PyObject *oy2=0;

    
    PyArrayObject* ax1;
    PyArrayObject* ay1;
    PyArrayObject* ax2;
    PyArrayObject* ay2;
    
    PyArrayObject* out;
    
    double *px1;
    double *px2;
    double *py1;
    double *py2;
    double *px2o;
    double *py2o;

    float d, dx, dy, x1, y1;
    
    /*parameters*/
    int nBins = 1000;
    float binSize = 1;
    float rBinSize = 1;

    /*End paramters*/

    
      
    
    static char *kwlist[] = {"x1", "y1","x2", "y2","nBins","binSize", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|if", kwlist,
         &ox1, &oy1, &ox2, &oy2, &nBins, &binSize))
        return NULL; 

    /* Do the calculations */ 
        
    ax1 = (PyArrayObject *) PyArray_ContiguousFromObject(ox1, PyArray_DOUBLE, 0, 1);
    if (ax1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x1");
      return NULL;
    }

    ay1 = (PyArrayObject *) PyArray_ContiguousFromObject(oy1, PyArray_DOUBLE, 0, 1);
    if (ay1 == NULL)
    {
      Py_DECREF(ax1);
      PyErr_Format(PyExc_RuntimeError, "Bad y1");
      return NULL;
    }

    ax2 = (PyArrayObject *) PyArray_ContiguousFromObject(ox2, PyArray_DOUBLE, 0, 1);
    if (ax2 == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      PyErr_Format(PyExc_RuntimeError, "Bad x2");
      return NULL;
    }

    ay2 = (PyArrayObject *) PyArray_ContiguousFromObject(oy2, PyArray_DOUBLE, 0, 1);
    if (ay2 == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      Py_DECREF(ax2);
      PyErr_Format(PyExc_RuntimeError, "Bad y2");
      return NULL;
    }
      
    
    px1 = (double*)ax1->data;
    py1 = (double*)ay1->data;
    px2 = (double*)ax2->data;
    py2 = (double*)ay2->data;

    rBinSize = 1.0/binSize;
    
    
    x1_len = PyArray_Size((PyObject*)ax1);
    x2_len = PyArray_Size((PyObject*)ax2);

    outDimensions[0] = nBins;
        
    out = (PyArrayObject*) PyArray_SimpleNew(1,outDimensions,PyArray_DOUBLE);
    if (out == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      Py_DECREF(ax2);
      Py_DECREF(ay2);
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      return NULL;
    }
    
    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];
    
    res = (double*) out->data;

    //Initialise our histogram
    for (j =0; j < nBins; j++)
    {
        res[j] = 0;
    }
    
        
    px2o = px2;
    py2o = py2;

    Py_BEGIN_ALLOW_THREADS

    for (i1 = 0; i1 < x1_len; i1++) //loop through the 1st set of points
      {            
        x1 = (float) px1[i1];
        y1 = (float) py1[i1];
        for (i2 = (i1+1); i2 < x2_len; i2++) 
	  {//loop through the second set of points - Note we only need to take the upper triangle of the distance matrix

            //calculate the delta
            dx = x1 - (float)px2o[i2];
            dy = y1 - (float)py2o[i2];
            
            //and the distance            
            d = sqrtf(dx*dx + dy*dy);

            //convert distance to bin index            
            id = (int)(d*rBinSize);

            if (id < nBins) res[id] += 1;
            
            
	  }
      }
    
    Py_END_ALLOW_THREADS
    
    Py_DECREF(ax1);
    Py_DECREF(ay1);
    Py_DECREF(ax2);
    Py_DECREF(ay2);
    
    return (PyObject*) out;
}

static PyObject * distanceHistogram3D(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *res = 0;
    int i1,i2;
    //int size[2];

    int x1_len;
    int x2_len;
    npy_intp outDimensions[1];
    int id, j;

    PyObject *ox1 =0;
    PyObject *oy1=0;
    PyObject *ox2 =0;
    PyObject *oy2=0;
    PyObject *oz1 =0;
    PyObject *oz2=0;


    PyArrayObject* ax1=NULL;
    PyArrayObject* ay1=NULL;
    PyArrayObject* ax2=NULL;
    PyArrayObject* ay2=NULL;
    PyArrayObject* az1=NULL;
    PyArrayObject* az2=NULL;

    PyArrayObject* out = NULL;

    double *px1;
    double *px2;
    double *py1;
    double *py2;
    double *pz1;
    double *pz2;

    float d, dx, dy, x1, y1, z1, dz;

    /*parameters*/
    int nBins = 1000;
    float binSize = 1;
    float rBinSize = 1;

    /*End paramters*/




    static char *kwlist[] = {"x1", "y1", "z1", "x2", "y2", "z2","nBins","binSize", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOO|if", kwlist,
         &ox1, &oy1, &oz1, &ox2, &oy2, &oz2, &nBins, &binSize))
        return NULL;

    /* Do the calculations */

    ax1 = (PyArrayObject *) PyArray_ContiguousFromObject(ox1, PyArray_DOUBLE, 0, 1);
    if (ax1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x1");
      goto fail;
    }

    ay1 = (PyArrayObject *) PyArray_ContiguousFromObject(oy1, PyArray_DOUBLE, 0, 1);
    if (ay1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad y1");
      goto fail;
    }

    ax2 = (PyArrayObject *) PyArray_ContiguousFromObject(ox2, PyArray_DOUBLE, 0, 1);
    if (ax2 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x2");
      goto fail;
    }

    ay2 = (PyArrayObject *) PyArray_ContiguousFromObject(oy2, PyArray_DOUBLE, 0, 1);
    if (ay2 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad y2");
      goto fail;
    }

    az1 = (PyArrayObject *) PyArray_ContiguousFromObject(oz1, PyArray_DOUBLE, 0, 1);
    if (az1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad z1");
      goto fail;
    }

    az2 = (PyArrayObject *) PyArray_ContiguousFromObject(oz2, PyArray_DOUBLE, 0, 1);
    if (az2 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad z2");
      goto fail;
    }


    px1 = (double*)ax1->data;
    py1 = (double*)ay1->data;
    px2 = (double*)ax2->data;
    py2 = (double*)ay2->data;

    pz1 = (double*)PyArray_DATA(az1);
    pz2 = (double*)PyArray_DATA(az2);

    rBinSize = 1.0/binSize;


    x1_len = PyArray_Size((PyObject*)ax1);
    x2_len = PyArray_Size((PyObject*)ax2);

    outDimensions[0] = nBins;

    out = (PyArrayObject*) PyArray_SimpleNew(1,outDimensions,PyArray_DOUBLE);
    if (out == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      goto fail;
    }

    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];

    res = (double*) out->data;

    //Initialise our histogram
    for (j =0; j < nBins; j++)
    {
        res[j] = 0;
    }

    Py_BEGIN_ALLOW_THREADS

    for (i1 = 0; i1 < x1_len; i1++) //loop through the 1st set of points
      {
        x1 = (float) px1[i1];
        y1 = (float) py1[i1];
        z1 = (float) pz1[i1];
        for (i2 = (i1+1); i2 < x2_len; i2++)
	  {//loop through the second set of points - Note we only need to take the upper triangle of the distance matrix

            //calculate the delta
            dx = x1 - (float)px2[i2];
            dy = y1 - (float)py2[i2];
            dz = z1 - (float)pz2[i2];

            //and the distance
            d = sqrtf(dx*dx + dy*dy + dz*dz);

            //convert distance to bin index
            id = (int)(d*rBinSize);

            if (id < nBins) res[id] += 1;


	  }
      }

    Py_END_ALLOW_THREADS

    Py_XDECREF(ax1);
    Py_XDECREF(ax2);
    Py_XDECREF(ay1);
    Py_XDECREF(ay2);
    Py_XDECREF(az1);
    Py_XDECREF(az2);

    return (PyObject*) out;

fail:
    Py_XDECREF(ax1);
    Py_XDECREF(ax2);
    Py_XDECREF(ay1);
    Py_XDECREF(ay2);
    Py_XDECREF(az1);
    Py_XDECREF(az2);

    Py_XDECREF(out);

    return NULL;
}


static PyObject * vectDistanceHistogram3D(PyObject *self, PyObject *args, PyObject *keywds)
{
    int *res = 0;
    int i1,i2;
    //int size[2];

    int x1_len;
    int x2_len;
    npy_intp outDimensions[3];
    int id, j;

    PyObject *ox1 =0;
    PyObject *oy1=0;
    PyObject *ox2 =0;
    PyObject *oy2=0;
    PyObject *oz1 =0;
    PyObject *oz2=0;


    PyArrayObject* ax1=NULL;
    PyArrayObject* ay1=NULL;
    PyArrayObject* ax2=NULL;
    PyArrayObject* ay2=NULL;
    PyArrayObject* az1=NULL;
    PyArrayObject* az2=NULL;

    PyArrayObject* out=NULL;

    double *px1;
    double *px2;
    double *py1;
    double *py2;
    double *pz1;
    double *pz2;

    float d, dx, dy, x1, y1, z1, dz;
    int r_id;
    int unwrapped_size;

    float theta, phi;

    /*parameters*/
    int n_bins_r = 100;
    float bin_size_r = 1;
    int n_bins_angle = 10;
    float rBinSize = 1;
    float bin_size_angle = 0;
    float rBinAngle = 0;

    /*End paramters*/




    static char *kwlist[] = {"x1", "y1", "z1", "x2", "y2", "z2","n_bins_r","bin_size_r", "n_bins_angle", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOO|ifi", kwlist,
         &ox1, &oy1, &oz1, &ox2, &oy2, &oz2, &n_bins_r, &bin_size_r, &n_bins_angle))
        return NULL;

    bin_size_angle = M_PI/n_bins_angle;
    rBinAngle = 1.0/bin_size_angle;
    /* Do the calculations */

    ax1 = (PyArrayObject *) PyArray_ContiguousFromObject(ox1, PyArray_DOUBLE, 0, 1);
    if (ax1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x1");
      goto fail_vd;
    }

    ay1 = (PyArrayObject *) PyArray_ContiguousFromObject(oy1, PyArray_DOUBLE, 0, 1);
    if (ay1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad y1");
      goto fail_vd;
    }

    ax2 = (PyArrayObject *) PyArray_ContiguousFromObject(ox2, PyArray_DOUBLE, 0, 1);
    if (ax2 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x2");
      goto fail_vd;
    }

    ay2 = (PyArrayObject *) PyArray_ContiguousFromObject(oy2, PyArray_DOUBLE, 0, 1);
    if (ay2 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad y2");
      goto fail_vd;
    }

    az1 = (PyArrayObject *) PyArray_ContiguousFromObject(oz1, PyArray_DOUBLE, 0, 1);
    if (az1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad z1");
      goto fail_vd;
    }

    az2 = (PyArrayObject *) PyArray_ContiguousFromObject(oz2, PyArray_DOUBLE, 0, 1);
    if (az2 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad z2");
      goto fail_vd;
    }


    /*px1 = (double*)ax1->data;
    py1 = (double*)ay1->data;
    px2 = (double*)ax2->data;
    py2 = (double*)ay2->data;*/

    px1 = (double*)PyArray_DATA(ax1);
    py1 = (double*)PyArray_DATA(ay1);
    px2 = (double*)PyArray_DATA(ax2);
    py2 = (double*)PyArray_DATA(ay2);

    pz1 = (double*)PyArray_DATA(az1);
    pz2 = (double*)PyArray_DATA(az2);

    rBinSize = 1.0/bin_size_r;


    x1_len = PyArray_Size((PyObject*)ax1);
    x2_len = PyArray_Size((PyObject*)ax2);

    outDimensions[0] = n_bins_r;
    outDimensions[1] = n_bins_angle;
    outDimensions[2] = n_bins_angle;

    //printf("x1_len: %d, x2_len: %d, od: [%d, %d, %d]\n", x1_len, x2_len, outDimensions[0], outDimensions[1], outDimensions[2]);

    out = (PyArrayObject*) PyArray_SimpleNew(3,outDimensions,PyArray_INT);
    if (out == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      goto fail_vd;
    }

    //Py_INCREF(out);

    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];

    res = (int*) PyArray_DATA(out); //out->data;

    unwrapped_size = n_bins_r*n_bins_angle*n_bins_angle;

    //Initialise our histogram
    for (j =0; j < unwrapped_size; j++)
    {
        res[j] = 0;
    }

    for (i1 = 0; i1 < x1_len; i1++) //loop through the 1st set of points
      {
        x1 = (float) px1[i1];
        y1 = (float) py1[i1];
        z1 = (float) pz1[i1];
        for (i2 = (i1+1); i2 < x2_len; i2++)
	    {//loop through the second set of points - Note we only need to take the upper triangle of the distance matrix

            //calculate the delta
            dx = x1 - (float)px2[i2];
            dy = y1 - (float)py2[i2];
            dz = z1 - (float)pz2[i2];

            //and the distance
            d = sqrtf(dx*dx + dy*dy + dz*dz);

            //calculate angles
            //theta = fmodf(atanf(dy/dx), M_PI);
            theta = atanf(dy/dx) + M_PI/2.0;
            //phi = fmodf(asinf(dz/d), M_PI);
            //phi = fmodf(atanf(dz/sqrtf(dx*dx + dy*dy)), M_PI);
            phi = atanf(dz/sqrtf(dx*dx + dy*dy)) + M_PI/2.0;

            //convert distance to bin index
            r_id = (int)(d*rBinSize);
            id = r_id*n_bins_angle*n_bins_angle + (int)(n_bins_angle*theta*rBinAngle) + (int)(phi*rBinAngle);

            if (id < 0) printf("id=%d, theta=%3.2f, phi=%3.2f\n", id, theta, phi);
            if (id < unwrapped_size) res[id] += 1;


	  }
      }


    Py_XDECREF(ax1);
    Py_XDECREF(ax2);
    Py_XDECREF(ay1);
    Py_XDECREF(ay2);
    Py_XDECREF(az1);
    Py_XDECREF(az2);

    //Py_INCREF(out);

    return (PyObject*) out;

fail_vd:
    printf("failure\n");
    Py_XDECREF(ax1);
    Py_XDECREF(ax2);
    Py_XDECREF(ay1);
    Py_XDECREF(ay2);
    Py_XDECREF(az1);
    Py_XDECREF(az2);

    Py_XDECREF(out);

    return NULL;
}



static PyObject * distanceProduct(PyObject *self, PyObject *args, PyObject *keywds)
{
    double res = 0;
    int i1,i2;
    //int size[2];

    int x1_len;

    PyObject *ox1 =0;
    PyObject *oy1=0;

    PyArrayObject* ax1;
    PyArrayObject* ay1;

    double *px1;
    double *py1;
    double *px2o;
    double *py2o;

    float d, dx, dy, x1, y1;


    static char *kwlist[] = {"x", "y", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist,
         &ox1, &oy1))
        return NULL;

    /* Do the calculations */

    ax1 = (PyArrayObject *) PyArray_ContiguousFromObject(ox1, PyArray_DOUBLE, 0, 1);
    if (ax1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x");
      return NULL;
    }

    ay1 = (PyArrayObject *) PyArray_ContiguousFromObject(oy1, PyArray_DOUBLE, 0, 1);
    if (ay1 == NULL)
    {
      Py_DECREF(ax1);
      PyErr_Format(PyExc_RuntimeError, "Bad y");
      return NULL;
    }

    px1 = (double*)ax1->data;
    py1 = (double*)ay1->data;

    x1_len = PyArray_Size((PyObject*)ax1);

    px2o = px1;
    py2o = py1;

    for (i1 = 0; i1 < x1_len; i1++)
      {
        x1 = (float) *px1;
        y1 = (float) *py1;
        for (i2 = (i1+1); i2 < x1_len; i2++)
	  {
            //if (i2 != i1){
                dx = x1 - (float)px2o[i2];//*px2;
                dy = y1 - (float)py2o[i2];//*py2;

                d = sqrtf(dx*dx + dy*dy);
                //printf("%f\n", d);

                res += (d);
            //}

	  }
        px1++;
        py1++;

        //reset inner pointers
        //px2 = px2o;
        //py2 = py2o;
      }


    Py_DECREF(ax1);
    Py_DECREF(ay1);

    //return (PyObject*) out;
    return Py_BuildValue("d", res);
}


#ifdef SSE
static PyObject * distanceHistogramRS(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *res = 0;
    int i1,i2;
    //int size[2];

    int x1_len;
    int x2_len;
    npy_intp outDimensions[1];
    int id, j;

    PyObject *ox1 =0;
    PyObject *oy1=0;
    PyObject *ox2 =0;
    PyObject *oy2=0;


    PyArrayObject* ax1;
    PyArrayObject* ay1;
    PyArrayObject* ax2;
    PyArrayObject* ay2;

    PyArrayObject* out;

    double *px1;
    double *px2;
    double *py1;
    double *py2;
    double *px2o;
    double *py2o;

    float d, dx, dy, x1, y1;
    float(d2f);

    /*parameters*/
    int nBins = 1000;
    float binSize = 1;
    float rBinSize = 1;

    /*End paramters*/

    __m128 d2;




    static char *kwlist[] = {"x1", "y1","x2", "y2","nBins","binSize", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|if", kwlist,
         &ox1, &oy1, &ox2, &oy2, &nBins, &binSize))
        return NULL;

    /* Do the calculations */

    ax1 = (PyArrayObject *) PyArray_ContiguousFromObject(ox1, PyArray_DOUBLE, 0, 1);
    if (ax1 == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x1");
      return NULL;
    }

    ay1 = (PyArrayObject *) PyArray_ContiguousFromObject(oy1, PyArray_DOUBLE, 0, 1);
    if (ay1 == NULL)
    {
      Py_DECREF(ax1);
      PyErr_Format(PyExc_RuntimeError, "Bad y1");
      return NULL;
    }

    ax2 = (PyArrayObject *) PyArray_ContiguousFromObject(ox2, PyArray_DOUBLE, 0, 1);
    if (ax2 == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      PyErr_Format(PyExc_RuntimeError, "Bad x2");
      return NULL;
    }

    ay2 = (PyArrayObject *) PyArray_ContiguousFromObject(oy2, PyArray_DOUBLE, 0, 1);
    if (ay2 == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      Py_DECREF(ax2);
      PyErr_Format(PyExc_RuntimeError, "Bad y2");
      return NULL;
    }


    px1 = (double*)ax1->data;
    py1 = (double*)ay1->data;
    px2 = (double*)ax2->data;
    py2 = (double*)ay2->data;

    rBinSize = 1.0/binSize;


    x1_len = PyArray_Size((PyObject*)ax1);
    x2_len = PyArray_Size((PyObject*)ax2);

    outDimensions[0] = nBins;

    out = (PyArrayObject*) PyArray_SimpleNew(1,outDimensions,PyArray_DOUBLE);
    if (out == NULL)
    {
      Py_DECREF(ax1);
      Py_DECREF(ay1);
      Py_DECREF(ax2);
      Py_DECREF(ay2);
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      return NULL;
    }

    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];

    res = (double*) out->data;

    //Initialise our histogram
    for (j =0; j < nBins; j++)
    {
        res[j] = 0;
    }


    px2o = px2;
    py2o = py2;

    for (i1 = 0; i1 < x1_len; i1++)
      {
        x1 = (float) *px1;
        y1 = (float) *py1;
        for (i2 = (i1+1); i2 < x2_len; i2++)
	  {
            //dx = *px1 - *px2;
            //dy = *py1 - *py2;
            dx = x1 - (float)px2o[i2];//*px2;
            dy = y1 - (float)px2o[i2];//*py2;
            d2f = dx*dx + dy*dy;
            //d = sqrtf(dx*dx + dy*dy);

            d2 = _mm_load_ss(&d2f);
            //_mm_store_ss(&d, _mm_mul_ss( d2, _mm_rsqrt_ss( d2 ) ) );
            _mm_store_ss(&d,  _mm_rsqrt_ss( d2 ) );
            // Newton-Raphson step
            d *= ((3.0f - d * d * d2f) * 0.5f)*d2f;


            //id = (int)floorf(d*rBinSize);
            id = (int)d;

            if (id < nBins) res[id] += 1;
            //res[id] += (id < nBins);

            //px2++;
            //py2++;

	  }
        px1++;
        py1++;

        //reset inner pointers
        //px2 = px2o;
        //py2 = py2o;
      }


    Py_DECREF(ax1);
    Py_DECREF(ay1);
    Py_DECREF(ax2);
    Py_DECREF(ay2);

    return (PyObject*) out;
}

#endif


static PyObject * meanSquareDistHist(PyObject *self, PyObject *args, PyObject *keywds)
{
    int *ndists = 0;
    double *res = 0;
    int i,j;
    //int size[2];

    int x_len;
    npy_intp outDimensions[1];
    int id;

    PyObject *ox = 0;
    PyObject *oy = 0;
    PyObject *ot = 0;

    PyArrayObject* ax;
    PyArrayObject* ay;
    PyArrayObject* at;

    PyArrayObject* out;

    double *px;
    double *py;
    double *pt;

    double t_i, x_i, y_i, dx, dy, d, dt;

    /*parameters*/
    int nBins = 1000;
    double binSize = 1;

    /*End paramters*/




    static char *kwlist[] = {"x", "y","t","nBins","binSize", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOO|id", kwlist,
         &ox, &oy, &ot, &nBins, &binSize))
        return NULL;

    /* Do the calculations */

    ax = (PyArrayObject *) PyArray_ContiguousFromObject(ox, PyArray_DOUBLE, 0, 1);
    if (ax == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad x");
      return NULL;
    }

    ay = (PyArrayObject *) PyArray_ContiguousFromObject(oy, PyArray_DOUBLE, 0, 1);
    if (ay == NULL)
    {
      Py_DECREF(ax);
      PyErr_Format(PyExc_RuntimeError, "Bad y");
      return NULL;
    }

    at = (PyArrayObject *) PyArray_ContiguousFromObject(ot, PyArray_DOUBLE, 0, 1);
    if (at == NULL)
    {
      Py_DECREF(ax);
      Py_DECREF(ay);
      PyErr_Format(PyExc_RuntimeError, "Bad t");
      return NULL;
    }


    px = (double*)PyArray_DATA(ax);
    py = (double*)PyArray_DATA(ay);
    pt = (double*)PyArray_DATA(at);

    x_len = PyArray_Size((PyObject*)ax);

    ndists = (int*)malloc(sizeof(int)*nBins);
    if (ndists == NULL)
    {
      Py_DECREF(ax);
      Py_DECREF(ay);
      Py_DECREF(at);

      PyErr_Format(PyExc_RuntimeError, "Error allocating working array");
      return NULL;
    }

    outDimensions[0] = nBins;

    out = (PyArrayObject*) PyArray_SimpleNew(1,outDimensions,PyArray_DOUBLE);
    if (out == NULL)
    {
      Py_DECREF(ax);
      Py_DECREF(ay);
      Py_DECREF(at);
   
      PyErr_Format(PyExc_RuntimeError, "Error allocating output array");
      return NULL;
    }

    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];

    res = (double*) PyArray_DATA(out);

    //Initialise our histogram
    for (j =0; j < nBins; j++)
    {
        res[j] = 0;
        ndists[j] = 0;
    }


    //pxo = px;
    //pyo = py;

    for (i = 0; i < x_len; i++)
      {
        t_i = pt[i];
        x_i = px[i];
        y_i = py[i];
	for (j = (i+1); j < x_len; j++)
	  {
            dx = x_i - px[j];
            dy = y_i - py[j];
            d = dx*dx + dy*dy;

            dt = fabs(pt[j] - t_i);

            id = (int)floor(dt/binSize);

            if (id < nBins)
            {
                res[id] += d;
                ndists[id] += 1;
            }

            //printf("d: %f\t id: %d\t ij: %d,%d\tt_i: %d\ttj: %d\t dt: %f\n", d, id, i, j, t_i, pt[j]);

	  }
      }

    //Normalise the bins
    for (j =0; j < nBins; j++)
    {
        res[j] /= ndists[j];
    }

    free(ndists);

    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_DECREF(at);

    return (PyObject*) out;
}


static PyMethodDef distHistMethods[] = {
    {"distanceHistogram",  (PyCFunction)distanceHistogram, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
    {"distanceHistogram3D",  (PyCFunction)distanceHistogram3D, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'nBins'= 1e3, 'binSize' = 1"},
    {"distanceProduct",  (PyCFunction)distanceProduct, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
#ifdef SSE
    {"distanceHistogramRS",  (PyCFunction)distanceHistogramRS, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'x2', 'y2', 'nBins'= 1e3, 'binSize' = 1"},
#endif
    {"msdHistogram",  (PyCFunction)meanSquareDistHist, METH_VARARGS | METH_KEYWORDS,
    "calculate a histogram of msd vs time.\n. Arguments are: 'x', 'y', 't', 'nBins'= 1e3, 'binSize' = 1"},
    {"vectDistanceHistogram3D",  (PyCFunction)vectDistanceHistogram3D, METH_VARARGS | METH_KEYWORDS,
    "Generate a histogram of pairwise distances between two sets of points.\n. Arguments are: 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'nBins'= 1e3, 'binSize' = 1"},

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "distHist",     /* m_name */
        "",  /* m_doc */
        -1,                  /* m_size */
        distHistMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_distHist(void)
{
	PyObject *m;
    m = PyModule_Create(&moduledef);
    import_array()
    return m;
}

#else
PyMODINIT_FUNC initdistHist(void)
{
    PyObject *m;

    m = Py_InitModule("distHist", distHistMethods);
    import_array()
}
#endif
