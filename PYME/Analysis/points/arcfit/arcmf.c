/*
##################
# gauss_ap.c
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

#define MIN(a, b) ((a<b) ? a : b) 
#define MAX(a, b) ((a>b) ? a : b)

//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wmacro-redefined"

static PyObject * arcmf(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *res = 0;  
    int i; 
    npy_intp size[1];
    
    PyObject *oX =0;
    PyObject *oY=0;

    PyObject *osX =0;
    PyObject *osY=0;
        

    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    PyArrayObject* sXvals;
    PyArrayObject* sYvals;
    
    PyArrayObject* out;
    
    double *pXvals;
    double *pYvals;
    double *psXvals;
    double *psYvals;
    
    
    /*parameters*/
    //double A = 1;
    double x0 = 0;
    double y0 = 0;
    double dx = 1;
    double dy = 1;
    double c = 0;

    /*End paramters*/

    double r;
    double d, x1, y1, rhx, rhy, a, xp, yp, dis1, dis2;
    double r2, dtx, dty, x, y, arhx, arhy,isx, isy;

      
    
    static char *kwlist[] = {"x", "y", "sx", "sy","x0", "y0","dy","dy","c", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|ddddd", kwlist, 
         &oX, &oY, &osX, &osY, &x0, &y0, &dx, &dy, &c))
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

    sXvals = (PyArrayObject *) PyArray_ContiguousFromObject(osX, PyArray_DOUBLE, 0, 1);
    if (sXvals == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
        PyErr_Format(PyExc_RuntimeError, "Bad sX");
        return NULL;
    }

    sYvals = (PyArrayObject *) PyArray_ContiguousFromObject(osY, PyArray_DOUBLE, 0, 1);
    if (sYvals == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
        Py_DECREF(sXvals);
        PyErr_Format(PyExc_RuntimeError, "Bad sY");
        return NULL;
    }
    
    
    
    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    psXvals = (double*)sXvals->data;
    psYvals = (double*)sYvals->data;
    
    size[0] = PyArray_Size((PyObject*)Xvals);

    r = 1./c;
    d = sqrt(dx*dx + dy*dy);
    
    x1 = x0 + dx - r*dx/d;
    y1 = y0 + dy - r*dy/d;

    out = (PyArrayObject*) PyArray_New(&PyArray_Type, 1,size,NPY_DOUBLE, NULL, NULL, 0, 1, NULL);
    if (out == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
        Py_DECREF(sXvals);
        Py_DECREF(sYvals);
        PyErr_Format(PyExc_RuntimeError, "Failed to allocate memory");
        return NULL;    
    }
    

    res = (double*) PyArray_DATA(out);
    
    r2 = r*r;
        
    for (i = 0; i < size[0]; i++)
      {            
        x = pXvals[i];
        y = pYvals[i];
        isx = psXvals[i];
        isy = psYvals[i];
        
        rhx = x - x1;
        rhy = y - y1;

        a = r2/(rhx*rhx + rhy*rhy);
        arhx = a*rhx;
        arhy = a*rhy;

        xp = x1 + arhx;
        yp = y1 + arhy;

        dtx = (x-xp)*isx;
        dty = (y-yp)*isy;

        dis1 = dtx*dtx + dty*dty;

        xp = x1 - arhx;
        yp = y1 - arhy;

        dtx = (x-xp)*isx;
        dty = (y-yp)*isy;

        dis2 = dtx*dtx + dty*dty;

        *res = sqrt(MIN(dis1, dis2));

	   res++;
        
      }
    
    
    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    Py_DECREF(sXvals);
    Py_DECREF(sYvals);
    
    return (PyObject*) out;
}

//time varying arc missfit
static PyObject * arcmft(PyObject *self, PyObject *args, PyObject *keywds) 
{
    double *res = 0;  
    int i; 
    npy_intp size[1];
    
    PyObject *oX =0;
    PyObject *oY=0;

    PyObject *osX =0;
    PyObject *osY=0;

    PyObject *oT =0;    

    PyArrayObject *Xvals=0, *Yvals=0, *sXvals=0, *sYvals=0;
    PyArrayObject* Tvals=0;
    
    PyArrayObject* out=NULL;
    
    double *pXvals;
    double *pYvals;
    double *psXvals;
    double *psYvals;
    double *pTvals;
    
    
    /*parameters*/
    //double A = 1;
    double x0 = 0;
    double y0 = 0;
    double dx = 1;
    double dy = 1;
    double c = 0;
    double dxt = 0;
    double dyt = 0;

    /*End paramters*/

    double r;
    double d, x1, y1, rhx, rhy, a, xp, yp, dis1, dis2, t;
    double r2, dtx, dty, x, y, arhx, arhy,isx, isy;

      
    
    static char *kwlist[] = {"x", "y", "sx", "sy", "t","x0", "y0","dy","dy","c", "dxt", "dyt", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOO|ddddddd", kwlist, 
         &oX, &oY, &osX, &osY, &oT, &x0, &y0, &dx, &dy, &c, &dxt, &dyt))
        return NULL; 

    /* Do the calculations */
    #define ABORT(msg) {\
        PyErr_Format(PyExc_RuntimeError, msg);\
        goto FINALIZE_arcmft;\
        }
        
    Xvals = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_DOUBLE, 0, 1);
    if (Xvals == NULL) ABORT("Bad X")
    
    Yvals = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_DOUBLE, 0, 1);
    if (Yvals == NULL) ABORT("Bad Y")

    sXvals = (PyArrayObject *) PyArray_ContiguousFromObject(osX, PyArray_DOUBLE, 0, 1);
    if (sXvals == NULL) ABORT("Bad sX")


    sYvals = (PyArrayObject *) PyArray_ContiguousFromObject(osY, PyArray_DOUBLE, 0, 1);
    if (sYvals == NULL) ABORT("Bad sY")
    
    Tvals = (PyArrayObject *) PyArray_ContiguousFromObject(oT, PyArray_DOUBLE, 0, 1);
    if (Tvals == NULL) ABORT("Bad T")
    
    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    psXvals = (double*)sXvals->data;
    psYvals = (double*)sYvals->data;
    pTvals = (double*)Tvals->data;
    
    size[0] = PyArray_Size((PyObject*)Xvals);

    r = 1./c;
    d = sqrt(dx*dx + dy*dy);
    
    x1 = x0 + dx - r*dx/d;
    y1 = y0 + dy - r*dy/d;

    out = (PyArrayObject*) PyArray_New(&PyArray_Type, 1,size,NPY_DOUBLE, NULL, NULL, 0, 1, NULL);
    if (out == NULL) ABORT("Failed to allocate memory for output")

    res = (double*) PyArray_DATA(out);
    r2 = r*r;
        
    for (i = 0; i < size[0]; i++)
      {            
        x = pXvals[i];
        y = pYvals[i];
        isx = psXvals[i];
        isy = psYvals[i];
        t = pTvals[i]; 
        
        rhx = x - x1 + dxt*t;
        rhy = y - y1 + dyt*t;

        a = r2/(rhx*rhx + rhy*rhy);
        arhx = a*rhx;
        arhy = a*rhy;

        xp = x1 + arhx;
        yp = y1 + arhy;

        dtx = (x-xp)*isx;
        dty = (y-yp)*isy;

        dis1 = dtx*dtx + dty*dty;

        xp = x1 - arhx;
        yp = y1 - arhy;

        dtx = (x-xp)*isx;
        dty = (y-yp)*isy;

        dis2 = dtx*dtx + dty*dty;

        *res = sqrt(MIN(dis1, dis2));

	   res++;
        
      }
    
    FINALIZE_arcmft:

    #undef ABORT

    Py_XDECREF(Xvals);
    Py_XDECREF(Yvals);
    Py_XDECREF(sXvals);
    Py_XDECREF(sYvals);
    Py_XDECREF(Tvals);
    
    return (PyObject*) out;
}

int quad_surf_mf(float x0, float y0, float z0, float theta, float phi, float psi, float A, float B, float C,
                    int nPts, float *x, float *y, float *z, float *output)
{
    float ctheta, stheta, cphi, sphi, cpsi, spsi;
    float xs, ys, zs, xr,yr,zr;

    int i=0;

    ctheta = cosf(theta);
    stheta = sinf(theta);
    cphi = cosf(phi);
    sphi = sinf(phi);
    cpsi = cosf(psi);
    spsi = sinf(psi);

    for (i=0; i < nPts; i++)
    {
        xs = x[i] - x0;
        ys = y[i] - y0;
        zs = z[i] - z0;

        /*
        xr = ctheta*cpsi*xs + (cphi*spsi + sphi*stheta*cpsi)*ys + (stheta*spsi - cphi*stheta*cpsi)*zs;
        yr = -ctheta*spsi*xs + (cphi*cpsi - sphi*stheta*spsi)*ys + (sphi*cpsi + cphi*stheta*spsi)*zs;
        zr = stheta*xs -sphi*ctheta*ys + cphi*ctheta*zs;*/

        xr = cpsi*ctheta*xs + ctheta*spsi*ys - stheta*zs;
        yr = ctheta*sphi*zs + xs*(-cphi*spsi + cpsi*sphi*stheta) + ys*(cphi*cpsi + sphi*spsi*stheta);
        zr = cphi*ctheta*zs + xs*(cphi*cpsi*stheta + sphi*spsi) + ys*(cphi*spsi*stheta - cpsi*sphi);

        output[i] = zr - (xr*xr*A + yr*yr*B + C);
    }

    return 0;
}


static PyObject * py_quad_surf_mf_pos_fixed(PyObject *self, PyObject *args, PyObject *keywds)
{
    npy_intp size[1];
    int nPts = 0;

    PyObject *oX =0, *oY=0, *oZ=0, *oP =0, *oPos=0;
    PyArrayObject *aX=0, *aY=0, *aZ=0, *aP=0, *aPos=0;
    float *P = 0, *Pos = 0;

    PyArrayObject* out=NULL;

    /*parameters*/
    float x0 = 0, y0 = 0, z0 = 0;
    float theta = 0, phi = 0, psi = 0;
    float A = 0, B = 0, C = 0;
    /*End paramters*/

    static char *kwlist[] = {"p", "X", "Y", "Z", "pos", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOO", kwlist,
         &oP, &oX, &oY, &oZ, &oPos))
        return NULL;

    /* Munge our  input into arrays */
    #define ABORT(msg) {\
        PyErr_Format(PyExc_RuntimeError, msg);\
        goto FINALIZE_py_quad_surf_mf_pos_fixed;\
        }

    aX = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_FLOAT, 0, 1);
    if (aX == NULL) ABORT("Bad X")

    aY = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_FLOAT, 0, 1);
    if (aY == NULL) ABORT("Bad Y")

    aZ = (PyArrayObject *) PyArray_ContiguousFromObject(oZ, PyArray_FLOAT, 0, 1);
    if (aZ == NULL) ABORT("Bad Z")

    aP = (PyArrayObject *) PyArray_ContiguousFromObject(oP, PyArray_FLOAT, 0, 1);
    printf("aP: %d, PyArray_Size(aP): %d\n", (int)aP, PyArray_Size((PyObject*)aP));
    if ((aP == NULL) || (PyArray_Size((PyObject*)aP) < 5)) ABORT("Bad P")

    aPos = (PyArrayObject *) PyArray_ContiguousFromObject(oPos, PyArray_FLOAT, 0, 1);
    if ((aPos == NULL) || (PyArray_Size((PyObject*)aPos) != 3)) ABORT("Bad Pos")

    nPts = PyArray_Size((PyObject*)aX);
    size[0] = nPts;

    //Set parameters
    P = (float*) PyArray_DATA(aP);
    Pos = (float*) PyArray_DATA(aPos);

    x0 = Pos[0];
    y0 = Pos[1];
    z0 = Pos[2];
    theta = P[0];
    phi = P[1];
    psi = P[2];
    A = P[3];
    B = P[4];

    if (PyArray_Size((PyObject*)aP) == 6) C = P[5];

    //done extracting parameters


    out = (PyArrayObject*) PyArray_New(&PyArray_Type, 1,size,NPY_FLOAT, NULL, NULL, 0, 1, NULL);
    if (out == NULL) ABORT("Failed to allocate memory for output")


    printf("calc mf\n");
    printf("nPts: %d, PyArray_Size(oX): %d, , PyArray_Size(oY): %d, , PyArray_Size(oZ): %d, PyArray_Size(out): %d\n", nPts, PyArray_Size((PyObject*)aX), PyArray_Size((PyObject*)aY), PyArray_Size((PyObject*)aZ), PyArray_Size((PyObject*)out));
    quad_surf_mf(x0, y0, z0, theta, phi, psi, A, B, C, nPts, (float *) PyArray_DATA(aX), (float *) PyArray_DATA(aY),
                (float *) PyArray_DATA(aZ), (float *) PyArray_DATA(out));

    printf("done calc mf\n");
    //Py_INCREF(out);

    FINALIZE_py_quad_surf_mf_pos_fixed:

    #undef ABORT

    Py_XDECREF(aX);
    Py_XDECREF(aY);
    Py_XDECREF(aZ);
    Py_XDECREF(aP);
    Py_XDECREF(aPos);

    return (PyObject*) out;
}

static PyObject * py_quad_surf_mf(PyObject *self, PyObject *args, PyObject *keywds)
{
    npy_intp size[1];
    int nPts = 0;

    PyObject *oX =0, *oY=0, *oZ=0, *oP =0;
    PyArrayObject *aX=0, *aY=0, *aZ=0, *aP=0;
    double *P = 0;

    PyArrayObject* out=NULL;

    /*parameters*/
    float x0 = 0, y0 = 0, z0 = 0;
    float theta = 0, phi = 0, psi = 0;
    float A = 0, B = 0, C = 0;
    /*End paramters*/

    static char *kwlist[] = {"p", "X", "Y", "Z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO", kwlist,
         &oP, &oX, &oY, &oZ))
        return NULL;

    /* Munge our  input into arrays */
    #define ABORT(msg) {\
        PyErr_Format(PyExc_RuntimeError, msg);\
        goto FINALIZE_py_quad_surf_mf;\
        }

    aX = (PyArrayObject *) PyArray_ContiguousFromObject(oX, PyArray_FLOAT, 0, 1);
    if (aX == NULL) ABORT("Bad X")

    aY = (PyArrayObject *) PyArray_ContiguousFromObject(oY, PyArray_FLOAT, 0, 1);
    if (aY == NULL) ABORT("Bad Y")

    aZ = (PyArrayObject *) PyArray_ContiguousFromObject(oZ, PyArray_FLOAT, 0, 1);
    if (aZ == NULL) ABORT("Bad Z")

    aP = (PyArrayObject *) PyArray_ContiguousFromObject(oP, PyArray_DOUBLE, 0, 1);
    if (aP == NULL) ABORT("Bad P")
    if (PyArray_Size((PyObject*)aP) != 8) ABORT("P is wrong size")

    nPts = PyArray_Size((PyObject*)aX);
    size[0] = nPts;

    //Set parameters
    P = (double*) PyArray_DATA(aP);

    x0    = (float) P[0];
    y0    = (float) P[1];
    z0    = (float) P[2];
    theta = (float) P[3];
    phi   = (float) P[4];
    psi   = (float) P[5];
    A     = (float) P[6];
    B     = (float) P[7];

    //done extracting parameters


    out = (PyArrayObject*) PyArray_New(&PyArray_Type, 1,size,NPY_FLOAT, NULL, NULL, 0, 1, NULL);
    if (out == NULL) ABORT("Failed to allocate memory for output")

    quad_surf_mf(x0, y0, z0, theta, phi, psi, A, B, C, nPts, (float *) PyArray_DATA(aX), (float *) PyArray_DATA(aY),
                (float *) PyArray_DATA(aZ), (float *) PyArray_DATA(out));


    //Py_INCREF(out);
    FINALIZE_py_quad_surf_mf:

    #undef ABORT

    Py_XDECREF(aX);
    Py_XDECREF(aY);
    Py_XDECREF(aZ);
    Py_XDECREF(aP);

    return (PyObject*) out;
}


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"

static PyMethodDef arcmfMethods[] = {
    {"arcmf",  arcmf, METH_VARARGS | METH_KEYWORDS,
    "Generate a (fast) Gaussian.\n. Arguments are: 'X', 'Y', 'A'=1,'x0'=0, 'y0'=0,sigma=0,b=0,b_x=0,b_y=0"},
    {"arcmft",  arcmft, METH_VARARGS | METH_KEYWORDS,
    "Generate a (fast) Gaussian.\n. Arguments are: 'X', 'Y', 'A'=1,'x0'=0, 'y0'=0,sigma=0,b=0,b_x=0,b_y=0"},
    {"quad_surf_mf_fpos",  py_quad_surf_mf_pos_fixed, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"quad_surf_mf",  py_quad_surf_mf, METH_VARARGS | METH_KEYWORDS,
    ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#pragma GCC diagnostic pop

#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "arcmf",     /* m_name */
        "major refactoring of the Analysis tree",  /* m_doc */
        -1,                  /* m_size */
        arcmfMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_arcmf(void)
{
	PyObject *m;
    // m = PyModule_Create("edgeDB", edgeDBMethods);
    m = PyModule_Create(&moduledef);
    import_array()
    return m;
}

#else
PyMODINIT_FUNC initarcmf(void)
{
    PyObject *m;

    m = Py_InitModule("arcmf", arcmfMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
#endif
