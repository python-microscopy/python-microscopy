#include "Python.h"
//#include <complex.h>
#define _USE_MATH_DEFINES
#include <math.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <stdio.h>

#define MIN(a, b) ((a<b) ? a : b) 
#define MAX(a, b) ((a>b) ? a : b)

/////////////////////////
// See Gibson & Lanni 1991

double bessj0(double x)
{
    /*Returns the Bessel function J0(x) for any real x.*/

    double ax,z;
    double xx,y,ans,ans1,ans2; /*Accumulate polynomials in double precision.*/
    
    ax=fabs(x);
    
    if (ax < 8.0) /* Direct rational function fit.*/
    {
        y=x*x;
        ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7 +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));

        ans2=57568490411.0+y*(1029532985.0+y*(9494680.718  +y*(59272.64853+y*(267.8532712+y*1.0))));

        ans=ans1/ans2;
    } else /* Fitting function (6.5.9).*/
    {
        z=8.0/ax;
        y=z*z;
        xx=ax-0.785398164;

        ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4 +y*(-0.2073370639e-5+y*0.2093887211e-6)));

        ans2 = -0.1562499995e-1+y*(0.1430488765e-3 +y*(-0.6911147651e-5+y*(0.7621095161e-6  -y*0.934945152e-7)));

        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
    }
    
    return ans;
}

static PyObject * genWidefieldPSF(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *res = 0;
    int ix,iy,iz, ip;
    npy_intp size[3];

    int size_p;

    PyObject *oX =0;
    PyObject *oY=0;
    PyObject *oZ=0;
    PyObject *oP= 0;

    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    PyArrayObject* Zvals;
    PyArrayObject* Pvals;
    PyArrayObject* out;

    double *pXvals;
    double *pYvals;
    double *pZvals;
    double *pPvals;

    double p2, spa2, r, opd;
    double ni2, NA2, ns2, ng2, nis2, ngs2;


    /*parameters*/
    double A = 1;
    double x0 = 0;
    double y0 = 0;
    double z0 = 0;

    double k = 2*M_PI/488;
    double NA = 1.3;

    double zp = 20e3;
    double ni = 1.5;
    double ns = 1.47;
    double ng = 1.5;
    double nis = 1.5;
    double ngs = 1.5;
    double tg = 175e3;
    double tgs = 175e3;
    double tis = 90e3;
    /*End paramters*/

    double ps_a_r;
    double ps_a_i;

    double *opd_facs_r;
    double *opd_facs_i;
    double *bessel_lu;

    //printf("check0\n");

    static char *kwlist[] = {"X", "Y", "Z", "P", "A","x0", "y0", "z0", "k", "NA", "depthInSample", "nImmersionCorr", "nSample","nCoverslipCorr", "nImmersionSample", "nCoverslipSample", "CoverslipThicknessCorr", "CoverslipThicknessSample", "ImmersionThicknessSample", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|ddddddddddddddd", kwlist,
         &oX, &oY, &oZ, &oP, &A, &x0, &y0, &z0, &k, &NA, &zp, &ni, &ns, &nis, &ngs, &tg, &tgs, &tis))
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

    Zvals = (PyArrayObject *) PyArray_ContiguousFromObject(oZ, PyArray_DOUBLE, 0, 1);
    if (Zvals == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
	PyErr_Format(PyExc_RuntimeError, "Bad Z");
        return NULL;
    }

    Pvals = (PyArrayObject *) PyArray_ContiguousFromObject(oP, PyArray_DOUBLE, 1, 1);
    if (Pvals == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
        Py_DECREF(Zvals);
	PyErr_Format(PyExc_RuntimeError, "Bad Z");
        return NULL;
    }

    //printf("check1\n");

    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    pZvals = (double*)Zvals->data;
    pPvals = (double*)Pvals->data;



    size[0] = PyArray_Size((PyObject*)Xvals);
    size[1] = PyArray_Size((PyObject*)Yvals);
    size[2] = PyArray_Size((PyObject*)Zvals);

    size_p = PyArray_Size((PyObject*)Pvals);

    //out = (PyArrayObject*) PyArray_FromDims(3,size,PyArray_DOUBLE);
    out = (PyArrayObject*) PyArray_New(&PyArray_Type, 3,size,NPY_DOUBLE, NULL, NULL, 0, 1, NULL);
    if (out == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
        Py_DECREF(Zvals);
        Py_DECREF(Pvals);
        PyErr_Format(PyExc_RuntimeError, "Output array not allocated");
        return NULL;
    }
    //printf("check2\n");
    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];
    //out->strides[2] = sizeof(double)*size[0]*size[1];

    res = (double*) PyArray_DATA(out);
    //printf("check3\n");

    ni2 = ni*ni;
    NA2 = NA*NA;
    ns2 = ns*ns;
    ng2 = ng*ng;
    nis2 = nis*nis;
    ngs2 = ngs*ngs;


    /*temp arrays*/

    opd_facs_r = PyMem_Malloc(size_p*sizeof(double));
    opd_facs_i = PyMem_Malloc(size_p*sizeof(double));
    bessel_lu = PyMem_Malloc(size[0]*size[1]*size_p*sizeof(double));

    //printf("check4\n");

    for (iy = 0; iy < size[1]; iy++)
    {

        for (ix = 0; ix < size[0]; ix++)
        {
            r = sqrt(((pXvals[ix] - x0) * (pXvals[ix] - x0)) + ((pYvals[iy]-y0) * (pYvals[iy]-y0)));
            for (ip = 0; ip < size_p; ip++)
            {

                bessel_lu[ip + size_p*ix + size_p*size[0]*iy] = bessj0( k*r*NA*pPvals[ip])*pPvals[ip];
            }

        }

    }

    //printf("check5\n");

    for (iz = 0; iz < size[2]; iz++)
    {
        for (ip = 0; ip < ( size_p); ip++)
        {
            p2 = pPvals[ip] * pPvals[ip];
            spa2 = sqrt(ni2 - (NA2)*(p2));


            opd = (zp - (pZvals[iz] - z0 + zp))*spa2 + zp*(sqrt(ns2 - (NA2)*(p2)) - ni*spa2/ns)
                + tg*(sqrt(ng2 - (NA2)*(p2)) - ni*spa2/ng)  - tgs*(sqrt(ngs2 - (NA2)*(p2)) - ni*spa2/ngs)
                - tis*(sqrt(nis2 - (NA2)*(p2)) - ni*spa2/nis);



            opd_facs_r[ip] = (.01*cos(k*opd));
            opd_facs_i[ip] = (.01*sin(k*opd));
        }

        for (iy = 0; iy < size[1]; iy++)
        {

            for (ix = 0; ix < size[0]; ix++)
            {

                ps_a_r = 0;
                ps_a_i = 0;

                for (ip = 0; ip < size_p; ip++)
                {


                    ps_a_r += opd_facs_r[ip]*bessel_lu[ip + size_p*ix + size_p*size[0]*iy];
                    ps_a_i += opd_facs_i[ip]*bessel_lu[ip + size_p*ix + size_p*size[0]*iy];
                }


                *res = A*(ps_a_r*ps_a_r + ps_a_i*ps_a_i);

                res++;
            }
        }
    }

    //printf("check6\n");

    PyMem_Free(opd_facs_r);
    PyMem_Free(opd_facs_i);
    PyMem_Free(bessel_lu);

    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    Py_DECREF(Zvals);
    Py_DECREF(Pvals);

    //printf("check7\n");

    return (PyObject*) out;
}


/*Weighted version of Gibson-Lanni to take into account apodization and/or near field effects*/
static PyObject * genWidefieldPSFW(PyObject *self, PyObject *args, PyObject *keywds)
{
    double *res = 0;
    int ix,iy,iz, ip;
    npy_intp size[3];

    int size_p;

    PyObject *oX =0;
    PyObject *oY=0;
    PyObject *oZ=0;
    PyObject *oP= 0;
    PyObject *oW = 0;

    PyArrayObject* Xvals;
    PyArrayObject* Yvals;
    PyArrayObject* Zvals;
    PyArrayObject* Pvals;
    PyArrayObject* Wvals;
    PyArrayObject* out;

    double *pXvals;
    double *pYvals;
    double *pZvals;
    double *pPvals;
    double *pWvals;

    double p2, spa2, r, opd;
    double ni2, NA2, ns2, ng2, nis2, ngs2;


    /*parameters*/
    double A = 1;
    double x0 = 0;
    double y0 = 0;
    double z0 = 0;

    double k = 2*M_PI/488;
    double NA = 1.3;

    double zp = 20e3;
    double ni = 1.5;
    double ns = 1.47;
    double ng = 1.5;
    double nis = 1.5;
    double ngs = 1.5;
    double tg = 175e3;
    double tgs = 175e3;
    double tis = 90e3;
    /*End paramters*/

    double ps_a_r;
    double ps_a_i;

    double *opd_facs_r;
    double *opd_facs_i;
    double *bessel_lu;

    //printf("check0\n");

    static char *kwlist[] = {"X", "Y", "Z", "P", "W", "A","x0", "y0", "z0", "k", "NA", "depthInSample", "nImmersionCorr", "nSample","nCoverslipCorr", "nImmersionSample", "nCoverslipSample", "CoverslipThicknessCorr", "CoverslipThicknessSample", "ImmersionThicknessSample", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOO|ddddddddddddddd", kwlist,
         &oX, &oY, &oZ, &oP, &oW, &A, &x0, &y0, &z0, &k, &NA, &zp, &ni, &ns, &nis, &ngs, &tg, &tgs, &tis))
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

    Zvals = (PyArrayObject *) PyArray_ContiguousFromObject(oZ, PyArray_DOUBLE, 0, 1);
    if (Zvals == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
	PyErr_Format(PyExc_RuntimeError, "Bad Z");
        return NULL;
    }

    Pvals = (PyArrayObject *) PyArray_ContiguousFromObject(oP, PyArray_DOUBLE, 1, 1);
    if (Pvals == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
        Py_DECREF(Zvals);
	PyErr_Format(PyExc_RuntimeError, "Bad Z");
        return NULL;
    }

    Wvals = (PyArrayObject *) PyArray_ContiguousFromObject(oW, PyArray_DOUBLE, 1, 1);
    if (Wvals == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
        Py_DECREF(Zvals);
        Py_DECREF(Pvals);
	PyErr_Format(PyExc_RuntimeError, "Bad Z");
        return NULL;
    }

    //printf("check1\n");

    pXvals = (double*)Xvals->data;
    pYvals = (double*)Yvals->data;
    pZvals = (double*)Zvals->data;
    pPvals = (double*)Pvals->data;
    pWvals = (double*)Wvals->data;


    size[0] = PyArray_Size((PyObject*)Xvals);
    size[1] = PyArray_Size((PyObject*)Yvals);
    size[2] = PyArray_Size((PyObject*)Zvals);

    size_p = PyArray_Size((PyObject*)Pvals);

    //out = (PyArrayObject*) PyArray_FromDims(3,size,PyArray_DOUBLE);
    out = (PyArrayObject*) PyArray_New(&PyArray_Type, 3,size,NPY_DOUBLE, NULL, NULL, 0, 1, NULL);
    if (out == NULL)
    {
        Py_DECREF(Xvals);
        Py_DECREF(Yvals);
        Py_DECREF(Zvals);
        Py_DECREF(Pvals);
        Py_DECREF(Wvals);
        PyErr_Format(PyExc_RuntimeError, "Output array not allocated");
        return NULL;
    }
    //printf("check2\n");
    //fix strides
    //out->strides[0] = sizeof(double);
    //out->strides[1] = sizeof(double)*size[0];
    //out->strides[2] = sizeof(double)*size[0]*size[1];

    res = (double*) PyArray_DATA(out);
    //printf("check3\n");

    ni2 = ni*ni;
    NA2 = NA*NA;
    ns2 = ns*ns;
    ng2 = ng*ng;
    nis2 = nis*nis;
    ngs2 = ngs*ngs;


    /*temp arrays*/

    opd_facs_r = PyMem_Malloc(size_p*sizeof(double));
    opd_facs_i = PyMem_Malloc(size_p*sizeof(double));
    bessel_lu = PyMem_Malloc(size[0]*size[1]*size_p*sizeof(double));

    //printf("check4\n");

    for (iy = 0; iy < size[1]; iy++)
    {

        for (ix = 0; ix < size[0]; ix++)
        {
            r = sqrt(((pXvals[ix] - x0) * (pXvals[ix] - x0)) + ((pYvals[iy]-y0) * (pYvals[iy]-y0)));
            for (ip = 0; ip < size_p; ip++)
            {

                bessel_lu[ip + size_p*ix + size_p*size[0]*iy] = bessj0( k*r*NA*pPvals[ip])*pPvals[ip];
            }

        }

    }

    //printf("check5\n");

    for (iz = 0; iz < size[2]; iz++)
    {
        for (ip = 0; ip < ( size_p); ip++)
        {
            p2 = pPvals[ip] * pPvals[ip];
            spa2 = sqrt(ni2 - (NA2)*(p2));


            opd = (zp - (pZvals[iz] - z0 + zp))*spa2 + zp*(sqrt(ns2 - (NA2)*(p2)) - ni*spa2/ns)
                + tg*(sqrt(ng2 - (NA2)*(p2)) - ni*spa2/ng)  - tgs*(sqrt(ngs2 - (NA2)*(p2)) - ni*spa2/ngs)
                - tis*(sqrt(nis2 - (NA2)*(p2)) - ni*spa2/nis);



            opd_facs_r[ip] = (.01*cos(k*opd))*pWvals[ip];
            opd_facs_i[ip] = (.01*sin(k*opd))*pWvals[ip];
        }

        for (iy = 0; iy < size[1]; iy++)
        {

            for (ix = 0; ix < size[0]; ix++)
            {

                ps_a_r = 0;
                ps_a_i = 0;

                for (ip = 0; ip < size_p; ip++)
                {


                    ps_a_r += opd_facs_r[ip]*bessel_lu[ip + size_p*ix + size_p*size[0]*iy];
                    ps_a_i += opd_facs_i[ip]*bessel_lu[ip + size_p*ix + size_p*size[0]*iy];
                }


                *res = A*(ps_a_r*ps_a_r + ps_a_i*ps_a_i);

                res++;
            }
        }
    }

    //printf("check6\n");

    PyMem_Free(opd_facs_r);
    PyMem_Free(opd_facs_i);
    PyMem_Free(bessel_lu);

    Py_DECREF(Xvals);
    Py_DECREF(Yvals);
    Py_DECREF(Zvals);
    Py_DECREF(Pvals);
    Py_DECREF(Wvals);

    //printf("check7\n");

    return (PyObject*) out;
}

static PyMethodDef ps_appMethods[] = {
    {"genWidefieldPSF",  (PyCFunction)genWidefieldPSF, METH_VARARGS | METH_KEYWORDS,
    "Generate a (shifted, spherically abberated) widefield PSF based on the paraxial approximation. Arguments are: 'X', 'Y', 'Z', 'P' , 'A'=1,'x0'=0, 'y0'=0, 'z0'=0, 'k'=2*pi/488, 'NA'=1.3, 'depthInSample'=20um, 'nImmersionCorr'=1.5, 'nSample'=1.47,'nCoverslipCorr'=1.5, 'nImmersionSample'=1.5, 'nCoverslipSample'=1.5, 'CoverslipThicknessCorr'=175um, 'CoverslipThicknessSample'=175um, 'ImmersionThicknessSample'=90um. All values should be given in nm."},
    {"genWidefieldPSFW",  (PyCFunction)genWidefieldPSFW, METH_VARARGS | METH_KEYWORDS,
    "Generate a (shifted, spherically abberated) widefield PSF based on the paraxial approximation. Arguments are: 'X', 'Y', 'Z', 'P', 'W' , 'A'=1,'x0'=0, 'y0'=0, 'z0'=0, 'k'=2*pi/488, 'NA'=1.3, 'depthInSample'=20um, 'nImmersionCorr'=1.5, 'nSample'=1.47,'nCoverslipCorr'=1.5, 'nImmersionSample'=1.5, 'nCoverslipSample'=1.5, 'CoverslipThicknessCorr'=175um, 'CoverslipThicknessSample'=175um, 'ImmersionThicknessSample'=90um. All values should be given in nm."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ps_app",     /* m_name */
        "Gibson & Lanni PSF approximation",  /* m_doc */
        -1,                  /* m_size */
        ps_appMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_ps_app(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else

PyMODINIT_FUNC initps_app(void)
{
    PyObject *m;

    m = Py_InitModule("ps_app", ps_appMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}

#endif
