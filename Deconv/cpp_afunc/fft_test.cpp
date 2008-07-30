#include <Python.h>
#include <complex>
#include <math.h>
#include <fftw3.h>
#include "Numeric/arrayobject.h"


int main(int argc, char **argv)
{
    fftw_complex *in, *out;
    fftw_plan p;
    int N = 64;
    
    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N*N*128);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N*N*128);
    p = fftw_plan_dft_3d(N,N,128, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    fftw_execute(p); /* repeat as needed */
    
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
}

static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return Py_BuildValue("i", sts);
}

PyMODINIT_FUNC
initspam(void)
{
    PyObject *m;

    m = Py_InitModule("spam", SpamMethods);

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_INCREF(SpamError);
    PyModule_AddObject(m, "error", SpamError);
}
