//#include <Python.h>
//#include <complex.h>
#include <math.h>
#include <fftw3.h>
//#import  <blas.h>
//#include "Numeric/arrayobject.h"


int main(int argc, char **argv)
{
    fftwf_complex *in, *out;
    fftwf_plan p;
    int N = 64;
    
    in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N*N*256);
    out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N*N*256);
    p = fftwf_plan_dft_3d(N,N,256, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    fftwf_execute(p); /* repeat as needed */
    
    fftwf_destroy_plan(p);
    fftwf_free(in); fftwf_free(out);
}
