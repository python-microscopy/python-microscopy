"""
Allow creation of aligned numpy arrays. A wrapper around fftw3f.create_aligned_array on python2,
around pyfftw.empty_aligned on python 3.

"""

try:
    from fftw3f import create_aligned_array
except ImportError:
    from pyfftw import empty_aligned as create_aligned_array