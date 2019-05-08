"""
Compatibility shim to translate between pyfftw3 and pyfftw calls
"""

try:
    from fftw3f import Plan, create_aligned_array
except ImportError:
    import pyfftw
    from pyfftw import empty_aligned as create_aligned_array
    
    class Plan(object):
        def __init__(self, A, B, direction, flags = [], nthreads=1):
            
            if direction == 'forward':
                dir = 'FFTW_FORWARD'
            else:
                dir = 'FFTW_BACKWARD'
                
            out_flags = []
            for f in flags:
                if f == 'measure':
                    out_flags.append('FFTW_MEASURE')

            self._plan = pyfftw.FFTW(A, B, direction=dir, axes=[int(i) for i in range(A.ndim)], threads=int(nthreads),  flags = out_flags)
            
        def __call__(self):
            self._plan.execute()
            