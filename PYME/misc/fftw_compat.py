"""
Compatibility shim to translate between pyfftw3 and pyfftw calls
"""

try:
    from fftw3f import Plan, create_aligned_array
except ImportError:
    import pyfftw
    from pyfftw import empty_aligned as create_aligned_array
    
    class Plan(pyfftw.FFTW):
        def __init__(self, A, B, direction, flags = [], nthreads=1):
            
            if direction == 'forward':
                dir = 'FFTW_FORWARD'
            else:
                dir = 'FFTW_BACKWARD'
                
            out_flags = []
            for f in flags:
                if f == 'measure':
                    out_flags.append('FFTW_MEASURE')

            pyfftw.FFTW.__init__(self, A, B, direction=dir, axes=range(A.ndim), threads=nthreads,  flags = out_flags)
            
        def __call__(self):
            self.execute()
            