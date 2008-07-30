from distutils.core import setup, Extension

module1 = Extension('cDec',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['/usr/local/include'],
                    libraries = ['fftw3f'],
                    library_dirs = ['/usr/local/lib'],
                    sources = ['fft_test.c'])

setup (name = 'cDec',
       version = '1.0',
       description = 'c routines for deconvolution package',
       author = 'David Baddeley',
       author_email = 'baddeley@kip.uni-heidelberg.de',
       url = '',
       long_description = '''
optimised routines for processor intensive parts of deconvolution
''',
       ext_modules = [module1])