:: "%PYTHON%" setup.py build_ext -c mingw32
"%PYTHON%" setup.py install

"%PYTHON%" pymecompress\setup.py install --compiler=mingw32

if errorlevel 1 exit 1

:: Add more build steps here, if they are necessary.

:: See
:: http://docs.continuum.io/conda/build.html
:: for a list of environment variables that are set during the build process.
