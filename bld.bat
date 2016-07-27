:: "%PYTHON%" setup.py build_ext -c mingw32
"%PYTHON%" setup.py install

copy "%RECIPE_DIR%"\libpython27.a "%PREFIX%"\libs

"%PYTHON%" pymecompress\setup.py build --compiler=mingw32
"%PYTHON%" pymecompress\setup.py install 

if errorlevel 1 exit 1

:: Add more build steps here, if they are necessary.

:: See
:: http://docs.continuum.io/conda/build.html
:: for a list of environment variables that are set during the build process.
