xcopy %PREFIX%\DLLs\libfftw3* %SRC_DIR%\pyfftw\ /f

"%PYTHON%" setup.py install