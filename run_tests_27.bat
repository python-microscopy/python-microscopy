
set oldenv=%CONDA_DEFAULT_ENV%
echo %oldenv%

call conda activate pm_test_27

if errorlevel 1 exit 1

"%PYTHON%" setup.py develop

if errorlevel 1 exit 1

cd tests
pytest -v --html=tests_py27.html --cov=..\PYME --cov-report html:cov_html .
cd ..

if errorlevel 1 exit 1

conda activate %oldenv%




