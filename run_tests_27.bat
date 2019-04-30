@echo on
set oldenv=%CONDA_DEFAULT_ENV%
echo %oldenv%

call conda activate pm_test_27

python setup.py develop
if errorlevel 1 exit /B 1


cd tests
pytest -v --html=tests_py27.html --cov=..\PYME --cov-report html:cov_html .
if errorlevel 1 exit /B 1

cd ..

call conda activate %oldenv%




