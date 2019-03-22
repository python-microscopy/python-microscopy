@echo on
set oldenv=%CONDA_DEFAULT_ENV%
echo %oldenv%

call conda activate pm_test_27

python setup.py develop

cd tests
pytest -v --html=tests_py27.html --cov=..\PYME --cov-report html:cov_html .
cd ..

conda activate %oldenv%




