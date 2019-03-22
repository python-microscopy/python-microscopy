
set oldenv=%CONDA_DEFAULT_ENV%
echo %oldenv%

conda activate pm_test_27

python setup.py develop

cd tests
pytest -v --html=tests_py27.html --cov=..\PYME --cov-report html:cov_html .

conda activate %oldenv%




