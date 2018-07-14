#!/usr/bin/env bash

cd tests

exit_code=0

conda create -n pm_test_27 -y -q python=2.7 pyme-depends pytest pytest-cov pytest-html nose
if [ $? -eq 0 ]
then
    source activate pm_test_27
    if [ $? -eq 0 ]
    then
        python setup.py develop
        if [ $? -eq 0 ]
        then
            pytest -v --html=tests_py27.html --cov=../PYME --cov-report html:cov_html .
        else
            exit_code=$?
        fi
        source deactivate
    else
        exit_code=$?
    fi
    conda env remove --name pm_test_27 -y -q
else
   exit $?
fi

if [ $exit_code -ne 0 ]
then
    exit $exit_code
fi

#conda create -n pm_test_36 -y python=3.6 pyme-depends pytest pytest-cov pytest-html nose
#source activate pm_test_36
#python setup.py develop
#pytest -v --html=tests_py36.html .
#source deactivate
#conda env remove --name pm_test_36 -y


conda create -n pm_test_36 -y -q python=3.6 pyme-depends pytest pytest-cov pytest-html nose
if [ $? -eq 0 ]
then
    source activate pm_test_36
    if [ $? -eq 0 ]
    then
        python setup.py develop
        if [ $? -eq 0 ]
        then
            pytest -v --html=tests_py36.html  .
        else
            exit_code=$?
        fi
        source deactivate
    else
        exit_code=$?
    fi
    conda env remove --name pm_test_36 -y -q
else
   exit $?
fi

if [ $exit_code -ne 0 ]
then
    exit $exit_code
fi
