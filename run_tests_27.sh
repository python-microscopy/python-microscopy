#!/usr/bin/env bash

#magic to make sure we can access anaconda
if [ -e ~/anaconda/bin/activate ]
then
    source ~/anaconda/bin/activate
else
    if [ -e ~/anaconda2/bin/activate ]
    then
        source ~/anaconda2/bin/activate
    else
        source ~/anaconda3/bin/activate
    fi
fi

exit_code=0

if [[ `conda env list` = *"pm_test_27"* ]]
    then
        echo "Using existing pm_test_27 environment"
    else
        conda create -n pm_test_27 -y -q python=2.7 pyme-depends pytest pytest-cov pytest-html nose cython
fi

if [ $? -eq 0 ]
then
    oldenv=$CONDA_DEFAULT_ENV
    #conda deactivate
    conda activate pm_test_27
    if [ $? -eq 0 ]
    then
        #pwd
        python setup.py develop
        if [ $? -eq 0 ]
        then
            cd tests
            pytest -v --html=tests_py27.html --cov=../PYME --cov-report html:cov_html .
            exit_code=$?
            cd ..
        else
            exit_code=$?
        fi
        #conda deactivate
        conda activate $oldenv
    else
        exit_code=$?
    fi
    #conda env remove --name pm_test_27 -y -q
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


