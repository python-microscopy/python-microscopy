#!/usr/bin/env bash

exit_code=0

bash ./run_tests_27.sh
if [ $? -eq 0 ]
then
    exit $?
fi

#bash ./run_tests_36.sh
#if [ $? -eq 0 ]
#then
#    exit $?
#fi