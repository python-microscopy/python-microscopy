import os
#import PYME

os.system('pytest -v --html=test_report.html --cov=PYME --cov-report html:cov_html ../../tests')