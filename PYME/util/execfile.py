"""
Patch around different python versions of exec and execfile
"""
import sys
import six

def _exec(codeObj, localVars = None, globalVars = None):
    return six.exec_(codeObj,localVars,globalVars)


if sys.version_info.major == 2:
    # def _exec(codeObj, localVars = None, globalVars = None):
    #     exec codeObj in localVars,globalVars
    def _execfile(filename, localVars=None, globalVars=None):
        # noinspection PyCompatibility
        execfile(filename, localVars, globalVars)
else: #Python 3
    #def _exec(codeObj, localVars = None, globalVars = None):
    #    exec(codeObj,localVars,globalVars)
    def _execfile(filename, localVars=None, globalVars=None):
        exec(compile(open(filename).read(), filename, 'exec'), localVars, globalVars)