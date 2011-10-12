# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:46:42 2011

@author: David
"""

import os
import glob
import shutil
from distutils.sysconfig import get_python_lib


def main():
    wxFiles = glob.glob(os.path.join(get_python_lib(), 'wx*'))
    
    for f in wxFiles:
        if os.path.isdir(f):
            #recursively delete directory
            shutil.rmtree(f)
        else:
            os.remove(f)
            
if __name__ == '__main__':
    main()