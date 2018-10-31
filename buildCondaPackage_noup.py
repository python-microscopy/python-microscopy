# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:39:19 2015

@author: david
"""

import os
import sys
import subprocess
import json

sys.path.append('.')

from PYME import update_version
update_version.update_version()

from PYME import version

os.environ['PYME_VERSION'] = version.version

def get_package_versions(packagename):
    s = subprocess.check_output(['conda', 'search', '--json',packagename])
    
    packages = json.loads(s)[packagename]
    
    return packages

#find out if there are any previous builds for this version
#import conda.api
prev_builds = [p['build_number'] for p in get_package_versions('python-microscopy') if p['version'] == version.version]

print('%d previous builds of this version' % len(prev_builds))

if len(prev_builds) > 0:
    os.environ['BUILD_NUM'] = str(max(prev_builds) +1)

os.system('conda build --no-anaconda-upload conda-recipes/python-microscopy')