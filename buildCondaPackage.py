# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:39:19 2015

@author: david
"""

import os
import sys

sys.path.append('.')

from PYME import update_version
update_version.update_version()

from PYME import version

os.environ['PYME_VERSION'] = version.version

#find out if there are any previous builds for this version
import conda.api
prev_builds = [p.build_number for p in conda.api.get_package_versions('python-microscopy') if p.version == version.version]

if len(prev_builds) > 0:
    os.environ['BUILD_NUM'] = str(max(prev_builds) +1)

os.system('conda build . --numpy 1.9')