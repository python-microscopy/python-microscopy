#!/usr/bin/python
##################
# install_dependencies.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################
"""install_dependencies trys to pull binary packages from the distro, if present
and of a suitable version, falling back on pip for any packages not found."""

import apt #hope this is there
import sys
import os

#version numbers here are minumum versions
depends = {'python' : '2.6',
'python-imaging' : '1.1.6',
'python-opengl' : '3.0.1',
'pyro' : '3.9.1',
'python-dev' : '',
'python-matplotlib' : '0.98',
'python-numpy' : '1.2.1',
'python-scipy' : '0.7.0',
'python-tables' :  '2.1.1',
'build-essential' : '',
'python-wxgtk2.8' :  '2.8.11.0', 
'python-pip' : ''}

recommends = {
'python-sympy' :  '0.6.3',
'python-serial' : '',
'ipython' : '0.10',
'python-sphinx' : '0.6.5',
'python-django' : '1.2.1',
'python-django-south' :  '0.7.1',
'mayavi2' : '3.3',
'python-mysqldb' :  '1.2.2'}

installRecommends = False

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'recommends':
        installRecommends = True
        
        #add recommended packages to dependencies
        depends.update(recommends)

    cache = apt.Cache()

    for packageName, version in depends.items():
        try:
            package = cache[packageName]

            if not (package.isInstalled and package.installedVersion >= version):
                if package.candidateVersion >= version:
                    if package.isInstalled:
                        package.markUpgrade()
                    else:
                        package.markInstall()
        except KeyError: #if the distro doesn't have the package
            pass

    #install the modules available as packages
    cache.commit()


    #run pip to mop up everything else
    #these have everything above plus a couple of additional dependencies which 
    #I'm not going to even try getting from the repositories as I know they're not there
    #recommended-modules.txt includes required-modules.txt
    if installRecommends:
        os.system('pip install -r recommended-modules.txt')
    else:
        os.system('pip install -r required-modules.txt')







