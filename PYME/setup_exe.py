#!/usr/bin/python

##################
# setup.py
#
# Copyright David Baddeley, 2009
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


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('PYME',parent_package,top_path)
    config.add_subpackage('Analysis')
    config.add_subpackage('Acquire')
    config.add_subpackage('DSView')
    config.add_subpackage('PSFGen')
    config.add_subpackage('cSMI')
    config.add_subpackage('ParallelTasks')
    config.add_subpackage('io')
    config.add_subpackage('misc')
    config.add_subpackage('pad')
    config.add_subpackage('dataBrowser')
    
    #config.make_svn_version_py()  # installs __svn_version__.py
    #config.make_config_py()
    return config

if __name__ == '__main__':
    #from numpy.distutils.core import setup
    #setup(**configuration(top_path='').todict())

    from cx_Freeze import setup, Executable
    import matplotlib
    setup(executables=[Executable('LMVis/VisGUI.py'),Executable('Acquire/PYMEAquire.py'),Executable('DSView/dh5view.py')],
        options= {'build_exe' : {
          'excludes' : ['pyreadline', 'Tkconstants', 'Tkinter', 'tcl', '_imagingtk', 'PIL._imagingtk', 'ImageTK', 'PIL.ImageTK', 'FixTk'],
          'packages' : ['OpenGL', 'OpenGL.platform', 'OpenGL.arrays']}},

        #data_files=matplotlib.get_py2exe_datafiles(),
      #cmdclass = {'build_ext': build_ext},
      #ext_modules = ext_modules
      )
