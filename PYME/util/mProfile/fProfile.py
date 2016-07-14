#!/usr/bin/python

###############
# mProfile.py
#
# Copyright David Baddeley, 2012
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
################
#!/usr/bin/python
""" mProfile.py - Matlab(TM) style line based profiling for Python

Copyright: David Baddeley 2008
	   david_baddeley <at> yahoo.com.au

useage is similar to profiling in matlab (profile on, profile off, report),
with the major difference being that you have to specify the filenames in which
you want to do the profiling (to improve performance & to save wading through
lots of standard library code etc ...).

e.g.

mProfile.profileOn(['onefile.py', 'anotherfile.py'])

stuff to be profiled ....

mProfile.profileOff()
mProfile.report()

Due to the fact that we're doing this in python, and hooking every line, 
there is a substantial performance hit, although for the numeric code I wrote it
for (lots of vectorised numpy/scipy stuff) it's only on the order of ~30%.

Licensing: Take your pick of BSD or GPL
"""

import sys
import time
import os
import colorize_db_t
import webbrowser
import tempfile
import threading

import re


class thread_profiler(object):
    def __init__(self):
        self.outfile = None

    def profileOn(self, regex='.*PYME.*', outfile='profile.txt'):
        self.regex = re.compile(regex)
        self.outfile = open(outfile, 'w')

        sys.setprofile(self.prof_callback)
        threading.setprofile(self.prof_callback)



    def profileOff(self):
        sys.setprofile(None)
        threading.setprofile(None)

        self.outfile.flush()
        self.outfile.close()

    def prof_callback(self, frame, event, arg):
        if not frame.f_code.co_filename == __file__ and self.regex.match(frame.f_code.co_filename):
            fn = frame.f_code.co_filename #.split(os.sep)[-1]
            funcName = fn + '\t' + frame.f_code.co_name

            t = time.clock()

            self.outfile.write('%f\t%s\t%s\t%s\n' % (t, threading.current_thread().getName(), funcName, event))



    def report(self):
        tpath = os.path.join(tempfile.gettempdir(), 'mProf')
        if not os.path.exists(tpath):
            os.makedirs(tpath)

        for f in filenames:
            tfn = os.path.join(tpath,  f + '.html')
            colorize_db_t.colorize_file(files[f], linecounts[f], fullfilenames[f],open(tfn, 'w'))
            webbrowser.open('file://' + tfn, 2)
        
