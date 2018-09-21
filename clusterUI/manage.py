#!/usr/bin/env python
import os
import sys
from PYME.util import fProfile, mProfile

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "clusterUI.settings")

    from django.core.management import execute_from_command_line

    #prof = fProfile.thread_profiler()
    #prof.profileOn('.*clusterUI.*', '/home/ubuntu/clusterUI_prof.txt')
    #mProfile.profileOn(['clusterIO.py',])
    execute_from_command_line(sys.argv)
    #mProfile.profileOff()
    #mProfile.report(False, '/home/ubuntu/')
    #prof.profileOff()
