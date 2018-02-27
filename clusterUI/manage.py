#!/usr/bin/env python
import os
import sys
from PYME.util import fProfile

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "clusterUI.settings")

    from django.core.management import execute_from_command_line

    prof = fProfile.thread_profiler()
    prof.profileOn('.*clusterUI.*', '/home/ubuntu/clusterUI_prof.txt')
    execute_from_command_line(sys.argv)
    prof.profileOff()
