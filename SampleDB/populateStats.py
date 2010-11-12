#!/usr/bin/python
import sys
import os

#put the folder above this one on the python path
#print __file__
#print os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])

#let Django know where to find the settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'SampleDB.settings'

from SampleDB.samples.models import File
from SampleDB.samples import calcEventStats


def addStats():
    #dir_size = 0
    files = File.objects.all()[8100:8200]
    for f in files:
        calcEventStats.getStats(f)


if __name__ == '__main__':
    addStats()
    