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


def addFiles(directory, extensions=['.h5r', '.h5']):
    #dir_size = 0
    for (path, dirs, files) in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] in extensions:
                filename = os.path.join(path, file)
                #print filename
                try:
                    f = File.GetOrCreate(filename)

                    if f.filesize <=1: #was added before database updated
                        f.filesize = os.path.getsize(filename)
                        f.save()
                except ValueError as e:
                    print e


if __name__ == '__main__':
    addFiles(sys.argv[1])
    