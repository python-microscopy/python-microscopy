#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:26:21 2011

@author: David
"""

from datetime import datetime
import os
import subprocess

def hook(ui, repo, **kwargs):
    update_version()
    return 0

def update_version():
    now = datetime.now()
    
    p = subprocess.Popen('hg id -i', shell=True, stdout = subprocess.PIPE)
    id = p.stdout.readline().strip()
    
    f = open(os.path.join(os.path.split(__file__)[0], 'version.py'), 'w')
    
    f.write('#PYME uses date based versions (yy.m.d)\n')    
    f.write("version = '%d.%d.%d'\n\n" % (now.year - 2000, now.month, now.day))
    f.write('#Mercurial changeset id\n')
    f.write("changeset = '%s'\n" % id)
    f.close()
    
if __name__ == '__main__':
    update_version()