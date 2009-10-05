#!/usr/bin/python

##################
# startCommentify.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import os

copyr = '''#!/usr/bin/python

##################
# %s
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

'''

def walkFunc(arg, dirname, fnames):
    for fn in fnames:
        if fn.endswith('.py'):
          f = open(os.path.join(dirname, fn), 'r')

          text = f.read()

          f.close()

          #f = open()

          print '------------------------------------------------'
          print fn
          print '------------------------------------------------\n'

          print text[:500]

          res = raw_input('Add header to file? y/n:')

          if res.upper() == 'Y':
              f = open(os.path.join(dirname, fn), 'w')
              f.write(((copyr % fn) + text))
              f.close()





os.path.walk('/home/david/PYME/PYME', walkFunc, 0)

