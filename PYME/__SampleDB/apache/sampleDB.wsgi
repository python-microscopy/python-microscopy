import os, sys
sys.path.append('/home/david/PYME/PYME')
sys.path.append('/home/david/PYME')
os.environ['DJANGO_SETTINGS_MODULE'] = 'SampleDB.settings'
os.environ['MPLCONFIGDIR'] = '/tmp'

import django.core.handlers.wsgi

application = django.core.handlers.wsgi.WSGIHandler()