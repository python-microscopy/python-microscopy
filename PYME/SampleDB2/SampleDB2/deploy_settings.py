from settings import *

DATABASE_HOST = 'DBSRV1'
#look for database host in environment variable
if 'PYME_DATABASE_HOST' in os.environ.keys():
    DATABASE_HOST = os.environ['PYME_DATABASE_HOST']

DATABASES = {
	'default': {
        'NAME': 'sample_db',
        'HOST' : DATABASE_HOST,
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'sample_db',
        'PASSWORD': 'PYMEUSER',
        'OPTIONS': {"connect_timeout": 5,},
    },
    #'default': {
    #    'ENGINE': 'django.db.backends.sqlite3',
    #    'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    #}
}