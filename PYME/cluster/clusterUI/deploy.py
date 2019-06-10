"""
Deployment script for clusterUI. Will set up to run with apache on linux.

Run as root.

Creates (and serves) /var/www/clusterUI/static for the static files.


"""
import os
import tempfile
import site

print('Installing mod-wsgi')
os.system('sudo apt-get install libapache2-mod-wsgi')

STATIC_INSTALATION_DIR = '/var/www/clusterUI'

print('collecting static files...')
os.system('python manage.py collectstatic --noinput')

print('Copying static files to installation directory')
if not os.path.exists(STATIC_INSTALATION_DIR):
    os.system('sudo mkdir -p %s' % STATIC_INSTALATION_DIR)

os.system('sudo mv _deploy/* %s/' % STATIC_INSTALATION_DIR)

basedir = os.path.dirname(os.path.abspath(__file__))
anaconda_site_dir = os.path.join(os.path.dirname(site.__file__), 'site-packages')

clusterUI_conf= '''
WSGIScriptAlias / {basedir}/clusterUI/wsgi.py
WSGIPythonPath {anaconda_site_dir}:{basedir}/

<Directory {basedir}/clusterUI/>
<Files wsgi.py>
Require all granted
</Files>
</Directory>

Alias /static/ {static_dir}/static/
<Directory {static_dir}/static/>
Order deny,allow
Allow from all
</Directory>
'''.format(basedir=basedir, static_dir=STATIC_INSTALATION_DIR, anaconda_site_dir=anaconda_site_dir)

print('Writing temporary clusterUI.conf')
temp_conf_file = os.path.join(tempfile.gettempdir(), 'clusterUI.conf')
with open(temp_conf_file, 'w') as f:
    f.write(clusterUI_conf)

print('moving temporary file to /etc/apache2/conf-available/clusterUI.conf')
os.system('sudo mv %s /etc/apache2/conf-available/clusterUI.conf' % temp_conf_file)

print('Enabling clusterUI configuration')
os.system('sudo a2enconf clusterUI')

print('Restarting apache')
os.system('sudo service apache2 reload')


