from PYME import version
from PYME import config
import os
import json
import logging
import subprocess
import datetime
logger = logging.getLogger(__name__)
import shelve

update_info_fn = os.path.join(config.user_config_dir, 'update.shv')

update_messages = {
    'git' : '''You have a development install, to update, execute the following in the repository root folder:
    
    git pull origin master
    python setup.py build_ext -i
    
NB: because this is a developent install, this will update to the bleeding edge (i.e. past the last official release)''',
    
    'pypi' : '''You have a pip install, to update, close all PYME programs and execute the following in the python
prompt for your PYME virtual environment.
    
    pip install python-microscopy -U''',
    
    'conda' : '''You have a conda install, to update, close all PYME programs and execute the following in the python
prompt for your PYME environment.
    
    conda update -C david_baddeley -S python-microscopy'''
}

# notes: - the explicit channel specification may not strictly be required, but is the most robust option
#        - the -S option does a fast solve and doesn't aggressively update packages (i.e. only updates dependencies if
#          explicitly forced to by a pin). This will hopefully a) make it faster and b) stop us breaking too may conda environments.


update_available = None
update_ver = None

def guess_install_type():
    # check for git
    pyme_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(pyme_parent_dir)
    
    if os.path.exists(os.path.join(pyme_parent_dir, '.git')):
        logger.info('Detected a .git folder, assuming a development install')
        return 'git'
    
    # check for conda package
    try:
        pkg_info = json.loads(subprocess.check_output(['conda', 'list', '--json', 'python-microscopy']))
    except FileNotFoundError:
        logger.exception('Could not find `conda`\nPYME is not installed in a conda environment')
        return 'unknown'
    
    chan = pkg_info[0]['channel']
    
    if chan == 'pypi':
        logger.info('Detected a pip install')
        return 'pip'
    elif chan == 'david_baddeley':
        logger.info('Detected a conda install')
        return 'conda'
    

def check_for_updates(gui=True, force=False):
    global update_available, update_ver
    import requests
    import packaging.version
    
    with shelve.open(update_info_fn) as s:
        next_update_time = s.get('last_update_check', datetime.datetime.fromtimestamp(0)) + datetime.timedelta(days=1)
        t = datetime.datetime.now()
    
        if not(force or (config.get('check_for_updates', True) and t > next_update_time)):
            # respect config setting and bail
            # called with force=True when called from the menu (i.e. explicitly rather than automatically)
            return

        s['last_update_check'] = t
    
    logger.info('Checking for updates ...')
    try:
        version_info = requests.get('http://www.python-microscopy.org/current_version.json').json()
        
        update_ver = version_info['version']
        
        if packaging.version.parse(update_ver) > packaging.version.parse(version.version):
            update_msg = 'A new version of PYME is available\nYou have version %s, the current version is %s' % (version.version, update_ver)
            logger.info(update_msg)
            
            install_type = guess_install_type()
            logger.info(update_messages[install_type])
            
            update_available = True
            
            if gui:
                gui_prompt_update()
                
        else:
            update_available = False
        
    except:
        logger.exception('Error getting info on updates')
    
    
def gui_prompt_update():
    import wx
    
    # make sure there is an update available
    assert update_available
    
    install_type = guess_install_type()
    update_msg = 'A new version of PYME is available\nYou have version %s, the current version is %s' % (version.version, update_ver)
    msg = update_msg + '\n\n' + update_messages[install_type]
    wx.MessageBox(msg, 'A PYME update is available')
    
    with shelve.open(update_info_fn) as s:
        s['last_update_offered'] =  update_ver
    
    
def gui_prompt_once():
    if update_available is None:
        #make doubly sure we've checked
        check_for_updates(False)
        
        
    if update_available:
        with shelve.open(update_info_fn) as s:
            last_offered = s.get('last_update_offered', None)
            
        if (last_offered != update_ver):
            gui_prompt_update()
        
        
check_for_updates(False)
        
        

