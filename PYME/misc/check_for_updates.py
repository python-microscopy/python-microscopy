from PYME import version
import os
import json
import logging
logger = logging.getLogger(__name__)

update_messages = {
    'git' : '''You have a development install, to update, execute the following in the repository root folder:
    
    git pull origin master
    python setup.py build_ext -i
    
    NB: this will update to the bleeding edge (i.e. past the last official release)
    ''',
    
    'pypi' : '''You have a pip install, to update, close all PYME programs and execute the following in the python
    prompt for your PYME virtual environment.
    
    pip install python-microscopy -U
    ''',
    
    'conda' : '''You have a conda install, to update, close all PYME programs and execute the following in the python
    prompt for your PYME environment.
    
    conda update -C david_baddeley -S python-microscopy
    '''
}

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
    

def check_for_updates(gui=True):
    import requests
    import packaging.version
    
    try:
        version_info = requests.get('http://www.python-microscopy.com/current_version.json').json()
        
        if packaging.version.parse(version_info['version']) > packaging.version.parse(version.version):
            update_msg = 'A new version of PYME is available\nYou have version %s, the current version is %s' % (version.version, version_info['version']))
            logger.info(update_msg)
            
            install_type = guess_install_type()
            logger.info(update_messages[install_type])
            
            if gui:
                import wx
                msg = update_msg + '\n\n' + update_messages[install_type]
                # tODO - add an option to supress this dialog
                wx.MessageBox(msg, 'A PYME update is available')
        
    except:
        logger.exception('Error getting info on updates')
    