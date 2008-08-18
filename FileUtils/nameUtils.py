import os
import re
import datetime
import sys

seps = re.compile('[\\\\/*]')

def getUsername():
    '''
    Returns the user name in a platform dependant way
    '''
    if sys.platform == 'win32':
        import win32api
        return win32api.GetUserName()
    else: #linux, _should_ hopefully also work for mac
        #return os.getlogin() #broken when not runing from command line
        return os.environ['USERNAME']
        
    

dtn = datetime.datetime.now()

homedir = os.path.expanduser('~') #unix & possibly others ...
if 'USERPROFILE' in os.environ.keys(): #windows
    homedir = os.environ['USERPROFILE']

datadir = '/media/data/'
if 'PYMEDATADIR' in os.environ.keys():
    datadir = os.environ['PYMEDATADIR']
        

dateDict = {'username' : getUsername(), 'day' : dtn.day, 'month' : dtn.month, 'year':dtn.year, 'sep' : os.sep, 'dataDir' : datadir, 'homeDir': homedir}


#\\ / and * will be replaced with os dependant separator
datadirPattern = '%(dataDir)s/%(username)s/%(day)d-%(month)d-%(year)d'
filePattern = '%(day)d_%(month)d_series'

resultsdirPattern = '%(homeDir)s/analysis/%(dday)d-%(dmonth)d-%(dyear)d'
resultsdirPatternShort = '%(homeDir)s/analysis/'

def genHDFDataFilepath(name, create=True):
    p =  os.path.join(*sep.split(datadirPattern)) % dateDict
    if create and not os.path.exists(p): #create the necessary directories
        os.makedirs(p)

    return os.path.join(p, name)

def genResultFileName(dataFileName, create=True):
    fn, ext = os.path.splitext(dataFileName) #remove extension
    #print os.path.join(*seps.split(resultsdirPatternShort)) % dateDict
    p = os.path.join(*(seps.split(resultsdirPatternShort) + seps.split(fn)[-2:])) %dateDict

    if create and not os.path.exists(os.path.split(p)[0]): #create the necessary directories
        os.makedirs(os.path.split(p)[0])

    return p + '.h5r'
    
