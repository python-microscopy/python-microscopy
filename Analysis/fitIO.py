import wx
import os
import cPickle
import string
import Image

def _genResultFilename(res, resultsDir = None):
    if resultsDir == None:
        resultsDir = '/home/%s/analysis/' % os.getlogin()

    imfname = res.keys()[0]

    if not imfname.__class__ == int:
        (aqDate, serName) = imfname.split(os.sep)[-3:-1]
    else:
        imfname = res[imfname].filename
        (aqDate, serName) = imfname.split(os.sep)[-2:]
        serName = serName.split('.')[0]
    
    return resultsDir + aqDate + '/' + serName + '_res.pik'
    


def saveResults(res, fname=None):
    if fname == None:
        fname = _genResultFilename(res)

    #make directory if needed
    dirs = string.join(fname.split(os.sep)[:-1], os.sep)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    fid = open(fname, 'wb')
    cPickle.dump(res, fid, 2)
    fid.close()


def loadResults(fname = None):
    if fname == None:
        fname = wx.FileSelector('Please select pickled fit results', wildcard='Pickle objects|*.pik', default_extension='pik')
        if fname == '':
            raise 'No file selected'

    fid = open(fname, 'rb')

    res = cPickle.load(fid)

    fid.close()
    return res

