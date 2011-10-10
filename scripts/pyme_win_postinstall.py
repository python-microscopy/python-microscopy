# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 21:24:45 2011

@author: David
"""
import os
import sys
import subprocess
import logging
logging.basicConfig(filename=r'c:\Temp\example.log',level=logging.DEBUG)

if sys.argv[1] == '-install':
    #Set up file ascociations
    #logging.debug('about to get path')
    os.system('pyme_ascociations.bat')
    #os.system(['assoc .h5=PYME.RawData'])
    #logging.debug('about to get path')
    #subprocess.call(['assoc', '.h5r=PYME.AnalysedData'])
    #subprocess.call(['assoc', '.psf=PYME.PSF'])
    #subprocess.call(['assoc', '.sf=PYME.ShiftField'])
    #subprocess.call(['assoc', '.md=PYME.Metadata'])
    
    #most people probably don't want to ascociate .tif with PYME
    #os.system('assoc .tiff=PYME.Tiff')
    #os.system('assoc .tif=PYME.Tiff')
    
    #subprocess.call(['ftype', 'PYME.AnalysedData=VisGUI %*'])
    #subprocess.call(['ftype',  'PYME.RawData=dh5view %*'])
    #subprocess.call(['ftype',  'PYME.PSF=dh5view %*'])
    #os.system('ftype PYME.Tiff=dh5view %*')
    #subprocess.call(['ftype', 'PYME.Metadata=dh5view %*'])
    
    logging.debug('about to get path')
    programsPath = get_special_folder_path("CSIDL_COMMON_PROGRAMS")
    logging.debug('programsPath')
    pymeShortcutPath = os.path.join(programsPath, 'PYME')
    logging.debug('shortcut path = %s' % pymeShortcutPath)
    
    if not os.path.exists(pymeShortcutPath):
        os.mkdir(pymeShortcutPath)
    directory_created(pymeShortcutPath)
    
    sp = os.path.join(pymeShortcutPath, 'DSView.lnk')
    if os.path.exists(sp):
        os.remove(sp)
    create_shortcut('dh5view.cmd', 'Data viewer & analysis', 
                    os.path.join(pymeShortcutPath, 'DSView.lnk'))
    file_created(os.path.join(pymeShortcutPath, 'DSView.lnk'))
    
    sp = os.path.join(pymeShortcutPath, 'LMVis.lnk')
    if os.path.exists(sp):
        os.remove(sp)    
    create_shortcut('VisGUI.cmd', 'Visualisation', 
                    os.path.join(pymeShortcutPath,  'LMVis.lnk'))
    file_created(os.path.join(pymeShortcutPath,  'LMVis.lnk'))
                    
    
    sp = os.path.join(pymeShortcutPath, 'PYMEAcquire.lnk')
    if os.path.exists(sp):
        os.remove(sp)
    create_shortcut('PYMEAcquire.cmd', 'Data acquisition [simulated]', 
                    os.path.join(pymeShortcutPath,  'PYMEAcquire.lnk'))
    file_created(os.path.join(pymeShortcutPath,  'PYMEAcquire.lnk'))
                    
    
    sp = os.path.join(pymeShortcutPath, 'launchWorkers.lnk')
    if os.path.exists(sp):
        os.remove(sp)
    create_shortcut('launchWorkers.cmd', 'Worker processes for data processing', 
                    os.path.join(pymeShortcutPath,  'launchWorkers.lnk'), '', 'c:\\')#os.path.split(__file__)[0])
    file_created(os.path.join(pymeShortcutPath,  'launchWorkers.lnk'))
    
    
    #create_shortcut('dh5view.cmd', 'Data viewer', pymeShortcutPath + [, arguments[, workdir[, iconpath[, iconindex]]]])
    
    


