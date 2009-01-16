from distutils.core import setup
import py2exe
import os
import shutil
#import matplotlib
setup(console=['taskWorkerME.py'],
      options={'py2exe':{'excludes':['matplotlib','pyreadline', 'Tkconstants','Tkinter','tcl', '_imagingtk','PIL._imagingtk', 'ImageTK', 'PIL.ImageTK', 'FixTk'], 'includes':['pyscr']}})
os.system('python buildScr.py PYMEScreensaver.py')
shutil.copyfile('Dist/PYMEScreensaver/PYMEScreensaver.scr', 'Dist/PYMEScreensaver.scr')
shutil.rmtree('Dist/PYMEScreensaver')
