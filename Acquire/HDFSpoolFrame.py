#Boa:Frame:FrSpool

import wx
import datetime
import HDFSpooler
import QueueSpooler
#import win32api
from PYME.FileUtils import nameUtils
from PYME.ParallelTasks.relativeFiles import getRelFilename
import os
import subprocess

def create(parent):
    return FrSpool(parent)

[wxID_FRSPOOL, wxID_FRSPOOLBSETSPOOLDIR, wxID_FRSPOOLBSTARTSPOOL, 
 wxID_FRSPOOLBSTOPSPOOLING, wxID_FRSPOOLCBCOMPRESS, wxID_FRSPOOLCBQUEUE, 
 wxID_FRSPOOLPANEL1, wxID_FRSPOOLSTATICBOX1, wxID_FRSPOOLSTATICBOX2, 
 wxID_FRSPOOLSTATICTEXT1, wxID_FRSPOOLSTNIMAGES, wxID_FRSPOOLSTSPOOLDIRNAME, 
 wxID_FRSPOOLSTSPOOLINGTO, wxID_FRSPOOLTCSPOOLFILE, 
] = [wx.NewId() for _init_ctrls in range(14)]

def baseconvert(number,todigits):
        x = number
    
        # create the result in base 'len(todigits)'
        res=""

        if x == 0:
            res=todigits[0]
        
        while x>0:
            digit = x % len(todigits)
            res = todigits[digit] + res
            x /= len(todigits)

        return res


class FrSpool(wx.Frame):
    def _init_ctrls(self, prnt):
        # generated method, don't edit
        wx.Frame.__init__(self, id=wxID_FRSPOOL, name='FrSpool', parent=prnt,
              pos=wx.Point(543, 403), size=wx.Size(285, 253),
              style=wx.DEFAULT_FRAME_STYLE, title='Spooling')
        self.SetClientSize(wx.Size(277, 226))

        self.panel1 = wx.Panel(id=wxID_FRSPOOLPANEL1, name='panel1',
              parent=self, pos=wx.Point(0, 0), size=wx.Size(277, 226),
              style=wx.TAB_TRAVERSAL)

        self.bStartSpool = wx.Button(id=wxID_FRSPOOLBSTARTSPOOL,
              label='Start Spooling', name='bStartSpool', parent=self.panel1,
              pos=wx.Point(186, 67), size=wx.Size(88, 23), style=0)
        self.bStartSpool.Bind(wx.EVT_BUTTON, self.OnBStartSpoolButton,
              id=wxID_FRSPOOLBSTARTSPOOL)

        self.staticBox1 = wx.StaticBox(id=wxID_FRSPOOLSTATICBOX1,
              label='Spooling Progress', name='staticBox1', parent=self.panel1,
              pos=wx.Point(7, 116), size=wx.Size(265, 104), style=0)
        self.staticBox1.Enable(False)

        self.stSpoolingTo = wx.StaticText(id=wxID_FRSPOOLSTSPOOLINGTO,
              label='Spooling to .....', name='stSpoolingTo',
              parent=self.panel1, pos=wx.Point(26, 140), size=wx.Size(76, 13),
              style=0)
        self.stSpoolingTo.Enable(False)

        self.stNImages = wx.StaticText(id=wxID_FRSPOOLSTNIMAGES,
              label='NNNNN images spooled in MM minutes', name='stNImages',
              parent=self.panel1, pos=wx.Point(26, 164), size=wx.Size(181, 13),
              style=0)
        self.stNImages.Enable(False)

        self.bStopSpooling = wx.Button(id=wxID_FRSPOOLBSTOPSPOOLING,
              label='Stop', name='bStopSpooling', parent=self.panel1,
              pos=wx.Point(55, 188), size=wx.Size(75, 23), style=0)
        self.bStopSpooling.Enable(False)

        self.bStopSpooling.Bind(wx.EVT_BUTTON, self.OnBStopSpoolingButton,
              id=wxID_FRSPOOLBSTOPSPOOLING)

        self.bAnalyse = wx.Button(id = -1,
              label='Analyse', name='bAnalyse', parent=self.panel1,
              pos=wx.Point(160, 188), size=wx.Size(75, 23), style=0)
        self.bAnalyse.Enable(False)

        self.bAnalyse.Bind(wx.EVT_BUTTON, self.OnBAnalyse)

        self.staticBox2 = wx.StaticBox(id=wxID_FRSPOOLSTATICBOX2,
              label='Spool Directory', name='staticBox2', parent=self.panel1,
              pos=wx.Point(8, 8), size=wx.Size(264, 48), style=0)

        self.stSpoolDirName = wx.StaticText(id=wxID_FRSPOOLSTSPOOLDIRNAME,
              label='Save images in: Blah Blah', name='stSpoolDirName',
              parent=self.panel1, pos=wx.Point(21, 28), size=wx.Size(136, 13),
              style=0)

        self.bSetSpoolDir = wx.Button(id=wxID_FRSPOOLBSETSPOOLDIR, label='Set',
              name='bSetSpoolDir', parent=self.panel1, pos=wx.Point(222, 23),
              size=wx.Size(40, 23), style=0)
        self.bSetSpoolDir.SetThemeEnabled(False)
        self.bSetSpoolDir.Bind(wx.EVT_BUTTON, self.OnBSetSpoolDirButton,
              id=wxID_FRSPOOLBSETSPOOLDIR)

        self.tcSpoolFile = wx.TextCtrl(id=wxID_FRSPOOLTCSPOOLFILE,
              name='tcSpoolFile', parent=self.panel1, pos=wx.Point(81, 68),
              size=wx.Size(100, 21), style=0, value='dd_mm_series_a')
        self.tcSpoolFile.Bind(wx.EVT_TEXT, self.OnTcSpoolFileText,
              id=wxID_FRSPOOLTCSPOOLFILE)

        self.staticText1 = wx.StaticText(id=wxID_FRSPOOLSTATICTEXT1,
              label='Series name:', name='staticText1', parent=self.panel1,
              pos=wx.Point(11, 72), size=wx.Size(66, 13), style=0)

        self.cbCompress = wx.CheckBox(id=wxID_FRSPOOLCBCOMPRESS,
              label=u'Enable Compression', name='cbCompress',
              parent=self.panel1, pos=wx.Point(148, 97), size=wx.Size(124, 13),
              style=0)
        self.cbCompress.SetValue(False)

        self.cbQueue = wx.CheckBox(id=wxID_FRSPOOLCBQUEUE,
              label=u'Save to Queue', name=u'cbQueue', parent=self.panel1,
              pos=wx.Point(12, 97), size=wx.Size(124, 13), style=0)
        self.cbQueue.SetValue(False)

    def __init__(self, parent, scope, defDir, defSeries='%(day)d_%(month)d_series'):
        self._init_ctrls(parent)
        self.scope = scope
        
        dtn = datetime.datetime.now()
        
        #dateDict = {'username' : win32api.GetUserName(), 'day' : dtn.day, 'month' : dtn.month, 'year':dtn.year}
        
        self.dirname = defDir % nameUtils.dateDict
        self.seriesStub = defSeries % nameUtils.dateDict

        self.seriesCounter = 0
        self.seriesName = self._GenSeriesName()
        
        #if we've had to quit for whatever reason start where we left off
        while os.path.exists(os.path.join(self.dirname, self.seriesName + '.h5')):
            self.seriesCounter +=1
            self.seriesName = self._GenSeriesName()
        
        self.stSpoolDirName.SetLabel(self.dirname)
        self.tcSpoolFile.SetValue(self.seriesName)

    def _GenSeriesName(self):
        return self.seriesStub + '_' + self._NumToAlph(self.seriesCounter)

    def _NumToAlph(self, num):
        return baseconvert(num, 'ABCDEFGHIJKLMNOPQRSTUVXWYZ')
        

    def OnBStartSpoolButton(self, event):
        #fn = wx.FileSelector('Save spooled data as ...', default_extension='.log',wildcard='*.log')
        #if not fn == '': #if the user cancelled 
        #    self.spooler = Spooler.Spooler(self.scope, fn, self.scope.pa, self)
        #    self.bStartSpool.Enable(False)
        #    self.bStopSpooling.Enable(True)
        #    self.stSpoolingTo.Enable(True)
        #    self.stNImages.Enable(True)
        #    self.stSpoolingTo.SetLabel('Spooling to ' + fn)
        #    self.stNImages.SetLabel('0 images spooled in 0 minutes')
        
        fn = self.tcSpoolFile.GetValue()

        if fn == '': #sanity checking
            wx.MessageBox('Please enter a series name', 'No series name given', wx.OK)
            return #bail
        
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        if (fn + '.h5') in os.listdir(self.dirname): #check to see if data with the same name exists
            ans = wx.MessageBox('A series with the same name already exists ... overwrite?', 'Warning', wx.YES_NO)
            if ans == wx.NO:
                return #bail
            
        if self.cbCompress.GetValue():
            compLevel = 6
        else:
            compLevel = 0

        if self.cbQueue.GetValue():
            self.queueName = getRelFilename(self.dirname + fn + '.h5')
            self.spooler = QueueSpooler.Spooler(self.scope, self.queueName, self.scope.pa, self, complevel=compLevel)
            self.bAnalyse.Enable(True)
        else:
            self.spooler = HDFSpooler.Spooler(self.scope, self.dirname + fn + '.h5', self.scope.pa, self, complevel=compLevel)
        self.bStartSpool.Enable(False)
        self.bStopSpooling.Enable(True)
        self.stSpoolingTo.Enable(True)
        self.stNImages.Enable(True)
        self.stSpoolingTo.SetLabel('Spooling to ' + fn)
        self.stNImages.SetLabel('0 images spooled in 0 minutes')
        

    def OnBStopSpoolingButton(self, event):
        self.spooler.StopSpool()
        self.bStartSpool.Enable(True)
        self.bStopSpooling.Enable(False)
        self.stSpoolingTo.Enable(False)
        self.stNImages.Enable(False)

        self.seriesCounter +=1
        self.seriesName = self._GenSeriesName() 
        self.tcSpoolFile.SetValue(self.seriesName)

    def OnBAnalyse(self, event):
        subprocess.Popen('../DSView/dh5view.py QUEUE://%s' % self.queueName, shell=True)
        
    def Tick(self):
        dtn = datetime.datetime.now()
        
        dtt = dtn - self.spooler.dtStart
        
        self.stNImages.SetLabel('%d images spooled in %d seconds' % (self.spooler.imNum, dtt.seconds))

    def OnBSetSpoolDirButton(self, event):
        ndir = wx.DirSelector()
        if not ndir == '':
            self.dirname = ndir + os.sep
            self.stSpoolDirName.SetLabel(self.dirname)

            #if we've had to quit for whatever reason start where we left off
            while os.path.exists(os.path.join(self.dirname, self.seriesName + '.h5' )):
                self.seriesCounter +=1
                self.seriesName = self._GenSeriesName()
                self.tcSpoolFile.SetValue(self.seriesName)

    def OnTcSpoolFileText(self, event):
        event.Skip()
        
