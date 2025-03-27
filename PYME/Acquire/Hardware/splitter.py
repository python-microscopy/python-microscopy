import os.path
#!/usr/bin/python

##################
# focusKeys.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import wx
from PYME.DSView.arrayViewPanel import ArrayViewPanel
from PYME.DSView.DisplayOptionsPanel import OptionsPanel
import numpy
import os
import wx.lib.agw.aui as aui
from PYME.IO import MetaDataHandler
from PYME.IO.FileUtils import nameUtils
from PYME.Analysis import splitting

class Splitter(object):
    def __init__(self, parent, scope, cam, dir='up_down', flipChan=1, dichroic = 'Unspecified', transLocOnCamera = 'Top',
                 constrain=True, flip = True, cam_name='', rois=None):
        self.dir = dir
        self.scope = scope
        self.cam = cam
        self.flipChan=flipChan
        self.parent = parent
        self.flip = flip
        self._rois=rois
        pixelsize_nm = 1e3*(scope.GetPixelSize(cam)[0])
        self.unmixer = splitting.Unmixer(flip=flip, axis = dir, chanROIs=self.rois, pixelsize=pixelsize_nm)


        #which dichroic mirror is installed
        self.dichroic = dichroic
        #where is the transmitted image on the camera (one of 'Top', 'Bottom', 'Left', 'Right')
        self.transLocOnCamera = transLocOnCamera

        #register as a producer of metadata
        MetaDataHandler.provideStartMetadata.append(self.ProvideMetadata)

        cam.splitting='none'
        cam.splitterFlip = flip

        self.offset = 0
        self.mixMatrix = numpy.array([[1.,0.],[0.,1.]])

        self.constrainROI = False
        self.flipView = False
        self.f = None

        if not cam_name == '':
            suff = ' [%s]' %cam_name
        else:
            suff = ''
        
        self.miConstrROI = parent.AddMenuItem('Splitter', 'Constrain ROI%s' % suff, self.OnConstrainROI, itemType = 'check')
        parent.AddMenuItem('Splitter', 'Flip view%s' % suff, self.OnFlipView)
        parent.AddMenuItem('Splitter', 'Unmix%s\tF7' % suff, self.OnUnmix)
        parent.AddMenuItem('Splitter', 'SetShiftField%s' % suff, self.OnSetShiftField)

#        idConstROI = wx.NewIdRef()
#        idFlipView = wx.NewIdRef()
#        idUnmix = wx.NewIdRef()
#        idShiftfield = wx.NewIdRef()
#
#        self.menu = wx.Menu(title = '')
        
#
#        self.menu.AppendCheckItem(idConstROI, 'Constrain ROI')
#        wx.EVT_MENU(parent, idConstROI, self.OnConstrainROI)
#
#        self.menu.AppendCheckItem(idFlipView, 'Flip view')
#        wx.EVT_MENU(parent, idFlipView, self.OnFlipView)
#        self.menu.Append(idUnmix, 'Unmix\tF7')
#        wx.EVT_MENU(parent, idUnmix, self.OnUnmix)
#
#        self.menu.Append(idShiftfield, 'Set Shift Field')
#        wx.EVT_MENU(parent, idShiftfield, self.OnSetShiftField)
#
#        menu.AppendSeparator()
#        menu.AppendMenu(-1, '&Splitter', self.menu)

        if constrain:
            self.OnConstrainROI()
            #self.menu.Check(idConstROI, True)
        

    @property
    def rois(self):
        if self._rois is not None:
            return self._rois
        if self.dir == 'up_down':
            return [[0,0,self.cam.GetCCDWidth(), self.cam.GetCCDHeight()/2], 
                    [0,self.cam.GetCCDHeight()/2,self.cam.GetCCDWidth(), self.cam.GetCCDHeight()/2]]
        else: #dir == 'left_right'
            return [[0,0,self.cam.GetCCDWidth()/2, self.cam.GetCCDHeight()],
                    [self.cam.GetCCDWidth()/2,0,self.cam.GetCCDWidth()/2, self.cam.GetCCDHeight()]]
    
    def ProvideMetadata(self, mdh):
        from PYME.LMVis import dyeRatios #TODO - move to somewhere common
        
        if self.scope.cam == self.cam:#only if the currently selected camera is being split
            mdh.setEntry('Splitter.Dichroic', self.dichroic)
            mdh.setEntry('Splitter.TransmittedPathPosition', self.transLocOnCamera)
            mdh.setEntry('Splitter.Flip', self.flip)
            
            try:
                mdh.setEntry('Splitter.Ratios', dyeRatios.get_ratios(self.dichroic, self.scope.microscope_name))
            except (KeyError, AttributeError):
                pass
            
            mdh['Splitter.Channel0ROI'], mdh['Splitter.Channel1ROI'] = self.rois

            if 'shiftField' in dir(self):
                mdh.setEntry('chroma.ShiftFilename', self.shiftFieldName)
                dx, dy = self.shiftField
                mdh.setEntry('chroma.dx', dx)
                mdh.setEntry('chroma.dy', dy)

    def OnConstrainROI(self,event=None):
        self.constrainROI = not self.constrainROI
        if self.constrainROI:
            self.cam.splitting = self.dir
        else:
            self.cam.splitting = 'none'

    def OnFlipView(self,event):
        self.flipView = not self.flipView
        if self.flipView:
            self.scope.vp.do.setFlip(self.flipChan, 1)
        else:
            self.scope.vp.do.setFlip(self.flipChan, 0)

    def OnUnmix(self,event):
        #self.Unmix()
        if (not 'f' in dir(self)) or (self.f is None):
            self.f = UnMixPanel(self.parent, splitter = self, size=(500, 275))
            #self.o = OptionsPanel(self.parent, self.f.vp.do, horizOrientation=True)
            self.o = UnMixSettingsPanel(self.parent, splitter = self, size=(240, 300))
            self.parent.AddCamTool(self.o, 'Unmixing Settings')
            #self.f.SetSize((800,500))
            #self.f.Show()
            self.cpinfo = aui.AuiPaneInfo().Name("unmixingDisplay").Caption("Unmixing").Bottom().CloseButton(True)
            #cpinfo.dock_proportion  = int(cpinfo.dock_proportion*1.6)

            self.parent._mgr.AddPane(self.f, self.cpinfo)
            self.parent._mgr.Update()
        elif not self.f.IsShown():
            self.parent._mgr.ShowPane(self.f, True)
        else:
            self.f.vp.do.Optimise()
            
        #if not self.f.update in self.scope.frameWrangler.WantFrameGroupNotification:
        #    self.scope.frameWrangler.WantFrameGroupNotification.append(self.f.update)
        self.scope.frameWrangler.onFrameGroup.connect(self.f.update)


    def OnSetShiftField(self, event):
        fdialog = wx.FileDialog(None, 'Select shift field',
            wildcard='*.sf', style=wx.FD_OPEN, defaultDir = nameUtils.genShiftFieldDirectoryPath())
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            sfname = fdialog.GetPath()
            
            self.SetShiftField(sfname)

    def SetShiftField(self, sfname):
        from PYME.IO.compatibility import np_load_legacy
        self.shiftField = np_load_legacy(sfname)
        self.shiftFieldName = sfname
        self.unmixer.SetShiftField(self.shiftField, self.scope)


    def Unmix(self):
        dsa = self.scope.frameWrangler.currentFrame.squeeze()

        return self.unmixer.Unmix(dsa, self.mixMatrix, self.offset, ROI=self.scope.cam.GetROI())


class UnMixSettingsPanel(wx.Panel):
    def __init__(self, parent, splitter = None, size=(-1, -1)):
        wx.Panel.__init__(self,parent, -1, size=size)

        self.splitter = splitter

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.op = OptionsPanel(self, splitter.f.vp.do, horizOrientation=True)
        vsizer.Add(self.op, 0, wx.ALL, 0)

        psizer = wx.BoxSizer(wx.HORIZONTAL)

        bsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Mix Matrix'), wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tMM00 = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.mixMatrix[0,0]), size=(40,-1))
        hsizer.Add(self.tMM00, 1, wx.ALL,2 )

        self.tMM01 = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.mixMatrix[0,1]), size=(40,-1))
        hsizer.Add(self.tMM01, 1, wx.ALL,2 )

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.tMM10 = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.mixMatrix[1,0]), size=(40,-1))
        hsizer.Add(self.tMM10, 1, wx.ALL,2 )

        self.tMM11 = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.mixMatrix[1,1]), size=(40,-1))
        hsizer.Add(self.tMM11, 1, wx.ALL,2 )

        bsizer.Add(hsizer, 0, wx.ALL, 0)

        psizer.Add(bsizer, 0, wx.ALL, 0)


        bsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Offset'), wx.HORIZONTAL)
        self.tOffset = wx.TextCtrl(self, -1, '%1.2f'%(self.splitter.offset), size=(40,-1))
        self.bGrabOffset = wx.Button(self, -1, 'C', style = wx.BU_EXACTFIT)

        bsizer.Add(self.tOffset, 1, wx.ALL, 0)
        bsizer.Add(self.bGrabOffset, 0, wx.LEFT, 5)
        psizer.Add(bsizer, 1, wx.LEFT|wx.RIGHT, 5)

#        self.bUpdate = wx.Button(self, -1, 'Update')
#        vsizer.Add(self.bUpdate, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)
#        self.bUpdate.Bind(wx.EVT_BUTTON, self.OnUpdateMix)

        vsizer.Add(psizer, 1, wx.ALL|wx.EXPAND, 0)
        self.SetSizerAndFit(vsizer)

        self.bGrabOffset.Bind(wx.EVT_BUTTON, self.OnGrabOffsetFromCamera)
        #self.splitter.scope.frameWrangler.WantFrameGroupNotification.append(self.OnUpdateMix)
        self.splitter.scope.frameWrangler.onFrameGroup.connect(self.OnUpdateMix)



    def OnUpdateMix(self, event=None, **kwargs):
        self.splitter.mixMatrix[0,0]= float(self.tMM00.GetValue())
        self.splitter.mixMatrix[0,1]= float(self.tMM01.GetValue())
        self.splitter.mixMatrix[1,0]= float(self.tMM10.GetValue())
        self.splitter.mixMatrix[1,1]= float(self.tMM11.GetValue())
        self.splitter.offset= float(self.tOffset.GetValue())

    def OnGrabOffsetFromCamera(self, event):
        if 'ADOffset' in dir(self.splitter.scope.cam):
            self.tOffset.SetValue('%3.2f' % self.splitter.scope.cam.ADOffset)
            self.update()





class UnMixPanel(wx.Panel):
    def __init__(self, parent=None, title='Unmixing', splitter = None, size=(-1, -1)):
        wx.Panel.__init__(self,parent, -1, size=size)

        self.splitter = splitter

        self.ds = self.splitter.Unmix()
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)

#        pan = wx.Panel(self, -1)
#        psizer = wx.BoxSizer(wx.VERTICAL)
#
#        bsizer = wx.StaticBoxSizer(wx.StaticBox(pan, -1, 'Mix Matrix'), wx.VERTICAL)
#
#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#
#        self.tMM00 = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.mixMatrix[0,0]), size=(40,-1))
#        hsizer.Add(self.tMM00, 1, wx.ALL,2 )
#
#        self.tMM01 = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.mixMatrix[0,1]), size=(40,-1))
#        hsizer.Add(self.tMM01, 1, wx.ALL,2 )
#
#        bsizer.Add(hsizer, 0, wx.ALL, 0)
#
#        hsizer = wx.BoxSizer(wx.HORIZONTAL)
#
#        self.tMM10 = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.mixMatrix[1,0]), size=(40,-1))
#        hsizer.Add(self.tMM10, 1, wx.ALL,2 )
#
#        self.tMM11 = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.mixMatrix[1,1]), size=(40,-1))
#        hsizer.Add(self.tMM11, 1, wx.ALL,2 )
#
#        bsizer.Add(hsizer, 0, wx.ALL, 0)
#
#        psizer.Add(bsizer, 0, wx.ALL, 5)
#
#
#        bsizer = wx.StaticBoxSizer(wx.StaticBox(pan, -1, 'Offset'), wx.HORIZONTAL)
#        self.tOffset = wx.TextCtrl(pan, -1, '%1.2f'%(self.splitter.offset), size=(40,-1))
#        self.bGrabOffset = wx.Button(pan, -1, 'C', style = wx.BU_EXACTFIT)
#
#        bsizer.Add(self.tOffset, 1, wx.ALL, 0)
#        bsizer.Add(self.bGrabOffset, 0, wx.LEFT, 5)
#        psizer.Add(bsizer, 0, wx.ALL|wx.EXPAND, 5)
#
##        self.bUpdate = wx.Button(pan, -1, 'Update')
##        psizer.Add(self.bUpdate, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)
##        self.bUpdate.Bind(wx.EVT_BUTTON, self.OnUpdateMix)
#
#        pan.SetSizerAndFit(psizer)
#
#        sizer.Add(pan, 0, 0, 0)

        self.vp = ArrayViewPanel(self, self.ds, initial_overlays=[])
        sizer.Add(self.vp, 1,wx.EXPAND,0)
        #self.SetAutoLayout(1)
        self.SetSizer(sizer)

        #self.Layout()
        #self.update()
        wx.EVT_CLOSE(self, self.OnCloseWindow)
        wx.EVT_SIZE(self, self.OnSize)
        wx.EVT_SHOW(self, self.OnShow)

        #self.bGrabOffset.Bind(wx.EVT_BUTTON, self.OnGrabOffsetFromCamera)

        #self.statusbar = self.CreateStatusBar(1, wx.STB_SIZEGRIP)

        #self.Layout()
        


    def update(self, caller=None, **kwargs):
        #print 'u'
        #print self.tMM00.GetValue(), self.tMM01.GetValue()
#        self.splitter.mixMatrix[0,0]= float(self.tMM00.GetValue())
#        self.splitter.mixMatrix[0,1]= float(self.tMM01.GetValue())
#        self.splitter.mixMatrix[1,0]= float(self.tMM10.GetValue())
#        self.splitter.mixMatrix[1,1]= float(self.tMM11.GetValue())
#        self.splitter.offset= float(self.tOffset.GetValue())


        if self.IsShown():
            self.vp.ResetDataStack(self.splitter.Unmix())
            self.vp.Redraw()#imagepanel.Refresh()

    def OnCloseWindow(self, event):
        #self.splitter.scope.frameWrangler.WantFrameGroupNotification.remove(self.update)
        self.splitter.scope.frameWrangler.onFrameGroup.disconnect(self.update)
        
        self.splitter.f = None
        self.Destroy()

#    def OnUpdateMix(self, event):
#        self.splitter.mixMatrix[0,0]= float(self.tMM00.GetValue())
#        self.splitter.mixMatrix[0,1]= float(self.tMM01.GetValue())
#        self.splitter.mixMatrix[1,0]= float(self.tMM10.GetValue())
#        self.splitter.mixMatrix[1,1]= float(self.tMM11.GetValue())
#        self.splitter.offset= float(self.tOffset.GetValue())

        #print self.splitter.mixMatrix

        self.update()

    def OnGrabOffsetFromCamera(self, event):
        if 'ADOffset' in dir(self.splitter.scope.cam):
            self.tOffset.SetValue('%3.2f' % self.splitter.scope.cam.ADOffset)
            self.update()

    def OnSize(self, event):
        self.Layout()
        event.Skip()

    def OnShow(self, event):
        self.splitter.o.GetParent().Show(event.IsShown())


   
    

        

        
