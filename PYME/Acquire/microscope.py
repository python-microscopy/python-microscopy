#!/usr/bin/python

##################
# funcs.py
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

#imports for timing
#import requests
#import tables
#import numpy as np
#import yaml

#import wx
#from PYME.Acquire import previewaquisator as previewaquisator
from PYME.Acquire.frameWrangler import FrameWrangler

import PYME.Acquire.protocol as protocol

from PYME.IO import MetaDataHandler
from PYME.Acquire.Hardware import ccdCalibrator
from PYME.Acquire.SpoolController import SpoolController
from PYME.Acquire.ActionManager import ActionManager

import sqlite3

import os
import datetime

import warnings
from PYME.contrib import dispatch

#register handlers for ndarrays
from PYME.misc import sqlitendarray

import weakref
import re

import logging
logger = logging.getLogger(__name__)

from contextlib import contextmanager

#class DummyJoystick(object):
#    def enable(*args, **kwargs):
#        pass

class StateHandler(object):
    def __init__(self, key, getFcn=None, setFcn=None, needCamRestart = False):
        """Define a state-change handler for a give state key, based on supplied
        get and set functions which interact with the underlying hardware.
        
        This wrapper class serves two functions - a) allowing the get and set methods
        to be stored under a single dictionary key in the StateManager, and b)
        making sure a signal is fired when the state changes.
        
        Parameters
        ----------
        
        key : string
            The hardware key - e.g. "Positioning.x", or "Lasers.405.Power". This
            will also be how the hardware state is recorder in the metadata.
        getFcn : function
            The function to call to get the value of the parameter. Should take
            one parameter which is the value to get
        setFcn : function
            The function to call to set the value of the parameter. Should take
            one parameter which is the value to set. Not providing a setFcn results
            in a read-only property.
        needCamRestart : bool
            Does this absolutely need a camera restart (e.g. we are changing which 
            camera we are using). Will override other preferences.
        
        """
        self.getValue = getFcn
        self.setFcn = setFcn
        self.key = key
        
        self.needCamRestart = needCamRestart
        
        self.onChange = dispatch.Signal()
        
    def setValue(self, value, force=False):
        """Set the state of the controlled hardware to the given value and fire
        a signal. The underlying function will not be called if old and new values
        are the same unless force=True.
        """
        if force or (not self.getValue() == value):
            if not self.setFcn is None:
                self.setFcn(value)
                self.onChange.send(self, key=self.key, value=value)
            else:
                logging.debug('No set method registered for key %s - assuming immutable' %  self.key)
            
class StateManager(object):
    """Manages object (microscope) state by calling an appropriate state-handler
    for a given key.
    
    Looks and behaves like a dictionary to the user, with individual pieces of 
    hardware registering themselves to handle individual keys.
    """
    def __init__(self, scope, handlers = {}):
        self._stateHandlers = {}
        self._cachedStates = {}
        
        self._stateHandlers.update(handlers)
        
        self.scope = weakref.ref(scope)
        self.stateChanged = dispatch.Signal()
        
    def __getitem__(self, key):
        #print key
        try: 
            handler = self._stateHandlers[key]
        except KeyError:
            raise KeyError('No handler registered for %s' % key)
            
        return handler.getValue()
        
    def __setitem__(self, key, value):
        self.setItem(key, value)
        #raise NotImplementedError("Cannot set items via the dictionary interface - use setItem()instead")
        #try: 
        #    handler = self._stateHandlers[key]
        #except KeyError:
        #    raise KeyError('No handler registered for %s' % key)
        #    
        #return handler.setState(value)
        
    def setItem(self, key, value, stopCamera=False, force=False):
        """ Set the value of one of our hardware components
        
        Parameters
        ----------
        
        key : string
            The parameter identifier - e.g. "Positioning.x"
        value : object 
            The value to set. This can be anything the registered handler understands
            but designing the handlers to accept something which is easily converted
            to text for the Metadata (see PYME.IO.MetaDataHandler) is advised.
        stopCamera : bool
            Should we stop the camera before setting the value and restart afterwards.
            Useful for things like integration time  which might not be able to be 
            changed on the fly, but also potentially as a way of achieving hardware 
            synchronization.
        force : bool
            Whether we should call the set method even if the current state is 
            already the desired state.
        
        """
            
        return self.setItems({key:value}, stopCamera, force)
        
    def setItems(self, stateDict, stopCamera = False, force=False, ignoreMissing=False):
        """Set multiple items at once - see setItem for details.
        
        Parameters
        ----------
        
        stateDict : dictionary (or dictionary like object)
            a dictionary of key : value pairs for each piece of hardware which
            wants to be updated.
        stopCamera : bool
            Should we stop the camera before setting the values and restart afterwards?
            Useful for things like integration time  which might not be able to be 
            changed on the fly, but also potentially as a way of achieving hardware 
            synchronization.
        force : bool
            Whether we should call the set method even if the current state is 
            already the desired state.
        ignoreMissing : bool
            Should we silently ignore keys that we don't have handlers for. Used
            when setting state from e.g. sequence metadata which might have other
            irrelevent info.
        """
        
        restartCamera = False
        
        
        if stopCamera:
            try:
                restartCamera = self.scope().frameWrangler.isRunning()
                self.scope().frameWrangler.stop()
            except AttributeError:
                logger.error("We don't have a camera yet")
            
        
        for key, value in stateDict.items():
            try: 
                handler = self._stateHandlers[key]
                if handler.needCamRestart:
                    #our hardware absolutely needs a camera restart - 
                    #e.g. changing integration time on IXon. Override stopCamera
                    #setting and force a restart.
                    try:
                        if self.scope().frameWrangler.isRunning():
                           self.scope().frameWrangler.stop()
                           restartCamera = True
                    except AttributeError:
                        logger.error("We don't have a camera yet")
                       
                handler.setValue(value, force)
            except KeyError:
                if not ignoreMissing:
                    raise KeyError('No handler registered for %s' % key)
                
            
        #logger.debug('sending state changed')
        self.stateChanged.send(self)
            
        if restartCamera:
            #logger.debug('preparing framewrangler')
            self.scope().frameWrangler.Prepare()
            logger.debug('restarting framewrangler')
            self.scope().frameWrangler.start()
            logger.debug('restarted framewrangler')
        
        #return a function which tells us if we're done (for use as a task)
        return lambda : True
                   
            
    def update(self, state):
        """Update state from a dictionary
        
        Parameters
        ----------
        
        state : dict
            A dictionary containing the new state
        """
        return self.setItems(state)
        
    def __len__(self):
        return len(self._stateHandlers)
        
    def keys(self):
        return self._stateHandlers.keys()
        
    def __repr__(self):
        s = 'StateManager:\n'
        for k in self.keys():
            s += '%s : %s\n' % (repr(k), repr(self[k]))
            
        return s
        
    def registerHandler(self, key, getFcn = None, setFcn=None, needCamRestart = False):
        """Register a harware key and the ascociated handlers
        
        Parameters
        ----------
        
        key : string
            The hardware key - e.g. "Positioning.x", or "Lasers.405.Power". This
            will also be how the hardware state is recorded in the metadata.
        getFcn : function
            The function to call to get the value of the parameter. Should take
            one parameter which is the value to get
        setFcn : function
            The function to call to set the value of the parameter. Should take
            one parameter which is the value to set. Not providing a setFcn results
            in a read-only property.
        needCamRestart : bool
            Does this absolutely need a camera restart (e.g. we are changing which 
            camera we are using). Will override other preferences.
            
        Notes
        -----
        There are a few conventions and special key names which should be observed
        as this is how the camera will find hardware.
        
        - Keys should be hierachial, separated by dots
        - Piezos and stages should start with "Positioning.", followed by an axis
          name, and use um as a unit. 
        - Lasers should start with "Laser." and define both "<lasername>.Power"
          and "<lasername>.On" (this is to allow lasers with external 
          switches/shutters as well as )
        
        """
        self._stateHandlers[key] = StateHandler(key, getFcn, setFcn, needCamRestart)
        
    def registerChangeListener(self, key, callback):
        """ Registers a function to be called when the state of a particular
        key changes.
        
        The *key* and *value* are provided as keyword arguments to the callback
        the callback should accept have the signature `callback(**kwargs)`,
        or `callback(key, value, **kwargs)`. Other keyword arguments might also
        be given.
        """
        
        self._stateHandlers[key].onChange.connect(callback)
            
        
        
        
    

class microscope(object):
    """

    Attributes
    ----------
    frameWrangler : PYME.Acquire.frameWrangler.FrameWrangler
        Initialized in between hardware initializations and gui initializations.
    
    """
    def __init__(self):
        #list of tuples  of form (class, chan, name) describing the instaled piezo channels
        self.piezos = []
        self.lasers = []
        self.hardwareChecks = []
        
        #entries should be of the form: "x" : (piezo, channel, multiplier)
        # where multiplyier is what to multiply by to get the units to micrometers
        self.positioning = {}
        self.joystick = None

        self.cam = None
        self.cameras = {}
        self.camControls = {}

        self.stackNum = 0

        #self.WantEventNotification = []
 
        self.StatusCallbacks = [] #list of functions which provide status information
        self.CleanupFunctions = [] #list of functions to be called at exit
        self.PACallbacks = [] #list of functions to be called when a new aquisator is created
        
        
        self.saturationThreshold = 16383 #14 bit
        self.lastFrameSaturated = False
        #self.cam.saturationIntervened = False
        
        self.microscope_name = None
        
        self.saturatedMessage = ''

        protocol.scope = self
        ccdCalibrator.setScope(self)
        self.initDone = False

        self._OpenSettingsDB()
        
        self.spoolController = SpoolController(self)#, defDir, **kwargs)
        
        self.state = StateManager(self)
        
        self.state.registerHandler('ActiveCamera', self.GetActiveCameraName, self._SetCamera, True)
        self.state.registerHandler('Camera.IntegrationTime', self._GetActiveCameraIntegrationTime, self._SetActiveCameraIntegrationTime, True)
        self.state.registerHandler('Camera.ROI', self._GetActiveCameraROI, self._SetActiveCameraROI, True)
        self.state.registerHandler('Camera.Binning', self._GetActiveCameraBinning, self._SetActiveCameraBinning, True)
        
        self.actions = ActionManager(self)
        
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)
        
        #provision to set global metadata values in startup script
        self.mdh = MetaDataHandler.NestedClassMDHandler()
        
    def EnableJoystick(self, enable=True):
        if not self.joystick is None:
            self.joystick.Enable(enable)   
        
    def GetPos(self):
        res = {}
        
        axes = [k.split('.')[-1] for k in self.state.keys() if k.startswith('Positioning.')]
        
        for k in axes:
            res[k] = self.state['Positioning.%s' % k]
            
        return res
        
    def SetPos(self, **kwargs):
        state_updates = {'Positioning.%s' % k : v for k, v in kwargs.items()}
        self.state.setItems(state_updates)
        
    def centre_roi_on(self, x, y):
        """Convenience function to center the ROI on a given location"""
        dx, dy = self.get_roi_offset()
        
        self.SetPos(x=(x-dx), y=(y-dy))
            
    def GetPosRange(self):
        #Todo - fix to use positioning
        res = {}
        for k in self.positioning.keys():
            p, c, m = self.positioning[k]
            if m > 0: 
                res[k] = (p.GetMin(c)*m,p.GetMax(c)*m)
            else:
                res[k] = (p.GetMax(c)*m,p.GetMin(c)*m)

        return res
    
    @property
    @contextmanager
    def settingsDB(self):
        fstub = os.path.split(__file__)[0]
        dbfname = os.path.join(fstub, 'PYMESettings.db')
    
        settingsDB = sqlite3.connect(dbfname, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        #self.settingsDB.isolation_level = None
        yield(settingsDB)

        settingsDB.commit()
        settingsDB.close()
        

    def _OpenSettingsDB(self):
        with self.settingsDB as conn:
            tableNames = [a[0] for a in conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]
    
            if not 'CCDCalibration2' in tableNames:
                conn.execute("CREATE TABLE CCDCalibration2 (time timestamp, temperature integer, serial integer, nominalGains ndarray, trueGains ndarray)")
            if not 'VoxelSizes' in tableNames:
                conn.execute("CREATE TABLE VoxelSizes (ID INTEGER PRIMARY KEY, x REAL, y REAL, name TEXT)")
            if not 'VoxelSizeHistory2' in tableNames:
                conn.execute("CREATE TABLE VoxelSizeHistory2 (time timestamp, sizeID INTEGER, camSerial INTEGER)")
            if not 'StartupTimes' in tableNames:
                conn.execute("CREATE TABLE StartupTimes (component TEXT, time REAL)")
                conn.execute("INSERT INTO StartupTimes VALUES ('total', 5)")

    def GetPixelSize(self):
        """Get the (sample space) pixel size for the current camera
        
        Returns
        -------
        
        pixelsize : tuple
            the pixel size in the x and y axes, in um
        """
        with self.settingsDB as conn:
            currVoxelSizeID = conn.execute("SELECT sizeID FROM VoxelSizeHistory2 WHERE camSerial=? ORDER BY time DESC", (self.cam.GetSerialNumber(),)).fetchone()
            if not currVoxelSizeID is None:
                voxx, voxy = conn.execute("SELECT x,y FROM VoxelSizes WHERE ID=?", currVoxelSizeID).fetchone()
                
                return voxx*self.cam.GetHorizontalBin(), voxy*self.cam.GetVerticalBin()
            elif hasattr(self.cam, 'XVals'): # change to looking for attribute so that wrapped (e.g. multiview) cameras still work
                # read voxel size from directly from our simulated camera
                logger.info('Reading voxel size directly from simulated camera')
                vx_um = float(self.cam.XVals[1] - self.cam.XVals[0]) / 1.0e3
                vy_um = float(self.cam.YVals[1] - self.cam.YVals[0]) / 1.0e3
                return vx_um * self.cam.GetHorizontalBin(), vy_um * self.cam.GetVerticalBin()
                    

    def GenStartMetadata(self, mdh):
        """Collects the metadata we want to record at the start of a sequence
        
        Parameters
        ----------
        
        mdh : object derived from PYME.IO.MetaDataHandler.MDHandlerBase
            The metadata handler to which we should write our metadata         
        """
        try:
            voxx, voxy = self.GetPixelSize()
            mdh.setEntry('voxelsize.x', voxx)
            mdh.setEntry('voxelsize.y', voxy)
            mdh.setEntry('voxelsize.units', 'um')
        except TypeError:
            logger.error('No pixel size setting available for current camera - please configure the pixel size')
            pass

        for p in self.piezos:
            mdh.setEntry('Positioning.%s' % p[2].replace(' ', '_').replace('-', '_'), p[0].GetPos(p[1]))
            
        for k, v in self.GetPos().items():
            mdh.setEntry('Positioning.%s' % k.replace(' ', '_').replace('-', '_'), v)
        
        try:
            mdh['CameraOrientation.Rotate'] = self.cam.orientation['rotate']
            mdh['CameraOrientation.FlipX'] = self.cam.orientation['flipx']
            mdh['CameraOrientation.FlipY'] = self.cam.orientation['flipy']
        except AttributeError:
            pass
        
        if not self.microscope_name is None:
            mdh['MicroscopeName'] = self.microscope_name
            
        mdh.update(self.state)
        mdh.copyEntriesFrom(self.mdh)

    def AddVoxelSizeSetting(self, name, x, y):
        """ Adds a new voxel size setting.
        
        The reason for multiple settings is to easily support switching between 
        different cameras and/or optical configurations such as tube lenses, splitters,
        etc ...
        
        Parameters
        ----------
        
        name : string
            The name to use for the new setting
        x : float
            the pixelsize along the x axis in um
        y : float
            the pixelsize along the y axis in um
        """
        with self.settingsDB as conn:
            conn.execute("INSERT INTO VoxelSizes (name, x, y) VALUES (?, ?, ?)", (name, x, y))
        
        

    def SetVoxelSize(self, voxelsizename, camName=None):
        """Set the camera voxel size, from a pre-existing voxel size seting.
        
        Parameters
        ----------
        
        voxelsizename : string
            The name of the voxelsize setting to use. This should have been created
            using the **AddVoxelSize** function.
        camName : string
            The name of the camera to ascociate the setting with. If None, then
            the currently selected camera is used.
        """
        if camName is None:
            cam = self.cam
        else:
            cam = self.cameras[camName]

        with self.settingsDB as conn:
            voxelSizeID = conn.execute("SELECT ID FROM VoxelSizes WHERE name=?", (voxelsizename,)).fetchone()[0]
            conn.execute("INSERT INTO VoxelSizeHistory2 VALUES (?, ?, ?)", (datetime.datetime.now(), voxelSizeID, cam.GetSerialNumber()))
     


    def satCheck(self, source, **kwargs): # check for saturation
        """Check to see if the current frame is saturated and stop the camera/
        close the shutter if necessary
        
        TODO: could use a rewrite / new inspection.        
        """
        import wx
        
        if not 'shutterOpen' in dir(self.cam):
            return
        im = source.currentFrame
        IMax = im.max()

        if not self.cam.shutterOpen:
            self.cam.ADOffset = im.mean()
        elif (IMax >= self.cam.SaturationThreshold): #is saturated

            source.cam.StopAq()

            if self.lastFrameSaturated: #last frame was also saturated - our intervention obviously didn't work - close the shutter
                if 'SetShutter' in dir(source.cam):
                    source.cam.SetShutter(False)
                source.cam.StartExposure()
                self.saturatedMessage = 'Camera shutter has been closed'
                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                wx.MessageBox(self.saturatedMessage, "Saturation detected", wx.OK|wx.ICON_HAND)
                return

            #fracPixelsSat = (im > self.saturationThreshold).sum().astype('f')/im.size

            #try turning the e.m. gain off
            if 'SetEMGain' in dir(source.cam) and not source.cam.GetEMGain() == 0:
                self.oldEMGain = source.cam.GetEMGain()
                source.cam.SetEMGain(0)
                if self.oldEMGain  < 50: #poor chance of resolving by turning EMGain down alone
                    if 'SetShutter' in dir(source.cam):
                        source.cam.SetShutter(False)
                        self.saturatedMessage = 'Camera shutter closed'
                else:
                    self.saturatedMessage = 'EM Gain turned down'
                    
                source.cam.StartExposure()

                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                wx.MessageBox(self.saturatedMessage, "Saturation detected", wx.OK|wx.ICON_HAND)
                #return
            else:
                if 'SetShutter' in dir(source.cam):
                    source.cam.SetShutter(False)
                source.cam.StartExposure()
                self.saturatedMessage = 'Camera shutter closed'
                self.lastFrameSaturated = True
                self.cam.saturationIntervened = True
                wx.MessageBox(self.saturatedMessage, "Saturation detected", wx.OK|wx.ICON_HAND)
                #return

            self.lastFrameSaturated = True

        else:
            self.lastFrameSaturated = False


    def genStatus(self):
        """Generate a status message. TODO - move this to the GUI?"""
        stext = ''
        if self.cam.CamReady():
            self.cam.GetStatus()
            stext = 'CCD Temp: %d' % self.cam.GetCCDTemp()
        else:
            stext = '<Camera ERROR>'    
            
        if 'saturationIntervened' in dir(self.cam):
            if self.lastFrameSaturated:
                stext = stext + '    Camera Saturated!!'
            if self.cam.saturationIntervened:
                stext = stext + '    ' + self.saturatedMessage        
                
        #if 'step' in dir(self):
        #    stext = stext + '   Stepper: (XPos: %1.2f  YPos: %1.2f  ZPos: %1.2f)' % (self.step.GetPosX(), self.step.GetPosY(), self.step.GetPosZ())

        #stext = stext + '    Position:        
        
        if self.frameWrangler.isRunning():
            try:
                stext = stext + '    FPS = (%2.2f/%2.2f)' % (self.cam.GetFPS(),self.frameWrangler.getFPS())
            except (AttributeError, NotImplementedError):
                stext = stext + '    FPS = %2.2f' % self.frameWrangler.getFPS()

            if 'GetNumImsBuffered' in dir(self.cam):
                stext = stext + '    Buffer Level: %d of %d' % (self.cam.GetNumImsBuffered(), self.cam.GetBufferSize())
        
        for sic in self.StatusCallbacks:
            stext = stext + '    ' + sic()
        return stext

    @property
    def pa(self):
        """property to catch access of what was previously called the scope.frameWrangler (the PreviewAcquisator)"""
        warnings.warn(".pa is deprecated, please use .frameWrangler instead", DeprecationWarning)
        return self.frameWrangler
    
    def startFrameWrangler(self, event_loop=None):
        """Start the frame wrangler. Gets called during post-init phase of aquiremainframe.
        
        """
        #stop an old acquisition
        try:
            self.frameWrangler.stop()
        except AttributeError:
            pass
        
        self.frameWrangler = FrameWrangler(self.cam, event_loop=event_loop)
        self.frameWrangler.HardwareChecks.extend(self.hardwareChecks)
        self.frameWrangler.Prepare()

        if 'shutterOpen' in dir(self.cam):
            #self.pa.WantFrameGroupNotification.append(self.satCheck)
            #self.frameWrangler.onFrameGroup.connect(self.satCheck)
            pass
            
        self.frameWrangler.start()
        self.CleanupFunctions.append(self.frameWrangler.destroy)
        
        for cb in self.PACallbacks:
            cb()
            
    ##############################
    # The microscope object manages multiple cameras
    #
    # The following functions deal with selecting the active camera and
    # performing operations on the currently active camera
    # The integration time and ROI setting functions can be thought of as
    # proxying the underlying calls to the camera

    def _SetCamera(self, camName):
        """Set the currently used camera by name, selecting from the dictionary
        self.cameras. Calling code should use self.state['ActiveCamera']= camName
        instead.
        
        Parameters
        ----------
        
        camName : string
            The name of the camera to switch to
        
        """
        #try:
        #    self.frameWrangler.stop()
        #except AttributeError:
        #    pass

        #deactivate cameras
        for c in self.cameras.values():
            c.SetActive(False)
            c.SetShutter(False)


        
        self.cam = self.cameras[camName]
        if 'lightpath' in dir(self):
            self.lightpath.SetPort(self.cam.port)
        
        self.cam.SetActive(True)
        try:
            self.cam.SetShutter(self.camControls[camName].cbShutter.GetValue())
        except AttributeError:
            pass #for cameras which don't have a shutter

        # TODO this needs to move to the GUI
        from PYME.ui import manualFoldPanel as afp
        for k in self.cameras.keys():
            ctrl = self.camControls[k]
            if isinstance(ctrl, afp.foldingPane):
                ctrl.Hide()
            else:
                ctrl.GetParent().Hide()  # GetParent().UnPin()

        ctrl = self.camControls[camName]
        if isinstance(ctrl, afp.foldingPane):
            ctrl.Show()
            ctrl.GetParent().Layout()
        else:
            ctrl.GetParent().Show()
            ctrl.GetParent().GetParent().Layout()
        # End GUI TODO

        try:
            self.frameWrangler.cam = self.cam
            self.frameWrangler.Prepare()
            
            #self.frameWrangler.start()
        except AttributeError:
            pass
    
    def GetActiveCameraName(self):
        """Get the name / key of the currently active camera"""
        for name, cam in self.cameras.items():
            if cam is self.cam:
                return name
                
    def _SetActiveCameraIntegrationTime(self, integrationTime):
        """
        Sets the integration time for the active camera
        Parameters
        ----------
        integrationTime: float
            Units of seconds.


        Notes
        -----
        This is a state handler, use
        `scope.state['Camera.IntegrationTime'] = integrationTime`
        instead of calling directly
        """
        self.cam.SetIntegTime(integrationTime)
    
    def _GetActiveCameraIntegrationTime(self):
        """Gets the integration time for the active camera (in seconds)"""
        return self.cam.GetIntegTime()
        
    def _SetActiveCameraROI(self, ROI):
        """Sets the ROI for the active camera
        
        NB: This is a state handler, use 
        `scope.state['Camera.ROI'] = ROI` 
        instead of calling directly 
        
        Parameters
        ----------

        ROI : tuple / sequence
            The co-ordinates (in pixels) of the ROI, in the form (x0, y0, x1, y1)       
        
        """
        logger.debug('Setting camera ROI')
        self.cam.SetROI(*ROI)
        
    def _GetActiveCameraROI(self):
        """Gets the ROI for the active camera
        
        Parameters
        ----------

        ROI : tuple / sequence
            The co-ordinates (in pixels) of the ROI, in the form (x0, y0, x1, y1)       
        
        """
        
        return self.cam.GetROI()

    def _SetActiveCameraBinning(self, binning):
        """
        Sets Binning on the active camera
        
        Parameters
        ----------
        binning : tuple/sequence (binx, biny)
            the binning in x and y respectively
            
        """
        binx, biny = binning
        
        self.cam.SetHorizontalBin(binx)
        self.cam.SetVerticalBin(biny)

    def _GetActiveCameraBinning(self):
        """
        Sets Binning on the active camera

        Parameters
        ----------
        binning : tuple/sequence (binx, biny)
            the binning in x and y respectively

        """
        binx = self.cam.GetHorizontalBin()
        biny = self.cam.GetVerticalBin()
        
        return (binx, biny)
            
    def PanCamera(self, dx, dy):
        """Moves / pans the stage my a given offset, in pixels relative to the
        camera, correcting for any differences in rotation and mirroring between 
        camera and stage axes.
        
        TODO: fix to use state based position setting
        """
        vx, vy = self.GetPixelSize()
        
        p = self.GetPos()
        
        ox = p['x']
        oy = p['y']
        
        if 'orientation' in dir(self.cam):
            if self.cam.orientation['rotate']:
                dx, dy = dy, dx
                
            if self.cam.orientation['flipx']:
                dx *= -1
                
            if self.cam.orientation['flipy']:
                dy *= -1
        
        self.SetPos(x=(ox + dx*vx), y=(oy + dy*vy))


    def turnAllLasersOff(self):
        """Turn all attached lasers off.
        
        TODO - does this fit with the new state based paradigm?
        """
        import re
        for k in self.state.keys():
            if re.match(r'Lasers\.(?P<laser_name>.*)\.On', k):
                self.state[k] = False
        #for l in self.lasers:
        #    l.TurnOff()
                
    def initialize(self, init_script_name, locals={}):
        from PYME.Acquire import ExecTools
        
        # add ExecTools functions to namespace (for backwards compatibility - note that new scripts should have
        # `from PYME.Acquire.ExecTools import InitBG, joinBGInit, InitGUI, HWNotPresent` as their first line.
        from PYME.Acquire.ExecTools import InitBG, joinBGInit, InitGUI, HWNotPresent
        locals.update(InitBG=InitBG, joinBGInit=joinBGInit, InitGUI=InitGUI, HWNotPresent=HWNotPresent)
        
        locals.update(scope=self)
        ExecTools.setDefaultNamespace(locals, globals())
        self._init_thread = ExecTools.execFileBG(init_script_name, locals, globals(), then=self._post_init)
        
        return self._init_thread
        
    @property
    def initialized(self):
        """
        Checks if the initialisation thread has run (replaces check of initDone flag which needed to be manually set
        in initialisation script). Semantics are slightly different in that this returns true even if there is an error
        whist the old way of checking would get forever stuck at the splash screen.
        
        Returns
        -------
        
        True if the initialisation script has run, False otherwise

        """
        try:
            return not self._init_thread.is_alive()
        except AttributeError:
            # don't have and _init_thread yet
            return False
        
    def wait_for_init(self):
        self._init_thread.join()
        
    def _post_init(self):
        """
        Called after the init script runs to do stuff based on what hardware is present
        """
        
        if len(self.positioning) > 0:
            # we have registered piezos => we can do z-stacks
            from PYME.Acquire import stackSettings
            
            # FIXME - this looks like it will create a circular reference
            self.stackSettings = stackSettings.StackSettings(self)
                
    def register_piezo(self, piezo, axis_name, multiplier=1, needCamRestart=False, channel=0):
        """
        Register a piezo with the microscope object
        
        Parameters
        ----------
        piezo : `PYME.Acquire.Hardware.Piezos.base_piezo.PiezoBase` instance
            the piezo to register
        axis_name : string
            the axis name, e.g. 'x', 'y', 'z'
        multiplier : float, typically either 1 or -1
            what to multiply the positions by to match the directionality in the displayed image and make panning etc
            work.
        needCamRestart : bool
            whether to restart the camera after changing the position (mostly for simulation and fake piezos)

        Notes
        -----
        
        The 'x' and 'y' axes should be defined in terms of the axes of the principle camera (i.e. if a stage moves the
        image on the camera left and right it is the 'x' stage, regardless of the physical motion of the sample). Likewise,
        the multipliers should be set so that a positive movement of the x stage moves the image right, and a positive
        movement of the y stage moves the image down.

        """
        try:
            display_name = piezo.gui_description % axis_name
        except:
            display_name = 'Piezo %s' % axis_name

        self.piezos.append((piezo, channel, display_name))
        
        try:
            units_um = float(piezo.units_um)
        except:
            units_um = 1.0
        
        self.positioning[axis_name] = (piezo, channel, 1*multiplier*units_um)
        self.state.registerHandler('Positioning.%s' % axis_name, lambda: units_um*multiplier*piezo.GetPos(channel),
                                    lambda v: piezo.MoveTo(channel, v/(multiplier*units_um)), needCamRestart=needCamRestart)
        
        self.state.registerHandler('Positioning.%s_target' % axis_name,
                                       lambda: units_um*multiplier*piezo.GetTargetPos(channel)) #FIXME - target position does not belong in state ---
        
    def register_camera(self, cam, name, port='', rotate=False, flipx=False, flipy=False):
        """
        Register a camera with the microscope
        
        Parameters
        ----------
        cam     : The camera object
        name    : A human-readable name
        port    : Which sideport the camera is attached to (used to automatically select the right camera when changing
                  sideports & vice versa, but currently only works on the Nikon Ti stand)
        rotate  : Is this camera rotated with respect to the cardinal camera (used for multi-camera support)
        flipx   : Is this camera flipped in x w.r.t. the cardinal camera (used for multi-camera support)
        flipy   : Is this camera flipped in y w.r.t. the cardinal camera (used for multi-camera support)

        Notes
        -----
        The optional port, rotate, flipx, and flipy parameters all relate to multiple camera support and should be
        defined with respect to the principle camera (not stages etc which should be in the frame of the principle camera).
        The support for these parameters is currently pretty limited - they work when panning the sample using the mouse,
        but are generally ignored in other parts of the code.

        """
        cam.port = port
        cam.orientation = dict(rotate=rotate, flipx=flipx, flipy=flipy)
        
        self.cameras[name] = cam
        if self.cam is None:
            self.cam = cam
            cam.SetActive(True)
            
    def get_roi_offset(self):
        """
        Stage (positioning) coordinates are referenced to the (0,0) pixel of the primary camera. Return the offset to
        the centre of the current ROI. Used when, e.g. moving the stage to centre the current view on a given set of
        coordinates.
        
        Returns
        -------
        
        roi_offset_x, roi_offset_y : float
                                        offsets in um

        """
        from PYME.Acquire.Hardware.multiview import MultiviewWrapper
        from PYME.Acquire.Hardware.Camera import MultiviewCameraMixin
        
        x0, y0, _, _ = self.state['Camera.ROI']
    
        if isinstance(self.cam, (MultiviewWrapper, MultiviewCameraMixin)):
            # fix multiview crazyness
            if self.cam.multiview_enabled:
                #always use the 0th ROI for determining relative position, regardless of which ROIs are active
                x0, y0 = self.cam.view_origins[0]
        
            roi_offset_x = self.GetPixelSize()[0] * (x0 + 0.5 * self.cam.size_x)
            roi_offset_y = self.GetPixelSize()[1] * (y0 + 0.5 * self.cam.size_y)
        else:
            roi_offset_x = self.GetPixelSize()[0] * (x0 + 0.5 * self.cam.GetPicWidth())
            roi_offset_y = self.GetPixelSize()[1] * (y0 + 0.5 * self.cam.GetPicHeight())
            
        return roi_offset_x, roi_offset_y

    def __del__(self):
        pass
        #self.settingsDB.close()
