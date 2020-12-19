# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:32:07 2016

@author: david
"""
import wx

from PYME.localization import MetaDataEdit as mde
import PYME.localization.FitFactories
from  PYME.ui import manualFoldPanel
from PYME.IO import MetaDataHandler
import logging
from PYME.contrib import dispatch

logger = logging.getLogger(__name__)

class AnalysisSettingsPanel(wx.Panel):
    def __init__(self, parent, analysisSettings, mdhChangedSignal=None):
        wx.Panel.__init__(self, parent, -1)
        
        self.analysisSettings = analysisSettings        
        self.analysisMDH = analysisSettings.analysisMDH
        self.mdhChangedSignal = mdhChangedSignal
        
        self._inChange = False
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.fitFactories = PYME.localization.FitFactories.resFitFactories

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(self, -1, 'Type:'), 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.cFitType = wx.Choice(self, -1, choices = ['{:<35} - {:} '.format(f, PYME.localization.FitFactories.useFor[f]) for f in self.fitFactories], size=(110, -1))
        #self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
        self.cFitType.Bind(wx.EVT_CHOICE, self.OnFitModuleChanged)
        
        hsizer.Add(self.cFitType, 1, wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0, wx.EXPAND|wx.ALL, 4)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.cbLogSettings = wx.CheckBox(self, -1, 'Save analysis settings to metadata')
        self.cbLogSettings.SetValue(False)
        self.cbLogSettings.Bind(wx.EVT_CHECKBOX, lambda e : self.analysisSettings.SetPropagate(e.IsChecked()))
        hsizer.Add(self.cbLogSettings, 1, wx.ALIGN_CENTER_VERTICAL, 0)
        vsizer.Add(hsizer, 0, wx.EXPAND|wx.ALL, 4)
        
        self.SetSizerAndFit(vsizer)
        
        self.update()
        
    def OnFitModuleChanged(self, event):
        self._inChange = True
        self.analysisMDH['Analysis.FitModule'] = self.fitFactories[self.cFitType.GetSelection()]
        if self.mdhChangedSignal:
            self.mdhChangedSignal.send_robust(self, mdh=self.analysisMDH)
        
        self._inChange = False
            
    def update(self, **kwargs):
        if self._inChange:
            return
        try:
            self.cFitType.SetSelection(self.fitFactories.index(self.analysisMDH['Analysis.FitModule']))
        except:
            self.analysisMDH['Analysis.FitModule'] = 'LatGaussFitFR'
            self.cFitType.SetSelection(self.fitFactories.index('LatGaussFitFR'))
            
        
        

class AnalysisDetailsPanel(wx.Panel):    
    DEFAULT_PARAMS = [mde.FloatParam('Analysis.DetectionThreshold', 'Thresh:', 1.0),
                      mde.IntParam('Analysis.DebounceRadius', 'Debounce rad:', 4),
                      mde.IntParam('Analysis.StartAt', 'Start at:', default=30),
                      mde.RangeParam('Analysis.BGRange', 'Background:', default=(-30,0)),
                      mde.BoolParam('Analysis.subtractBackground', 'Subtract background in fit', default=True),
                      mde.BoolFloatParam('Analysis.PCTBackground' , 'Use percentile for background', default=False, helpText='', ondefault=0.25, offvalue=0),
                      #mde.FilenameParam('Camera.VarianceMapID', 'Variance Map:', prompt='Please select variance map to use ...', wildcard='TIFF Files|*.tif', filename=''),
                      #mde.FilenameParam('Camera.DarkMapID', 'Dark Map:', prompt='Please select dark map to use ...', wildcard='TIFF Files|*.tif', filename=''),
                      #mde.FilenameParam('Camera.FlatfieldMapID', 'Flatfield Map:', prompt='Please select flatfield map to use ...', wildcard='TIFF Files|*.tif', filename=''),
                      mde.BoolParam('Analysis.TrackFiducials', 'Track Fiducials', default=False),
                      mde.FloatParam('Analysis.FiducialThreshold', 'Fiducial Threshold', default=1.8),
    ]
    
    def __init__(self, parent, analysisSettings, mdhChangedSignal=None):
        wx.Panel.__init__(self, parent, -1)

        self.analysisSettings = analysisSettings
        self.analysisMDH = analysisSettings.analysisMDH
        self.mdhChangedSignal = mdhChangedSignal
        
        mdhChangedSignal.connect(self.OnMDChanged)
        
        self._analysisModule = ''
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        vsizer_std = wx.BoxSizer(wx.VERTICAL)
        self._populateStdOptionsPanel(self, vsizer_std)
        vsizer.Add(vsizer_std, 0, wx.EXPAND, 0)
        
        self.customOptionsSizer = wx.BoxSizer(wx.VERTICAL)
        self._populateCustomAnalysisPanel(self, self.customOptionsSizer)
        vsizer.Add(self.customOptionsSizer, 0, wx.EXPAND, 0)
        
        self.SetSizerAndFit(vsizer)
    

    def _populateStdOptionsPanel(self, pan, vsizer):
        for param in self.DEFAULT_PARAMS:
            pg = param.createGUI(pan, self.analysisMDH, syncMdh=True, 
                                 mdhChangedSignal = self.mdhChangedSignal)
            vsizer.Add(pg, 0,wx.BOTTOM|wx.EXPAND, 5)
        vsizer.Fit(pan)
               
        
    def _populateCustomAnalysisPanel(self, pan, vsizer):
        try:
            #fitMod = self.fitFactories[self.cFitType.GetSelection()]
            self._analysisModule = self.analysisMDH['Analysis.FitModule']
            fm = __import__('PYME.localization.FitFactories.' + self._analysisModule, fromlist=['PYME', 'localization', 'FitFactories'])
            
            #vsizer = wx.BoxSizer(wx.VERTICAL)
            for param in fm.PARAMETERS:
                pg = param.createGUI(pan, self.analysisMDH, syncMdh=True, 
                                 mdhChangedSignal = self.mdhChangedSignal)
                vsizer.Add(pg, 0,wx.BOTTOM|wx.EXPAND, 5)
            vsizer.Fit(pan)
            self.Layout()
            self.SetMinSize([200, self.GetBestSize()[1]])
            self.GetParent().Layout()
            print('custom analysis settings populated')
                
        except (KeyError, AttributeError):
            pass
        
    def OnMDChanged(self, event=None, sender=None, signal=None, mdh=None):
        if not self._analysisModule == self.analysisMDH['Analysis.FitModule']:
            self.customOptionsSizer.Clear(True)
            self._populateCustomAnalysisPanel(self, self.customOptionsSizer)
            # FIXME - can't figure out a better way to redo the vertical sizer for longer menus
            self.GetParent().fold1(self)
            self.GetParent().fold1(self)


class LocalizationSettingsPanel(manualFoldPanel.foldingPane):
    def __init__(self, wx_parent, localization_settings, mdh_changed_signal=None,
                 chained_analysis_page=None):
        from PYME.ui.autoFoldPanel import collapsingPane
        manualFoldPanel.foldingPane.__init__(self, wx_parent, caption='Localization Analysis')

        self.localization_settings = localization_settings 
        self.localization_mdh = localization_settings.analysisMDH
        self.mdh_changed_signal = mdh_changed_signal
        self.chained_analysis_page = chained_analysis_page
        
        clp = collapsingPane(self, caption='settings ...')
        clp.AddNewElement(AnalysisSettingsPanel(clp, 
                                                self.localization_settings,
                                                self.mdh_changed_signal))
        clp.AddNewElement(AnalysisDetailsPanel(clp, self.localization_settings,
                                               self.mdh_changed_signal))
        self.AddNewElement(clp)

        # add box to propagate rule to rule chain
        add_rule_panel = wx.Panel(self, -1)
        v_sizer = wx.BoxSizer(wx.VERTICAL)
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.b_add_rule = wx.Button(add_rule_panel, -1, 
                                      'Add Localization to Chained Analysis')
        self.b_add_rule.Bind(wx.EVT_BUTTON, self.OnAddLocalizationRule)
        h_sizer.Add(self.b_add_rule)
        v_sizer.Add(h_sizer, 0, wx.EXPAND|wx.TOP, 0)
        add_rule_panel.SetSizerAndFit(v_sizer)
        self.AddNewElement(add_rule_panel)

    
    def OnAddLocalizationRule(self, wx_event=None):
        from PYME.cluster.rules import LocalisationRuleFactory as LocalizaitonRuleFactory
        from PYME.Acquire.ui.rules import get_rule_tile
        from PYME.IO.MetaDataHandler import DictMDHandler
        if self.chained_analysis_page is None:
            logger.error('chained_analysis_page attribute unset')
            return

        mdh = DictMDHandler(self.localization_mdh)
        loc_rule = get_rule_tile(LocalizaitonRuleFactory)(analysisMetadata=mdh)
        self.chained_analysis_page.add_tile(loc_rule)
    

class AnalysisSettings(object):
    def __init__(self):
        self.analysisMDH = MetaDataHandler.NestedClassMDHandler()
        
        self.onMetadataChanged = dispatch.Signal()
        
        self.propagateToAcquisisitonMetadata = False
        
        MetaDataHandler.provideStartMetadata.append(self.genStartMetadata)
        
    def SetPropagate(self, prop=True):
        self.propagateToAcquisisitonMetadata = prop
        
    def genStartMetadata(self, mdh):
        if self.propagateToAcquisisitonMetadata:
            mdh.copyEntriesFrom(self.analysisMDH)
            

def Plug(scope, MainFrame):
    scope.analysisSettings = AnalysisSettings()
    
    analSettingsPan = AnalysisSettingsPanel(MainFrame, scope.analysisSettings, scope.analysisSettings.onMetadataChanged)
    analDetailsPan = AnalysisDetailsPanel(MainFrame, scope.analysisSettings, scope.analysisSettings.onMetadataChanged)
    
    MainFrame.anPanels.append((analSettingsPan, 'Analysis module', True))
    MainFrame.anPanels.append((analDetailsPan, 'Analysis details', True))
