#!/usr/bin/python
##################
# pointSetGeneration.py
#
# Copyright David Baddeley, 2011
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


from PYME.recipes.traits import HasTraits, Float, File, BaseEnum, Enum, List, Instance, Str
    
from PYME.simulation.pointsets import PointSource, WormlikeSource, ImageSource, FileSource


class Generator(HasTraits):
    meanIntensity = Float(500)
    meanDuration = Float(3)
    backgroundIntensity = Float(300)
    meanEventNumber = Float(2)
    scaleFactor = Float(2) # Note: we don't expose the scale factor in the view
    meanTime= Float(2000)
    mode = Enum(['STORM','PAINT'])

    sources = List([WormlikeSource(), ImageSource(), FileSource()])

    source = Instance(PointSource)

    helpInfo = {
        'source': '''
Select the type of point source to generate points.
A wormlike source, an image based source and a file based source are supported.
''',
        'meanIntensity': '''
This parameter specifies the mean number of photons in an event.
Typically values are in the range from 100 to several 10000.
''',
        'meanDuration': '''
The mean duration of events which is specified in units of frames.
''',
        'meanTime': '''
This parameter, the mean time of the series in frame units, is the average time at which you expect to get events
(i.e. the value of np.mean(pipeline['t']) for the simulated set of events). Since STORM mode draws event times from
an exponential distribution, PAINT from a uniform one, it can also be related to the resulting apparent series duration,
which may be more familar to experimentally minded users. For PAINT mode it works out as half the duration of the series,
for STORM simulation mode the relationship is a little more complex,
you can work it out from the decay time of an exponential distribution.
''',
        'meanEventNumber': '''
This parameter specifies the mean number of times an event occurs at a single marker location.
''',
        'backgroundIntensity' : '''
The background intensity per pixel in units of photons, typically in the range from a few tens to hundreds of photons.
''',
        'mode': '''
With the simulation mode you can choose between STORM or PAINT mode.
This parameter effects how event rate changes with time (it stays constant in PAINT mode).
''',
        'scaleFactor': '''
This parameter is related to the size of the PSF for purposes of thresholding
(used in combination with the background intensity, which is per pixel).
There should be no need to modify this from the default and it is accordingly not exposed in the view.
 
''',
    }

    def helpStr(self, name):
        def cleanupHelpStr(str):
            return str.strip().replace('\n', ' ').replace('\r', '')
        
        return cleanupHelpStr(self.helpInfo[name])
        
    def default_traits_view( self ):
        from traitsui.api import View, Item, InstanceEditor

        traits_view = View( Item( 'source',
                            label= 'Point source',
                            editor =
                            InstanceEditor(name = 'sources',
                                editable = True),
                            help=self.helpStr('source'),
                                ),
                        Item('_'),
                        Item('meanIntensity',tooltip='mean photon number of events',
                             help=self.helpStr('meanIntensity')),
                        Item('meanDuration',tooltip='mean duration of events in units of frames',
                             help=self.helpStr('meanDuration')),
                        Item('meanEventNumber',tooltip='mean number of times events occurs at a single marker location',
                             help=self.helpStr('meanEventNumber')),
                        Item('meanTime',tooltip='mean time of the series, roughly related to series duration, in frame units',
                             help=self.helpStr('meanTime')),
                        Item('_'),
                        Item('backgroundIntensity',tooltip='background intensity in units of photons',
                             help=self.helpStr('backgroundIntensity')),
                        Item('_'),
                        Item('mode',tooltip='STORM or PAINT mode, effects how event rate changes with time',
                             help=self.helpStr('mode')),  
                        
                        buttons = ['OK', 'Help'])
        
        return traits_view

    def __init__(self, visFr = None):
        self.visFr = visFr
        self.source = self.sources[0]

        if visFr:
            visFr.AddMenuItem('Extras>Synthetic Data', "Configure", self.OnConfigure)
            visFr.AddMenuItem('Extras>Synthetic Data', 'Generate fluorophore positions and events', self.OnGenPoints)
            visFr.AddMenuItem('Extras>Synthetic Data', 'Generate events', self.OnGenEvents)
            visFr.AddMenuItem('Extras>Synthetic Data', 'Save fluorophore positions', self.OnSavePoints)



    def OnConfigure(self, event):
        self.source.refresh_choices()
        self.edit_traits()

    def OnGenPoints(self, event):
        self.xp, self.yp, self.zp = self.source.getPoints()
        self.OnGenEvents(None)


    def OnSavePoints(self, event):
        import wx
        try:
            x = self.xp
        except AttributeError:
            wx.MessageBox('No points! Generate fluorophore positions first', 'Warning', style=wx.OK)
            return

        # using a pandas based CSV IO here, there may be preferences for a more direct implementation
        # I (CS) like it for the high-level interface
        import pandas as pd
        df = pd.DataFrame({'x': self.xp,
                           'y': self.yp,
                           'z': self.zp})
        filename = wx.FileSelector("Save coordinates as",
                                   wildcard='Comma separated values (*.csv)|*.csv',
                                   flags=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        df.to_csv(filename,index=False)

        
    def OnGenEvents(self, event):
        from PYME.simulation import locify
        from PYME.IO import tabular
        from PYME.IO.image import ImageBounds
        # import pylab
        import matplotlib.pyplot as plt
        
        #wc = wormlike2.WormlikeChain(100)
        
        pipeline = self.visFr.pipeline
        pipeline.filename='Simulation'

        plt.figure()
        plt.plot(self.xp, self.yp, 'x') #, lw=2)
        if isinstance(self.source, WormlikeSource):
            plt.plot(self.xp, self.yp, lw=2)

        if self.mode == 'STORM':
            res = locify.eventify(self.xp, self.yp, self.meanIntensity, self.meanDuration, self.backgroundIntensity,
                                  self.meanEventNumber, self.scaleFactor, self.meanTime, z=self.zp)
        else:
            res = locify.eventify2(self.xp, self.yp, self.meanIntensity, self.meanDuration, self.backgroundIntensity,
                                  self.meanEventNumber, self.scaleFactor, self.meanTime, z=self.zp)
        
        plt.plot(res['fitResults']['x0'],res['fitResults']['y0'], '+')

        ds = tabular.MappingFilter(tabular.FitResultsSource(res))

        try:
            # some data sources (current ImageSource) have image bound info. Use this if available
            # this could fail on either an AttributeError (if the data source doesn't implement bounds
            # or another error if something fails in get_bounds(). Only catch the AttributeError, as we have
            # should not be handling other errors here.
            pipeline.imageBounds = self.source.get_bounds()
        except AttributeError:
            pipeline.imageBounds = ImageBounds.estimateFromSource(ds)

        from PYME.IO.MetaDataHandler import NestedClassMDHandler
        ds.mdh = NestedClassMDHandler()
        ds.mdh['Camera.ElectronsPerCount'] = 1
        ds.mdh['Camera.TrueEMGain'] = 1
        ds.mdh['Camera.CycleTime'] = 1
        ds.mdh['voxelsize.x'] = .110
        # some info about the parameters
        ds.mdh['GeneratedPoints.MeanIntensity'] = self.meanIntensity
        ds.mdh['GeneratedPoints.MeanDuration'] = self.meanDuration
        ds.mdh['GeneratedPoints.MeanEventNumber'] = self.meanEventNumber
        ds.mdh['GeneratedPoints.BackgroundIntensity'] = self.backgroundIntensity
        ds.mdh['GeneratedPoints.ScaleFactor'] = self.scaleFactor
        ds.mdh['GeneratedPoints.MeanTime'] = self.meanTime
        ds.mdh['GeneratedPoints.Mode'] = self.mode
        # the source info
        self.source.genMetaData(ds.mdh)
            
        pipeline.addDataSource('Generated Points', ds)
        pipeline.selectDataSource('Generated Points')

        try:
            pipeline.filterKeys.pop('sig')
        except:
            pass

        pipeline.Rebuild()
        if len(self.visFr.layers) < 1:
            self.visFr.add_pointcloud_layer() #TODO - move this logic so that layer added automatically when datasource is added?
        #self.visFr.CreateFoldPanel()
        self.visFr.SetFit()



def Plug(visFr):
    """Plugs this module into the gui"""
    visFr.pt_generator = Generator(visFr)



