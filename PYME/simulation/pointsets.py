from PYME.recipes.traits import HasTraits, Float, File, BaseEnum, Enum, List, Instance, Str
from PYME.misc.exceptions import UserError
from PYME.IO import image


class PointSource(HasTraits):
    def refresh_choices(self):
        pass


class WormlikeSource(PointSource):
    kbp = Float(200)
    steplength = Float(1.0)
    lengthPerKbp = Float(10.0)
    persistLength = Float(150.0)
    
    #name = Str('Wormlike Chain')
    
    def getPoints(self):
        from PYME.simulation import wormlike2
        wc = wormlike2.wormlikeChain(self.kbp, self.steplength, self.lengthPerKbp, self.persistLength)
        
        return wc.xp, wc.yp, wc.zp

    def genMetaData(self, mdh):
        mdh['GeneratedPoints.Source.Type'] = 'Wormlike'
        mdh['GeneratedPoints.Source.Kbp'] = self.kbp
        mdh['GeneratedPoints.Source.StepLength'] = self.steplength
        mdh['GeneratedPoints.Source.LengthPerKbp'] = self.lengthPerKbp
        mdh['GeneratedPoints.Source.PersistLength'] = self.persistLength
        
class FileSource(PointSource):
    file = File(filter=['*.npy', '*.csv'],exists=True)
    
    #name = Str('Points File')
    
    def getPoints(self):
        if self.file.endswith('.csv'):
            # csv based storage format
            import pandas as pd
            df = pd.read_csv(self.file)
            return (df['x'].values,df['y'].values,df['z'].values)
        else:
            import numpy as np
            return np.load(self.file)

    def genMetaData(self, mdh):
        mdh['GeneratedPoints.Source.Type'] = 'File'
        mdh['GeneratedPoints.Source.FileName'] = self.file


class WRDictEnum(BaseEnum):
    def __init__(self, wrdict, *args, **metadata):
        self.wrdict = wrdict
        #self.values        = tuple( values )
        #self.fast_validate = ( 5, self.values )
        self.name = ''
        super(BaseEnum, self).__init__(None, **metadata)
    
    @property
    def values(self):
        return list(self.wrdict.keys())
    
    #def info ( self ):
    #    return ' or '.join( [ repr( x ) for x in self.values ] )
    
    def create_editor(self):
        from traitsui.api import EnumEditor
        #print dict(self.wrdict.items())
        
        ed = EnumEditor(values=self,
                        cols=self.cols or 3,
                        evaluate=self.evaluate,
                        mode=self.mode or 'radio')
        
        return ed


class ImageSource(PointSource):
    image = WRDictEnum(image.openImages)
    points_per_pixel = Float(0.1)
    #name = Str('Density Image')
    #foo = Enum([1,2,3,4])
    
    helpInfo = {
        'points_per_pixel': '''
Select average number of points (dye molecules or docking sites) per pixel in image regions where the density values are 1.
The number is a floating point fraction, e.g. 0.1, and shouldn't exceed 1. It is used for Monte-Carlo rejection of positions
and larger values (>~0.2) will result in images which have visible pixel-grid structure because the Monte-Carlo sampling
is no longer a good approximation to random sampling over the grid. If this is a problem for your application / you can't
get high enough density without a high acceptance fraction, use an up-sampled source image with a smaller pixel size.
''',
        'image': '''
Select an image from the list of open images.
Note that you need to open or generate the source image you want to use so that this
list is not empty. The image will be normalised for the purpose of the simulation,
with its maximum set to 1. It describes the density of markers in the simulated sample,
where values of 1 have a density of markers as given by the `points per pixel` parameter, i.e.
in the Monte-Carlo sampling the acceptance probability = image*points_per_pixel. Smaller
density values therefore give rise to proportionally fewer markers per pixel.
''',
    }
    
    def helpStr(self, name):
        def cleanupHelpStr(str):
            return str.strip().replace('\n', ' ').replace('\r', '')
        
        return cleanupHelpStr(self.helpInfo[name])
    
    def default_traits_view(self):
        from traitsui.api import View, Item
        
        traits_view = View(Item('points_per_pixel', help=self.helpStr('points_per_pixel'),
                                tooltip='mean number of marker points per pixel'),
                           Item('image', help=self.helpStr('image'),
                                tooltip='select the marker density image from the list of open images'),
                           buttons=['OK', 'Help'])
        
        return traits_view
    
    def getPoints(self):
        from PYME.simulation import locify
        # print((self.image))  # if still needed should be replaced by a logging statement
        
        try:
            im = image.openImages[self.image]
        except KeyError:
            # no image of that name:
            # If uncaught this will pop up in the error dialog from 'Computation in progress', so shouldn't need
            # an explicit dialog / explicit handing. TODO - do we need an error subclass - e.g. UserError or ParameterError
            # which the error dialog treats differently to more generic errors so as to make it clear that it's something
            # the user has done wrong rather than a bug???
            raise UserError('No open image found with name: "%s", please set "image" property of ImageSource to a valid image name\nThis must be an image which is already open.\n\n' % self.image)
        
        #import numpy as np
        d = im.data[:, :, 0, 0].astype('f')
        
        #normalise the image
        d = d / d.max()
        
        return locify.locify(d, pixelSize=im.pixelSize, pointsPerPixel=self.points_per_pixel)
    
    def get_bounds(self):
        return image.openImages[self.image].imgBounds
    
    def refresh_choices(self):
        ed = self.trait('image').editor
        
        if ed:
            try:
                ed._values_changed()
            except TypeError:
                # TODO - why can _values_changed be None??
                # is there a better way to handle/avoid this?
                pass
            #super( HasTraits, self ).configure_traits(*args, **kwargs)

#    traits_view = View( Item('points_per_pixel'),
#                        Item('image'),
##                        Item( 'image',
##                              label='Image',
##                              editor =
##                                  EnumEditor(values={'foo':0, 'bar' : 1}),#image.openImages),
##                              ),
#                        buttons = ['OK'])

    def genMetaData(self, mdh):
        mdh['GeneratedPoints.Source.Type'] = 'Image'
        mdh['GeneratedPoints.Source.PointsPerPixel'] = self.points_per_pixel
        mdh['GeneratedPoints.Source.Image'] = self.image
