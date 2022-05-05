from PYME.recipes.traits import HasTraits, Float, File, BaseEnum, Enum, List, Instance, Str, Int, Bool
from PYME.misc.exceptions import UserError
from PYME.IO import image
import numpy as np


class PointSource(HasTraits):
    def refresh_choices(self):
        pass

    def points(self):
        x, y, z = self.getPoints()
        c = np.zeros_like(x)
        yield np.array([x,y,z,c], 'f').T


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

class WiglyFibreSource(PointSource):
    length = Float(50e3)
    persistLength=Float(1500)
    numFluors= Int(1000)
    centre = Bool(True)
    flatten = Bool(False)

    zScale = Float(1.0)

    def getPoints(self):
        from PYME.simulation import wormlike2
        wc = wormlike2.wiglyFibre(self.length, self.persistLength, self.length/self.numFluors)

        if self.centre:
            wc.xp -= wc.xp.mean()
            wc.yp -= wc.yp.mean()
            wc.zp -= wc.zp.mean()


        if self.flatten:
            wc.zp *= 0
        else:
            wc.zp *= self.zScale
        
        return wc.xp, wc.yp, wc.zp

class SHNucleusSource(PointSource):
    axis_scaling = List(Float, [6000., 4000., 2000.])
    axis_sigmas = List(Float, [1000, 1000, 1000])
    #fixme - modes
    point_spacing = Float(500.)

    def getPoints(self):
        from PYME.simulation import locify
        from PYME.Analysis.points.spherical_harmonics import SHShell
        
        axis_scales = np.array(self.axis_scaling) + np.array(self.axis_sigmas)*np.random.normal(size=3)

        
        #axis_scales = np.array([1,1,1])
        sf = 0.5*np.sqrt(1./np.pi) # scipy sph harmonic normalisation

        #a_s = axis_scales/axis_scales.max()

        modes = [(0,0)]
        amplitudes = [1]

        for n in range(1,5):
            for m in range(-n, n+1):
                modes.append((m, n))
                amplitudes.append(np.random.randn(1)*(0.1*n**-1.5))

        #TODO - tip - we currently do rotation only
        theta = np.random.uniform(0, 2*np.pi)
        prin_axes = ((np.sin(theta), np.cos(theta), 0), (np.cos(theta), -np.sin(theta), 0), (0,0,1))

        shell = SHShell(principle_axes=prin_axes, axis_scaling=sf/axis_scales, modes=modes, coefficients=amplitudes)

        print('axis_scales:', axis_scales)
        r_max = 1.5*np.max(axis_scales)

        #pts = locify.points_from_sdf(shell.radial_distance_to_shell, r_max=r_max, dx_min=0.1*self.point_spacing)

        #print('pts.shape:', pts.shape)
        #return pts

        
        zenith = []
        azimuth = []
        for z in np.linspace(-np.pi/2, np.pi/2, 100):
            az = np.linspace(0, 2*np.pi, int(np.abs(np.cos(z))*100))
            azimuth.append(az)
            zenith.append(z*np.ones_like(az))

        azimuth = np.hstack(azimuth)
        zenith = np.hstack(zenith)

        x, y, z = shell.get_fitted_shell(azimuth, zenith)

        print(x.min(), x.max())
        
        return x.ravel(), y.ravel(), z.ravel()


        
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
