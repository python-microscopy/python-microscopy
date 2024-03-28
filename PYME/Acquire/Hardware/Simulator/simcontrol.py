import json
import numpy as np
import scipy

from PYME.simulation import wormlike2
from . import fluor
from . import rend_im

import logging
logger = logging.getLogger(__name__)

from PYME.recipes.traits import HasTraits, Float, Dict, Bool, List, Tuple, Int, Instance
from PYME.simulation import pointsets


class PSFSettings(HasTraits):
    wavelength_nm = Float(700.)
    NA = Float(1.47)
    vectorial = Bool(False)
    zernike_modes = Dict()
    zernike_modes_lower = Dict()
    phases = List([0, .5, 1, 1.5])
    four_pi = Bool(False)
    
    def default_traits_view(self):
        from traitsui.api import View, Item
        #from PYME.ui.custom_traits_editors import CBEditor
        
        return View(Item(name='wavelength_nm'),
                    Item(name='NA'),
                    Item(name='vectorial'),
                    Item(name='four_pi', label='4Pi'),
                    Item(name='zernike_modes'),
                    Item(name='zernike_modes_lower', visible_when='four_pi==True'),
                    Item(name='phases', visible_when='four_pi==True', label='phases/pi'),
                    resizable=True,
                    buttons=['OK'])


class Group(HasTraits):
    generators = List(Instance(HasTraits))

    def points(self):
        for g in self.generators:
            for pts in g.points():
                yield pts

class AssignChannel(HasTraits):
    channel = Int(0)
    generator = Instance(HasTraits)

    def points(self):
        for pts in self.generator.points():
            pts[:,3] = self.channel
            yield pts
class Shift(HasTraits):
    dx = Float(0)
    dy = Float(0)

    generator = Instance(HasTraits)

    def points(self):
        for pts in self.generator.points():
            pts[:,0] += self.dx
            pts[:,1] += self.dy
            yield pts

class RandomShift(HasTraits):
    magnitude = Float(1000)

    generator = Instance(HasTraits)

    def points(self):
        dx, dy = np.random.uniform(-self.magnitude, self.magnitude, 2)
        for pts in self.generator.points():
            pts[:,0] += dx
            pts[:,1] += dy
            yield pts
class RandomDistribution(HasTraits):
    n_instances = Int(1)
    region_size = Float(5000)
    generator = Instance(HasTraits)
    # force one of the points to be at the origin (dirty hack to make sure there is a structure present in the simulator at startup)
    force_at_origin = Bool(False) 

    def points(self):
        xp = self.region_size*np.random.uniform(-1, 1, self.n_instances)
        yp = self.region_size*np.random.uniform(-1, 1, self.n_instances)

        if self.force_at_origin:
            xp[0] = 0.0
            yp[0] = 0.0


        for xi, yi in zip(xp, yp):
            for p in self.generator.points():
                p1 = np.copy(p)
                p1[:,0] += xi
                p1[:,1] += yi

                yield p1



class SimController(object):
    """ Non-GUI part of simulation control"""
    
    def __init__(self, scope=None, states=['Caged', 'On', 'Blinked', 'Bleached'],
                 stateTypes=[fluor.FROM_ONLY, fluor.ALL_TRANS, fluor.ALL_TRANS, fluor.TO_ONLY], transistion_tensor=None,
                 excitation_crossections=(1., 100.),
                 activeState=fluor.states.active, n_chans=1, splitter_info=([0, -200, 300., 500.], [0, 1, 1, 0]),
                 spectral_signatures=[[1, 0.3], [.7, .7], [0.2, 1]],
                 point_gen=None):
        """

        Parameters
        ----------
        scope : PYME.Acquire.microscope.Microscope instance
        states : list
            The possible fluorophore states
        stateTypes : list
            list of types of transitions permitted from the states
        transistion_tensor: ndarray
            an n_states x n_states x 3 tensor with the transition probabilities per unit time and intensity for spontaneous,
            activation laser mediated, and readout laser mediated transitions between states.
        excitation_crossections: 2-tuple
            excitation crossections for the dye with respect to the photo-activation and readout lasers respectively.
            nominally in units of photons/mW/s (although this might be unreliable)
        activeState : integer
            the index of the state considered to be active (i.e. resulting in fluorescent emission)
        n_chans : int
            The total number of detection channels resulting from, e.g. ratiometric, biplane, 4PiSMS, or any combination
             of the above.
        splitter_info : tuple of zOffsets, specChans
            Information on splitting. zOffsets and specChans are both arrays of length n_chans which specify the z offset
            and ratiometric colour channel assignment of each each detection channel. We currently only support a 2-way colour splitting
            so each entry in specChans should either be 0 or 1, but you can have multiple detection channels which have the
            assigned to the same ratiometric
        spectral_signatures: list
            A list of ratiometric division ratios for each fluorophore species present. These specify how much of the
            fluorophore signal goes into each of the two specChans.

        """
        
        self.states = states
        self.stateTypes = stateTypes
        self.activeState = activeState
        self.scope = scope # type: PYME.Acquire.microscope.Microscope

        if scope:
            scope.StatusCallbacks.append(self.simulation_status)
        
        if (transistion_tensor is None): #use defaults
            transistion_tensor = fluor.createSimpleTransitionMatrix()
        
        self.transition_tensor = transistion_tensor
        self.excitation_crossections = excitation_crossections
        
        self.n_chans = n_chans
        self.z_offsets, self.spec_chans = splitter_info
        self.spectralSignatures = np.array(spectral_signatures)
        
        self.points = []
        self._empirical_hist = None

        self.point_gen = point_gen
        
    @property
    def splitter_info(self):
        return self.z_offsets[:self.n_chans], self.spec_chans[:self.n_chans]

    
    def gen_fluors_wormlike(self, kbp=50e3, persistLength=1500, numFluors=1000, flatten=False, z_scale=1.0, num_colours=1, wrap=True):
        import numpy as np
        
        #wc = wormlike2.fibre30nm(kbp, 10*kbp/numFluors)
        #wc = wormlike2.wiglyFibre(kbp, persistLength, kbp / numFluors)
        xp, yp, zp = pointsets.WiglyFibreSource(length=kbp, persistLength=persistLength, numFluors=numFluors, flatten=flatten, zScale=z_scale).getPoints()
        
        XVals = self.scope.cam.XVals
        YVals = self.scope.cam.YVals
        
        x_pixels = len(XVals)
        x_chan_pixels = int(x_pixels / self.n_chans)
        x_chan_size = XVals[x_chan_pixels - 1] - XVals[0]
        y_chan_size = YVals[-1] - YVals[0]
        
        #shift to centre of ROI
        xp += x_chan_size / 2
        yp += y_chan_size / 2
        
        if wrap:
            xp = np.mod(xp, x_chan_size) + XVals[0]
            yp = np.mod(yp, y_chan_size) + YVals[0]
        
        num_colours = min(num_colours, len(self.spectralSignatures))
        c = np.linspace(0, num_colours, len(xp)).astype('i')

        self.points = np.array([xp,yp,zp,c], 'f').T
        
        self.scope.cam.setFluors(None)
        self.generate_fluorophores()
    
    def load_fluors(self, filename):
        if filename.endswith('.npy'):
            self.points = list(np.load(filename))
        else:
            self.points = np.loadtxt(filename)
        
        self.scope.cam.setFluors(None)
        self.generate_fluorophores()
    
    def save_points(self, filename):
        np.save(filename, scipy.array(self.points))
    
    def set_psf_model(self, psf_settings):
        z_modes = {int(k): float(v) for k, v in psf_settings.zernike_modes.items()}
        
        if psf_settings.four_pi:
            z_modes_lower = {int(k): float(v) for k, v in psf_settings.zernike_modes_lower.items()}
            phases = [np.pi * float(p) for p in psf_settings.phases]
            
            #print z_modes, z_modes_lower, phases
            rend_im.genTheoreticalModel4Pi(rend_im.mdh, phases=phases, zernikes=[z_modes, z_modes_lower],
                                           lamb=psf_settings.wavelength_nm,
                                           NA=psf_settings.NA, vectorial=psf_settings.vectorial)
            
            label = 'PSF: 4Pi %s [%1.2f NA @ %d nm, zerns=%s]' % ('vectorial' if psf_settings.vectorial else 'scalar',
                                                                  psf_settings.NA, psf_settings.wavelength_nm, z_modes)
        else:
            logger.info('Setting PSF with zernike modes: %s' % z_modes)
            rend_im.genTheoreticalModel(rend_im.mdh, zernikes=z_modes, lamb=psf_settings.wavelength_nm,
                                        NA=psf_settings.NA, vectorial=psf_settings.vectorial)
            
            label = 'PSF: Widefield %s [%1.2f NA @ %d nm, zerns=%s]' % (
            'vectorial' if psf_settings.vectorial else 'scalar',
            psf_settings.NA, psf_settings.wavelength_nm, z_modes)
        
        return label
    
    def set_psf_from_file(self, filename):
        rend_im.setModel(filename, rend_im.mdh)
        return 'PSF: Experimental [%s]' % filename
    
    def get_psf(self):
        return rend_im.get_psf()
    
    def save_psf(self, filename):
        self.get_psf().Save(filename)

    def fluorophores_from_points_theoretical(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        if points.shape[1] == 4: #4th entry is index into spectrum table
            c = points[:, 3].astype('i')
            spec_sig = np.ones((len(x), 2))
            spec_sig[:, 0] = self.spectralSignatures[c, 0]
            spec_sig[:, 1] = self.spectralSignatures[c, 1]
            
            fluors = fluor.SpectralFluorophores(x, y, z, self.transition_tensor, self.excitation_crossections,
                                      activeState=self.activeState, spectralSig=spec_sig)
        else:
            fluors = fluor.Fluorophores(x, y, z, self.transition_tensor, self.excitation_crossections,
                                  activeState=self.activeState)
        
        return fluors
    
    def load_empirical_histogram(self, filename):
        from . import EmpiricalHist
        
        with open(filename, 'r') as f:
            data = json.load(f)
        self._empirical_hist = EmpiricalHist(**data[data.keys().pop()])
    
    def fluorophores_from_points_emperical(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        if points.shape[1] == 4: #4th entry is index into spectrum table
            c = points[:, 3].astype('i')
            spec_sig = np.ones((len(x), 2))
            spec_sig[:, 0] = self.spectralSignatures[c, 0]
            spec_sig[:, 1] = self.spectralSignatures[c, 1]
            
            fluors = fluor.EmpiricalHistFluors(x, y, z,
                                           histogram=self._empirical_hist,
                                           activeState=self.activeState, spectralSig=spec_sig)
        else:
            fluors = fluor.EmpiricalHistFluors(x, y, z,
                                           histogram=self._empirical_hist,
                                           activeState=self.activeState)
        
        return fluors
    
    def generate_fluorophores(self, mode='theoretical'):
        if mode == 'emperical':
            gen_fcn = self.fluorophores_from_points_emperical
        else:
            gen_fcn = self.fluorophores_from_points_theoretical
            
        if self.point_gen:
            objs = [gen_fcn(pts) for pts in self.point_gen.points()]
        else:
            if (len(self.points) == 0):
                raise RuntimeError('No points defined')
            objs = [gen_fcn(np.array(self.points).astype('f')),]
        
        self.scope.cam.setSplitterInfo(*self.splitter_info)
        self.scope.cam.set_objects(objs)
            
    
    def change_num_channels(self, n_chans):
        self.n_chans = n_chans
        try:
            self.scope.frameWrangler.stop()
            self.scope.cam.SetSensorDimensions(n_chans * len(self.scope.cam.YVals), len(self.scope.cam.YVals),
                                               self.scope.cam.pixel_size_nm)
            self.scope.frameWrangler.Prepare()
            self.scope.frameWrangler.start()
        except AttributeError:
            logger.exception('Error setting new camera dimensions')
            pass

    def simulation_status(self):
        if self.scope.cam._objects:
            if len(self.scope.cam._objects) > 1:
                #fixme for multiple objects
                return 'Multiple objects defined'
            fl = self.scope.cam._objects[0] 
        elif self.scope.cam.fluors is None:
            return 'No fluorophores defined'
        else:
            fl = self.scope.cam.fluors

        cts = np.zeros((len(self.states)))
        for i in range(len(cts)):
            cts[i] = int((fl.fl['state'] == i).sum())
        
        status = '/'.join(self.states) + ' = ' + '/'.join(['%d' % c for c in cts])

        return status


        
        
