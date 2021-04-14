import json
import numpy as np
import scipy

from PYME.simulation import wormlike2
from . import fluor
from . import rend_im

import logging
logger = logging.getLogger(__name__)

from PYME.recipes.traits import HasTraits, Float, Dict, Bool, List


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

class SimController(object):
    """ Non-GUI part of simulation control"""
    
    def __init__(self, scope=None, states=['Caged', 'On', 'Blinked', 'Bleached'],
                 stateTypes=[fluor.FROM_ONLY, fluor.ALL_TRANS, fluor.ALL_TRANS, fluor.TO_ONLY], transistion_tensor=None,
                 excitation_crossections=(1., 100.),
                 activeState=fluor.states.active, n_chans=1, splitter_info=([0, -200, 300., 500.], [0, 1, 1, 0]),
                 spectral_signatures=[[1, 0.3], [.7, .7], [0.2, 1]]):
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
        self.scope = scope
        
        if (transistion_tensor is None): #use defaults
            transistion_tensor = fluor.createSimpleTransitionMatrix()
        
        self.transition_tensor = transistion_tensor
        self.excitation_crossections = excitation_crossections
        
        self.n_chans = n_chans
        self.z_offsets, self.spec_chans = splitter_info
        self.spectralSignatures = np.array(spectral_signatures)
        
        self.points = []
        self._empirical_hist = None
        
    @property
    def splitter_info(self):
        return self.z_offsets[:self.n_chans], self.spec_chans[:self.n_chans]
    
    def gen_fluors_wormlike(self, kbp=50e3, persistLength=1500, numFluors=1000, flatten=False, z_scale=1.0, num_colours=1, wrap=True):
        import numpy as np
        
        #wc = wormlike2.fibre30nm(kbp, 10*kbp/numFluors)
        wc = wormlike2.wiglyFibre(kbp, persistLength, kbp / numFluors)
        
        XVals = self.scope.cam.XVals
        YVals = self.scope.cam.YVals
        
        x_pixels = len(XVals)
        y_pixels = len(YVals)
        
        x_chan_pixels = int(x_pixels / self.n_chans)
        x_chan_size = XVals[x_chan_pixels - 1] - XVals[0]
        
        y_chan_size = YVals[-1] - YVals[0]
        
        wc.xp = wc.xp - wc.xp.mean() + x_chan_size / 2
        wc.yp = wc.yp - wc.yp.mean() + y_chan_size / 2
        
        if wrap:
            wc.xp = np.mod(wc.xp, x_chan_size) + XVals[0]
            wc.yp = np.mod(wc.yp, y_chan_size) + YVals[0]
        
        if flatten:
            wc.zp *= 0
        else:
            wc.zp -= wc.zp.mean()
            wc.zp *= z_scale
        
        self.points = []
        
        num_colours = min(num_colours, len(self.spectralSignatures))
        
        for i in range(len(wc.xp)):
            if num_colours > 1:
                self.points.append((wc.xp[i], wc.yp[i], wc.zp[i], float(i / ((len(wc.xp) + 1) / num_colours))))
            else:
                self.points.append((wc.xp[i], wc.yp[i], wc.zp[i], 0))
        
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
    
    def generate_fluorophores_theoretical(self):
        if (len(self.points) == 0):
            raise RuntimeError('No points defined')
        
        points_a = scipy.array(self.points).astype('f')
        x = points_a[:, 0]
        y = points_a[:, 1]
        z = points_a[:, 2]
        
        if points_a.shape[1] == 4: #4th entry is index into spectrum table
            c = points_a[:, 3].astype('i')
            spec_sig = scipy.ones((len(x), 2))
            spec_sig[:, 0] = self.spectralSignatures[c, 0]
            spec_sig[:, 1] = self.spectralSignatures[c, 1]
            
            fluors = fluor.specFluors(x, y, z, self.transition_tensor, self.excitation_crossections,
                                      activeState=self.activeState, spectralSig=spec_sig)
        else:
            fluors = fluor.fluors(x, y, z, self.transition_tensor, self.excitation_crossections,
                                  activeState=self.activeState)
        
        self.scope.cam.setSplitterInfo(*self.splitter_info)
        self.scope.cam.setFluors(fluors)
    
    def load_empirical_histogram(self, filename):
        from . import EmpiricalHist
        
        with open(filename, 'r') as f:
            data = json.load(f)
        self._empirical_hist = EmpiricalHist(**data[data.keys().pop()])
    
    def generate_fluorophores_empirical(self):
        points_a = scipy.array(self.points).astype('f')
        x = points_a[:, 0]
        y = points_a[:, 1]
        z = points_a[:, 2]
        
        fluors = fluor.EmpiricalHistFluors(x, y, z,
                                           histogram=self._empirical_hist,
                                           activeState=self.activeState)
        
        self.scope.cam.setSplitterInfo(*self.splitter_info)
        self.scope.cam.setFluors(fluors)
    
    def generate_fluorophores(self, mode='theoretical'):
        if mode == 'emperical':
            self.generate_fluorophores_empirical()
        else:
            self.generate_fluorophores_theoretical()
            
    
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
        
