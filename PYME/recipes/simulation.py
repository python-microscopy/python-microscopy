from .base import register_module, ModuleBase,Filter
from .traits import Input, Output, Float, Enum, CStr, Bool, Int, Dict, DictStrFloat, DictStrBool, on_trait_change

import numpy as np


@register_module('RandomPoints')
class RandomPoints(ModuleBase):
    output = Output('points')

    def execute(self, namespace):
        from PYME.IO import tabular
        namespace[self.output]= tabular.RandomSource(1000, 1000, 1000)

@register_module('TheoreticalPSF')
class TheoreticalPSF(ModuleBase):
    """
    Generates theoretical PSFs

    """
    output = Output('psf')

    dx_nm = Float(70.)
    dz_nm = Float(200.)
    size_z = Int(100)
    wavelength = Float(700.)
    n = Float(1.51, desc='Refractive index')
    ns = Float(1.51, desc='Sample refractive index. NB - only used when calculating apodization function')
    NA = Float(1.4)
    vectorial = Bool(False)
    apodization = Enum('sine', )
    zernike_modes = Dict(Int, Float, desc='Dictionary of zernike mode amplitudes (e.g. {4:1, 8:2} would give a PSF with astigmatism and spherical abberation.)')
    psf_type = Enum('widefield', 'confocal', '2D STED')
    excitation_wavelength = Float(647.)
    pinhole_radius_nm = Float(200.)
    sted_wavelength = Float(775.)
    sted_saturation = Float(10., desc='I/I_sat')

    def execute(self, namespace):
        from PYME.Analysis.PSFGen import fourierHNA
        from PYME.IO import image, MetaDataHandler
        from scipy import fftn, ifftn

        z_ = self.dz_nm*np.arange(-self.size_z/2, self.size_z/2)
        
        # detection PSF is always the same
        det_psf = fourierHNA.GenZernikeDPSF(z_, dx = self.dx_nm,
                                       zernikeCoeffs = self.zernike_modes, 
                                       lamb=self.wavelength, 
                                       n=self.n, NA = self.NA, ns=self.ns, beadsize=0, 
                                       vect=self.vectorial, apodization=self.apodization)
        
        if self.psf_type == 'widefield':
            psf = det_psf
            
        elif self.psf_type in ('confocal', '2D STED'):
            illumination_psf = fourierHNA.GenZernikeDPSF(z_, dx = self.dx_nm,
                                       zernikeCoeffs = self.zernike_modes, 
                                       lamb=self.excitation_wavelength, 
                                       n=self.n, NA = self.NA, ns=self.ns, beadsize=0, 
                                       vect=self.vectorial, apodization=self.apodization)

            pinhole = np.zeros_like(det_psf)
            x, y = self.dx_nm*np.mgrid[0.:float(det_psf.shape[0]), 0.:float(det_psf.shape[0])]
            x = x - x[-1]/2
            y = y - y[-1]/2
            r2 = x*x + y*y

            pinhole[:,:,int(det_psf.shaope[2]/2)] = r2 < self.pinhole_radius_nm**2

            psf = illumination_psf*(ifftn(fftn(det_psf)*fftn(pinhole)))

        if self.psf_type == '2D STED':
            donut = fourierHNA.GenZernikeDonutPSF(z_, dx = self.dx_nm,
                                       zernikeCoeffs = self.zernike_modes, 
                                       lamb=self.excitation_wavelength, 
                                       n=self.n, NA = self.NA, ns=self.ns, 
                                       apodization=self.apodization, vectorial=(1,-1j),
                                       spiral_amp=1.0)

            psf = psf*np.exp(-self.sted_saturation*donut)

            
        psf_im = image.ImageStack(psf)
        psf_im.mdh['voxelsize.x'] = self.dx_nm*1e-3
        psf_im.mdh['voxelsize.y'] = self.dx_nm*1e-3
        psf_im.mdh['voxelsize.z'] = self.dz_nm*1e-3

        self._params_to_metadata(psf_im.mdh)

        namespace[self.output] = psf_im