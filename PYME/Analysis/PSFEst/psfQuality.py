#!/usr/bin/python

##################
# psfQuality.py
#
# Copyright David Baddeley, 2012
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

"""These functions assess a psf for quality and return a scalar 'merit' which indicates how
good or bad the psf is wrt the particular measure.

Lower values are better, with the following breakpoints:
    
    merit < 1           good            OK to use
    1 < merit < 2       marginal        Might work in some situations but undesirable - use with caution 
    merit > 3           unacceptable    Almost garuanteed not to work
"""

import numpy as np

localisation_tests = {}
deconvolution_tests = {}

test_names = set()

#define decorators
def loc_test(fcnName):
    def _ltest(fcn):
        test_names.add(fcnName)
        localisation_tests[fcnName] = fcn
        return fcn
    return _ltest
    
def dec_test(fcnName):
    def _ltest(fcn):
        test_names.add(fcnName)
        deconvolution_tests[fcnName] = fcn
        return fcn
    return _ltest
    
def glob_test(fcnName):
    def _ltest(fcn):
        test_names.add(fcnName)
        localisation_tests[fcnName] = fcn
        deconvolution_tests[fcnName] = fcn
        return fcn
    return _ltest
    
@loc_test('Voxelsize x')
@dec_test('Voxelsize x')
def voxelsize_x(image, psft):
    """The x pixel size should be sufficiently small to Nyquist the PSF. Making the
    pixels too large looses information whilst making them too small is empty 
    magnification and is incurs extra noise and computational cost. A high NA oil 
    immersion objective is assumed in these calculations"""
    vsx = image.voxelsize_nm.x
    merit = (np.abs(vsx - 70.)/20)**2    
    return vsx, merit

@loc_test('Size x')
#@dec_test('Size x')
def size_x(image, psft):
    """The extracted image should be large enough to capture the PSF, with a fair bit of wriggle room on either side.
    The default of 61 pixels does this in most situations. Smaller PSFs might work, but are are not guaranteed to - use
    with caution.
    """
    sx = image.data.shape[0]
    merit = 2.5*(sx < 61)
    return sx, merit
    
@loc_test('Voxelsize z')
def voxelsize_z_loc(image, psft):
    """3D localisation microscopy relies on capturing subtle variations in pixel shape.
    Although these should nominally be captured in a Nyquist sampled PSF, it is 
    wise to oversample in z (typically using 50 nm spacing) in order to reduce 
    the sensitivity of psf shape to the interpolation algortihms used in the 
    fitting. Oversampling also increases the effective SNR of the PSF"""
    vsz = image.voxelsize_nm.z
    merit = np.abs(vsz - 50.)/50 + 1.0*(vsz > 50)
    return vsz, merit
    
@dec_test('Voxelsize z')
def voxelsize_z_dec(image, psft):
    """z spacing in psfs for deconvolution purposes should match the data spacing,
    which should be at Nyquist for the expected frequency content of the PSF. 
    If abberated, PSFs will contain frequencies beyond the simple FWHM/2.35 
    calculalation typically used for voxel size selection. In most cases a
    z-spacing of 200 nm will be appropriate for a high NA oil immersion objective,
    although this might sometimes want to be reduced."""
    vsz = image.voxelsize_nm.z
    merit = 2*(1 + (170./(370.-vsz))**4 - 2*(170./(370-vsz))**2)
    return vsz, merit
    
@loc_test('Depth [um]')
def depth_loc(image, psft):
    """A psf used for 3D localisation should be sufficiently deep to capture the
    regions of the PSF which retain sufficient intensity to allow reasonable 
    detection and localisation. In practice, a total range of 2-3um should be 
    sufficient. Due to the implementation of the object finding algorithm, very
    large axial extents are actually a disadvantage as they will interfere with 
    reliable event detection."""
    vsz = image.voxelsize_nm.z
    depth = vsz*image.data.shape[2]
    merit = np.abs(depth - 3.)
    return depth, merit
    
@dec_test('Depth [um]')
def depth_dec(image, psft):
    """Deconvolution needs a PSF which is at least 5-6 times the axial resolution
    of the data set and preferably as large as the stack to be deconvolved. 
    In practice this means that the PSF should be at least 5 um deep."""
    vsz = image.mdh['voxelsize.z']
    depth = vsz*image.data.shape[2]
    merit = np.maximum((5.- depth) + 1, 0)
    return depth, merit
    
@loc_test('Background')
@dec_test('Background')
def background(image, psft):
    """This test uses the fact that the PSF should not extend to at edge of the
    image to judge background levels by looking at the mean values of the pixels 
    along one edge. It assumes the PSF is normalised to a peak intensity of 1
    (which it should be if extracted using the normal tools)."""
    bg = image.data[0,:,:].mean()
    merit = bg/.003    
    return bg, merit
    
@loc_test('Noise')
@dec_test('Noise')
def noise(image, psft):
    """This test looks at a region to the side of the in-focus plane to assess
    noise levels. Noise can be improved by using a longer integration time when
    acquiring the bead images or by averaging more beads."""
    n = image.data[:,:,image.data.shape[2]/2][:5,:5].std()
    merit = n/.0005    
    return n, merit
    
@loc_test('Positivity')
@dec_test('Positivity')
def positivity(image, psft):
    """The PSF should not contain any negative values"""
    n = image.data[:,:,:].min()
    merit = (n < 0)*3    
    return n, merit
    
@loc_test('3D CRB')
#@dec_test('voxelsize x')
def crb3d(image, psft):
    """The Cramar-Rao bound describes the ammount of position information
    encoded in a PSF. This test computes the average 3D CRB (the vector norm of the 
    3 individual axis CRBs) over a 1 um depth range and compares it to that obtained
    from a simulated astigmatic PSF with 500nm lobe separation. The PSF is classed 
    as good if the 3D CRB is less than twice that of the simulated PSF."""
    vsz = image.mdh['voxelsize.z']*1e3
    
    dz = 500./vsz
    
    zf = image.data.shape[2]/2

    crb_3D = np.sqrt(psft.crb.sum(1))[int(zf - dz):int(zf + dz +1)].mean()
    crb_3D_as = np.sqrt(psft.crb_as.sum(1))[int(zf - dz):int(zf + dz +1)].mean()
    
    print((crb_3D, crb_3D_as))
    
    ratio = crb_3D/crb_3D_as
    
    merit = max(0.5*ratio, 0)    
    return ratio, merit
    
#convert from set to list so ordering is preserved
test_names = list(test_names)

def runTests(image, psft):
#    report = ''
#    report += 'Test\t\tLocalisation\t\tDeconvolution\n\n'
#    for name in test_names:
#        line = '%s\t' % name
#        try:
#            line += '%s, %s\t' % localisation_tests[name](image, psft)
#        except KeyError:
#            line += 'N/A\t'
#            
#        try:
#            line += '%s, %s\n' % deconvolution_tests[name](image, psft)
#        except KeyError:
#            line += 'N/A\n'
#            
#        report += line
#        
#    return report

    loc_res = {}
    
    for k, v in localisation_tests.items():
        loc_res[k] = v(image, psft)
        
    dec_res = {}
    for k, v in deconvolution_tests.items():
        dec_res[k] = v(image, psft)
        
    return loc_res, dec_res
    
    
def colour(merit):
    merit = np.maximum(np.minimum(merit, 3), 0)
    r = 1.0*(merit > 1) 
    g = (1. - (0.5*np.abs(merit - 1)) - 0.3*(merit > 1))*(merit < 2)
    b = 0*merit
    
    return np.dstack((r, g, b)).squeeze()