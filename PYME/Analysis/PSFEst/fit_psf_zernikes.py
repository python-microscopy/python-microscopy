"""Fits zernike polynomials to a PSF image (as an alternative to Gerchberg-Saxton pupil estimation)

works by using the PSF simulation functions to generate as PSF image with a given set of zernike 
coefficients and comparing to the measured PSF image.
"""

import numpy as np
from PYME.Analysis.PSFGen import fourierHNA



class ZernikePSFFitter(object):
    def __init__(self, psf, voxelsize_nm, wavelength=700., n=1.51, NA=1.4, **kwargs):
        self.psf = psf
        self.voxelsize_nm = voxelsize_nm
        self.wavelength = wavelength
        self.n = n
        self.NA = NA

    def get_zernike_psf(self, zernike_coeffs=None):
        """
        Generate a PSF image with a given set of zernike coefficients

        Parameters
        ----------
        zernike_coeffs : ndarray
            array of zernike coefficients

        Returns
        -------
        psf : ndarray
            simulated PSF image
        """

        if zernike_coeffs is None:
            zernike_coeffs = self.zernike_coefficients

        zs = (np.arange(self.psf.shape[2]) - self.psf.shape[2]/2)*self.voxelsize_nm.z
        
        # pupil might be slightly undersampled at default PSF size,
        # so we use a larger pupil to avoid out-of focus grid artifacts
        # TODO - optimise the size here - doubling the pupil size
        # may well be overkill (and slow)
        X = np.arange(-64, 65)*self.voxelsize_nm.x
        Y = np.arange(-64, 65)*self.voxelsize_nm.y

        ps = fourierHNA.GenZernikePSF(zs, zernike_coeffs, X=X, Y=Y, dx=self.voxelsize_nm.x, lamb=self.wavelength, n=self.n, NA=self.NA, output_shape=self.psf.shape)
        
        # TODO - improve normalisation of PSF - maybe build into residual calculation
        return ps/ps.max()
    
    def zernike_residuals(self, zernike_coeffs):
        """
        Calculate the residuals between a measured PSF and a simulated PSF with a given set of zernike coefficients

        Parameters
        ----------
        zernike_coeffs : ndarray
            array of zernike coefficients

        Returns
        -------
        residuals : ndarray
            array of residuals between the measured and simulated PSFs
        """

        sim_psf = self.get_zernike_psf(zernike_coeffs)
        residuals = sim_psf - self.psf

        # TODO - change noise model???
        return np.sum(residuals**2)

    def refine_zernike(self, zernike_coeffs, zernike_order, delta=2.0):
        """
        Refine a single zernike coefficient, assuming a quadratic dependence on the residuals
        """
        coeffs = np.copy(zernike_coeffs)
        
        res0 = self.zernike_residuals(coeffs)

        coeffs[zernike_order] += delta
        res1 = self.zernike_residuals(coeffs)

        coeffs[zernike_order] -= 2*delta
        res2 = self.zernike_residuals(coeffs)

        # fit a quadratic to the residuals
        a = (res1 + res2 - 2*res0)/(2*delta**2)
        b = (res1 - res2)/(2*delta)
        c = res0

        return (-b/(2*a))
    
    def refine_zernikes(self, zernike_coeffs, delta):
        """
        Refine all zernike coefficients
        """
        refined_coeffs = np.copy(zernike_coeffs)
        for i in range(1, len(zernike_coeffs)): # skip the piston term
            d_i = self.refine_zernike(refined_coeffs, i, delta) # progressive update
            #d_i = self.refine_zernike(zernike_coeffs, i, delta) # update all at once (after all have been calculated) - this doesn't seem to be as good
            refined_coeffs[i] += np.clip(d_i, -delta, delta) # limit the step size to the range we are searching
        
        return refined_coeffs
    
    def approximate_zernikes(self, num_zernike_orders, num_iterations=5, starting_zernikes={}, plot=True, **kwargs):
        """
        Approximate the zernike coefficients of a PSF image using a least squares fit
        """
        zernikes = np.zeros(num_zernike_orders)
        for k,v in starting_zernikes.items():
            zernikes[k] = v

        delta = 2.0

        self._residuals = [] 
        self._deltas = []

        for i in range(num_iterations):
            z0 = zernikes
            zernikes = self.refine_zernikes(zernikes, delta)
            self._residuals.append(self.zernike_residuals(zernikes))
            self._deltas.append(delta)
            print(self._residuals[-1])

            # if we made a large move on any of the coefficients, hold step size, otherwise reduce it.
            max_change = np.max(np.abs(zernikes - z0))
            delta = np.max([delta/ 2.0, max_change])
        
        self.zernike_coefficients = zernikes

        if plot:
            self.plot_results()
        
        return zernikes
    

    def get_misfit_map(self, zernike_coeffs):
        """
        Get a map of the residuals between the measured and simulated PSFs
        as a 5x5 tiled image of z planes
        """
        sim_psf = self.get_zernike_psf(zernike_coeffs)
        res =  sim_psf - self.psf

        step = res.shape[2]//25
        out = np.zeros([5*self.psf.shape[0], 5*self.psf.shape[1]])

        for i in range(5):
            for j in range(5):
                out[i*self.psf.shape[0]:(i+1)*self.psf.shape[0], j*self.psf.shape[1]:(j+1)*self.psf.shape[1]] = res[:,:,(i*5+j)*step]

        return out
    
    def plot_results(self):
        """
        Plot the results of the zernike fitting
        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(100*self.get_misfit_map(self.zernike_coefficients), cmap=plt.cm.RdBu_r, clim=(-30, 30))
        plt.colorbar()
        plt.title('Fit residuals (% of max)')
        plt.xticks([])
        plt.yticks([])

        plt.figure()
        plt.subplot(311)
        plt.plot(self._residuals)
        plt.ylabel('Squared error')
        plt.xlabel('Iteration')
        plt.subplot(312)
        plt.plot(self._deltas)
        plt.ylabel('Step size')
        plt.xlabel('Iteration')

        plt.subplot(313)
        b = plt.bar(np.arange(len(self.zernike_coefficients)), self.zernike_coefficients)
        plt.bar_label(b, fmt='%.2f')
        plt.ylabel('Zernike coefficient')
        plt.xlabel('Zernike order')
        plt.axhline(0, color='k', linestyle='--')

        plt.show()







    


