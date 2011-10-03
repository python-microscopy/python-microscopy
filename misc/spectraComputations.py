#!/usr/bin/python
##################
# spectraComputations.py
#
# Copyright David Baddeley, 2011
# d.baddeley@auckland.ac.nz
# 
# This file may NOT be distributed without express permision from David Baddeley
#
##################
import numpy as np
from scipy.interpolate import interp1d

class spectrum(object):
    def __init__(self, data):
        self.data = data

    @property
    def wavelengths(self):
        return self.data['lambda']

    @property
    def magnitude(self):
        return self.data['T']

    def plot(self):
        import pylab
        pylab.plot(self.wavelengths, self.magnitude)
        pylab.xlabel('Wavelength [nm]')

    def _getAlignedMagnitude(self, other):
        if isinstance(other, np.ndarray) and other.dtype == self.data.dtype:
            other = spectrum(other)

        if isinstance(other, spectrum):
            intp = interp1d(other.wavelengths, other.magnitude, bounds_error=False, fill_value=0)

            #print other.wavelengths.shape, other.magnitude.shape, self.wavelengths.shape

            dxs = self.wavelengths[1] - self.wavelengths[0]
            dxo = other.wavelengths[1] - other.wavelengths[0]

            return intp(self.wavelengths) #*dxs/dxo
        else:
            return other

    def __add__(self, other):
        res = spectrum(self.data.copy())
        res.magnitude[:] = self.magnitude + self._getAlignedMagnitude(other)

        return res

    def __sub__(self, other):
        res = spectrum(self.data.copy())
        res.magnitude[:] = self.magnitude - self._getAlignedMagnitude(other)

        return res

    def __mul__(self, other):
        res = spectrum(self.data.copy())
        res.magnitude[:] = self.magnitude * self._getAlignedMagnitude(other)

        return res

    def __div__(self, other):
        res = spectrum(self.data.copy())
        res.magnitude[:] = self.magnitude / self._getAlignedMagnitude(other)

        return res

    def __truediv__(self, other):
        return self.__div__(self, other)


    def __radd__(self, other):
        return self.__add__(self, other)

    def __rsub__(self, other):
        res = spectrum(self.data.copy())
        res.magnitude[:] = self._getAlignedMagnitude(other) - self.magnitude

        return res

    def __rmul__(self, other):
        return self.__mul__(self, other)

    def __rdiv__(self, other):
        res = spectrum(self.data.copy())
        res.magnitude[:] = self._getAlignedMagnitude(other) / self.magnitude

        return res

    def __rtruediv__(self, other):
        return self.__rdiv__(self, other)
    

    def __neg__(self, other):
        res = spectrum(self.data.copy())
        res.magnitude[:] = -self.magnitude

        return res

    def __abs__(self, other):
        res = spectrum(self.data.copy())
        res.magnitude[:] = np.abs(self.magnitude)

        return res

    def sum(self):
        return self.magnitude.sum()

    def norm(self):
        return self/self.sum()

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == 0:
            return self.wavelengths
        elif key == 1:
            return self.magnitude
        else:
            raise IndexError



