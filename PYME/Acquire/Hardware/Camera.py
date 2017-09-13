#!/usr/bin/python

###############
# Camera.py
#
# Created: 12 September 2017
#
# Based on: AndorNeo.py
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
################

import ctypes

from PYME.IO import MetaDataHandler


class Camera(object):

    # Acquisition modes
    MODE_SINGLE_SHOT = 0
    MODE_CONTINUOUS = 1

    def __init__(self, *args, **kwargs):
        """
        Create a camera object. This gets called from the PYMEAcquire init
        script, which is custom for any given microscope and can take
        whatever arguments are needed for a given camera.

        .. note:: The one stipulation is that the Camera should register itself
                  as providing metadata.

        Parameters
        ----------
        args :
            Optional arguments, usually instantiated in inherited camera.
        kwargs :
            Optional dictionary of arguments, usually instantiated in
            inherited camera.

        Returns
        ----------
        Camera
            A camera object.
        """

        self.active = True  # Should the camera write its metadata?

        # Register as a provider of metadata (record camera settings)
        # this is important so that the camera settings get recorded
        MetaDataHandler.provideStartMetadata.append(
            self.generate_starting_metadata
            )

    def generate_starting_metadata(self, mdh):
        """
        Create Camera metadata. This ensures the Camera's settings get
        recorded.

        Parameters
        ----------
        mdh : MetaDataHandler
            MetaDataHandler object for Camera.

        Returns
        ----------
        None
        """

        if self.active:
            self.get_status()

            # Set Camera object metadata here with calls to mdh.setEntry

            # Personal identification
            mdh.setEntry('Camera.Name', 'Andor Neo')
            mdh.setEntry('Camera.Model', self.get_model())
            mdh.setEntry('Camera.SerialNumber', self.get_serial_number())

            # Time
            mdh.setEntry('Camera.IntegrationTime', self.get_integration_time())
            mdh.setEntry('Camera.CycleTime', self.get_integration_time())

            # Gain
            mdh.setEntry('Camera.EMGain', self.get_em_gain())
            mdh.setEntry('Camera.TrueEMGain', self.get_em_gain(true_gain=True))

            # Noise
            mdh.setEntry('Camera.ReadNoise', self.get_read_noise())
            mdh.setEntry('Camera.NoiseFactor', self.get_noise_factor())

            # QE
            mdh.setEntry('Camera.ElectronsPerCount',
                         self.get_electrons_per_count())

            # Temp
            mdh.setEntry('Camera.StartCCDTemp', self.get_temperature())

            # FOV
            mdh.setEntry('Camera.ROIPosX', self.get_roi_x1())
            mdh.setEntry('Camera.ROIPosY', self.get_roi_y1())
            mdh.setEntry('Camera.ROIWidth',
                         self.get_roi_x2() - self.get_roi_x1())
            mdh.setEntry('Camera.ROIHeight',
                         self.get_roi_y2() - self.get_roi_y1())

    def get_model(self):
        """
        Get the model name of the hardware represented by Camera object.

        Returns
        ----------
        str
            Hardware model name of Camera object
        """
        pass

    def get_serial_number(self):
        """
        Get the serial number of the hardware represented by Camera object.

        Returns
        ----------
        str
            Hardware serial number of Camera object
        """
        pass

    def get_integration_time(self):
        pass

    def get_read_noise(self):
        pass

    def get_noise_factor(self):
        pass

    def get_electrons_per_count(self):
        pass

    def get_temperature(self):
        pass

    def get_roi_x1(self):
        pass

    def get_roi_x2(self):
        pass

    def get_roi_y1(self):
        pass

    def get_roi_x2(self):
        pass

    def get_em_gain(self, true_gain=False):
        """
        Return electromagnetic gain of Camera object.

        Parameters
        ----------
        true_gain : bool
            Return true EM gain (adjusted).

        Returns
        ----------
        float
            Camera object gain or (adjusted) true gain.

        See Also
        ----------
        set_em_gain
        """
        pass

    def get_status(self):
        """
        Camera object status, called by the GUI. This is optional.

        Returns
        ----------
        str, optional
            String indicating Camera object status.

        """
        pass

    def get_acquisition_mode(self):
        """
        Get the Camera object readout mode.

        See Also
        ----------
        set_acquisition_mode
        """
        pass

    def set_acquisition_mode(self, mode):
        """
        Set the readout mode of the Camera object. PYME currently supports two
        modes: single shot, where the camera takes one image, and then a new
        exposure has to be manually triggered, or continuous / free running,
        where the camera runs as fast as it can until we tell it to stop.

        Parameters
        ----------
        mode : int
            One of self.MODE_CONTINUOUS, self.MODE_SINGLE_SHOT

        Returns
        ----------
        None

        See Also
        ----------
        get_acquisition_mode
        """
        pass

    def set_active(self, active=True):
        """
        Determine if Camera should write metadata.

        Parameters
        ----------
        active : bool
            Write metadata?

        Returns
        ----------
        None
        """

        self.active = active

    def set_integration_time(self, integration_time):
        """
        Sets the exposure time in s. Currently assumes that we will want to go
        as fast as possible at this exposure time and also sets the frame
        rate to match.

        Parameters
        ----------
        integration_time : float
            Exposure time in s.

        Returns
        ----------
        None
        """

        pass

    def set_em_gain(self, gain):
        """
        Set the electromagnetic gain. For EMCCDs this is typically the
        uncalibrated, gain register setting. The calibrated gain is computed
        separately and saved in the metadata as RealEMGain.

        Parameters
        ----------
        gain : float
            EM gain of Camera object.

        Returns
        ----------
        None

        See Also
        ----------
        get_em_gain
        """
        pass

    def shutdown(self):
        """ Clean up the Camera object. """
        pass

    def __del__(self):
        self.shutdown()
