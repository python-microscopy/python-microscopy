"""
Handle camera noise properties for all cameras
Support legacy, hard-coded properties as well as a more flexible .yaml based configuration within the PYME.config framework

This module provides a dictionary, `noise_properties` which is indexed by the camera serial number, under the assumption that serial numbers collisions
between manufacturers will be unlikely.

Camera noise properties may be specified in .yaml files within the `cameras` subdirectory of any of the PYME config directories (see `PYME.config` for details),.
The yaml files should encode a dictionary in the following form:

.. code-block:: yaml
    
    # An Andor Zyla entry
    VSC-00954:
        noise_properties:
            12-bit (high well capacity):
                ADOffset: 100
                ElectronsPerCount: 6.97
                ReadNoise: 5.96
                SaturationThreshold: 2047
            12-bit (low noise):
                ADOffset: 100
                ElectronsPerCount: 0.28
                ReadNoise: 1.1
                SaturationThreshold: 2047
            16-bit (low noise & high well capacity):
                ADOffset: 100
                ElectronsPerCount: 0.5
                ReadNoise: 1.33
                SaturationThreshold: 65535

    # An Andor IXon entry:
    5414:
        default_preamp_gain: 0
        noise_properties:
            Preamp Gain 0:
                ADOffset: 413
                DefaultEMGain: 90
                ElectronsPerCount: 25.24
                NGainStages: 536
                ReadNoise: 61.33
                SaturationThreshold: 16383

    # A HamamatsuORCA entry:
    '100233':
        noise_properties:
            fixed:
                ADOffset: 100
                DefaultEMGain: 1
                ElectronsPerCount: 0.47
                NGainStages: 0
                ReadNoise: 1.65
                SaturationThreshold: 65535

This dictionary is indexed by camera serial number. Each camera entry is itself a dictionary, and must have a dictionary called `noise_properties` as one entry.
The `noise_properties` dictionary contains a set of dictionaries of readout characteristics for each gain mode of the camera (the keys here will typically vary between
camera types). It is permissible to put additional entries in the camera dictionaries e.g. the `default_preamp_gain` entry for the IXon above, but these should be treated
as informational and code should ideally not depend on their prescence (we make a slight exception here for the IXon code as the noise properties for non-default modes have
not been recorded, but the behaviour is discouraged). 

All .yaml files in the cameras subdirectory are read and their contents amalgamated.

A number of hard-coded camera noise values are also provided in this file (_legacy_noise_properties)

NOTE: we handle this here, rather than in PYME.config to keep legacy info out of PYME.config and avoid introducing a back-dependancy 
from PYME.config on PYMEAcquire
"""
from PYME import config
import yaml
import os
import glob

noise_properties = {}

# we have historically recorded noise properties in the code of the relevant camera class. All these legacy settings are now moved here.
_legacy_noise_properties = {
    # Andor IXon cameras
    1823 : {
        'default_preamp_gain' : 0,
        'noise_properties': {
            'Preamp Gain 0': {
                'ReadNoise' : 109.8,
                'ElectronsPerCount' : 27.32,
                'NGainStages' : 536,
                'ADOffset' : 971,
                'DefaultEMGain' : 150,
                'SaturationThreshold' : (2**14 -1)
            }}},
    5414 : {
        'default_preamp_gain' : 0,
        'noise_properties': {
            'Preamp Gain 0': {
                'ReadNoise' : 61.33,
                'ElectronsPerCount' : 25.24,
                'NGainStages' : 536,
                'ADOffset' : 413,
                'DefaultEMGain' : 90,
                'SaturationThreshold' : (2**14 -1)
            }}},
    7863 : { #Gain setting of 3
        'defaultPreampGain' : 2,
        'noise_properties': {
            'Preamp Gain 2': {
                'ReadNoise' : 88.1,
                'ElectronsPerCount' : 4.99,
                'NGainStages' : 536,
                'ADOffset' : 203,
                'DefaultEMGain' : 90,
                'SaturationThreshold' : 5.4e4#(2**16 -1)
            }}},
    7546 : {
        'defaultPreampGain' : 2,
        'noise_properties': {
            'Preamp Gain 2': {
                #  preamp: currently using most sensitive setting (default according to docs)
                # if I understand the code correctly the fastest Horizontal Shift Speed will be selected
                # which should be 17 MHz for this camera; therefore using 17 MHz data
                'ReadNoise' : 85.23,
                'ElectronsPerCount' : 4.82,
                'NGainStages' : 536, # relevant?
                'ADOffset' : 150, # from test measurement at EMGain 85 (realgain ~30)
                'DefaultEMGain' : 85, # we start carefully and can bumb this later to be in the vicinity of 30
                'SaturationThreshold' : (2**16 -1) # this cam has 16 bit data
            }}},

    # Andor Zyla cameras
    'VSC-00954': {
        'model' : 'Zyla', # model param currently not used 
        'noiseProperties': {
            '12-bit (low noise)': {
                'ReadNoise' : 1.1,
                'ElectronsPerCount' : 0.28,
                'ADOffset' : 100, # check mean (or median) offset
                'SaturationThreshold' : 2**11-1#(2**16 -1) # check this is really 11 bit
            },
            '12-bit (high well capacity)': {
                'ReadNoise' : 5.96,
                'ElectronsPerCount' : 6.97,
                'ADOffset' : 100,
                'SaturationThreshold' : 2**11-1#(2**16 -1)         
            },
            '16-bit (low noise & high well capacity)': {
                'ReadNoise' : 1.33,
                'ElectronsPerCount' : 0.5,
                'ADOffset' : 100,
                'SaturationThreshold' : (2**16 -1)
            }}},
    'CSC-00425': { # this is info for a Sona
        'noiseProperties': {
            u'12-bit (low noise)': {
                'ReadNoise' : 1.21,
                'ElectronsPerCount' : 0.45,
                'ADOffset' : 100, # check mean (or median) offset
                'SaturationThreshold' : 1776  #(2**16 -1) # check this is really 11 bit
            },
            u'16-bit (high dynamic range)': {
                'ReadNoise' : 1.84,
                'ElectronsPerCount' : 1.08,
                'ADOffset' : 100,
                'SaturationThreshold' : 44185
            }}},
    'VSC-02858': {
        'noiseProperties': {
            '12-bit (low noise)': {
                'ReadNoise' : 1.19,
                'ElectronsPerCount' : 0.3,
                'ADOffset' : 100, # check mean (or median) offset
                'SaturationThreshold' : 2**11-1#(2**16 -1) # check this is really 11 bit
            },
            '12-bit (high well capacity)': {
                'ReadNoise' : 6.18,
                'ElectronsPerCount' : 7.2,
                'ADOffset' : 100,
                'SaturationThreshold' : 2**11-1#(2**16 -1)         
            },
            '16-bit (low noise & high well capacity)': {
                'ReadNoise' : 1.42,
                'ElectronsPerCount' : 0.5,
                'ADOffset' : 100,
                'SaturationThreshold' : (2**16 -1)
            }}},
    'VSC-02698': {
        'noiseProperties': {
            '12-bit (low noise)': {
                'ReadNoise' : 1.16,
                'ElectronsPerCount' : 0.26,
                'ADOffset' : 100, # check mean (or median) offset
                'SaturationThreshold' : 2**11-1#(2**16 -1) # check this is really 11 bit
            },
            '12-bit (high well capacity)': {
                'ReadNoise' : 6.64,
                'ElectronsPerCount' : 7.38,
                'ADOffset' : 100,
                'SaturationThreshold' : 2**11-1#(2**16 -1)         
            },
            '16-bit (low noise & high well capacity)': {
                'ReadNoise' : 1.36,
                'ElectronsPerCount' : 0.49,
                'ADOffset' : 100,
                'SaturationThreshold' : (2**16 -1)
            }}},
    
    # Hamamatsu ORCA flash
    '100233' : {
        'noiseProperties': {
            'fixed' : {
                'ReadNoise': 1.65, #CHECKME - converted from an ADU value of 3.51
                'ElectronsPerCount': 0.47,
                'NGainStages': 0,
                'ADOffset': 100,
                'DefaultEMGain': 1,
                'SaturationThreshold': (2**16 - 1)
            }}},
    '301777' : {
        'noiseProperties': {
            'fixed' : {
                'ReadNoise': 1.63,
                'ElectronsPerCount': 0.47,
                'NGainStages': 0,
                'ADOffset': 100,
                'DefaultEMGain': 1,
                'SaturationThreshold': (2**16 - 1)
            }}},        
    '720795' : {
        'noiseProperties': {
            'fixed' : {
                'ReadNoise': 0.997,  # rn is sqrt(var) in units of electrons. Median of varmap is 0.9947778 [e-^2] #CHECKME - converted from 2.394 ADU
                'ElectronsPerCount': 0.416613,
                'NGainStages': 0,
                'ADOffset': 101.753685,
                'DefaultEMGain': 1,
                'SaturationThreshold': (2**16 - 1)
            }}},
}

# add the legacy camera info to noise_properties
noise_properties.update(_legacy_noise_properties)

# parse info from .yamls in config/cameras/
cam_yamls = glob.glob(os.path.join(config.config_dir, 'cameras','*.yaml'))
for yamlfile in cam_yamls:
    with open(yamlfile,'r') as fi:
        noise_properties.update(yaml.safe_load(fi))

def add_camera_noise_info(serial_num, noise_info):
    """
    Programatically (e.g. in PYMEAcquire init script) add to our database of noise info. 

    NOTE: this will not persist across sessions.
    """

    # do some checks so we fail promptly
    np = noise_info['noise_properties']

    for k, v in np.items():
        #check for required keys
        rn = v['ReadNoise'] # will raise KeyError if not present
        epc = v['ElectronsPerCount'] # ditto

    noise_properties[serial_num] = noise_info
