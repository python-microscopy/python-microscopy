# template for pymeconf.py
# to use copy this file to pymeconf.py and edit as needed

from PYME.Acquire.Hardware.comports import ComPort

hwconfig = {
    'Mercury' :            ComPort('COM4', doc='Mercury translation stage'),
    'Pifoc'   :            ComPort('COM5', doc='Pifoc piezo focussing stage; prolific USB-to-Serial'),
    'Filterwheel' :        ComPort('COM8', doc='Thorlabs Filterwheel'),
    'Laser642' :           ComPort('COM7', doc='642nm laser'),
    'Laser671' :           ComPort('COM12', doc='671nm laser; leostick'),
    'Laser405' :           ComPort('COM13', doc='405nm laser'),
    'Laser561' :           ComPort('COM15', doc='561nm laser'),
    'Lumen200S' :          ComPort('COM6', doc='Prior Lumen200S arclamp shutter'),
}

# # bits to include in init_***.py files:
# from PYME.Acquire.Hardware.comports import ComPort

# try:
#     from PYME.Acquire.pymeconf import hwconfig
# except ImportError:
#     print "could not import pymeconf.py"
#     hwconfig = None

# if hwconfig is None:
#     hwconfig = ... # alternative local setup
