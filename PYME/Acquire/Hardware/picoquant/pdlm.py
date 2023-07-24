
#  PDL-M1 \"Taiko\"  Demo Application  \u00A9 2018 by PicoQuant GmbH, Keno Goertz
# modified 2023/06 Andrew Barentine

# Taiko PDL M1 Laser Driver GUI
# © 2018 - 2022 by PicoQuant GmbH

# Disclaimer

# PicoQuant GmbH disclaims all warranties with regard to this software
# and associated documentation including all implied warranties of
# merchantability and fitness. In no case shall PicoQuant GmbH be
# liable for any direct, indirect or consequential damages or any material
# or immaterial damages whatsoever resulting from loss of data, time
# or profits arising from use or performance of this software.


# License and Copyright Notice

# With this product, you have purchased a license to use the PDLM_Drv
# software. You have not purchased any other rights to the software.
# The software is protected by copyright and intellectual property laws.
# You may not distribute the software to third parties or reverse engineer,
# decompile or disassemble the software or any part thereof. Copyright
# of all product documentation belongs to PicoQuant GmbH. No parts of it
# may be reproduced or translated into other languages without written
# consent of PicoQuant GmbH. You may use and modify demo code to create
# your own software. Original or modified demo code may be re-distributed,
# provided that the original disclaimer and  notes are not
# removed from it.

import ctypes as ct
from ctypes import byref
import time
import os
import logging

logger = logging.getLogger(__name__)

# From common_files

PDLM_MAX_USBDEVICES = 8
PDLM_LDH_STRING_LENGTH = 16
PDLM_UI_COOPERATIVE = 0
PDLM_UI_EXCLUSIVE = 1
PDLM_LASER_MODE_PULSE = 1

PDLM_DEVSTATE_LOCKED_BY_ON_OFF_BUTTON = 1 << 23
PDLM_DEVSTATE_KEYLOCK = 1 << 25

MAX_TAGLIST_LEN = 64
PDLM_TAGNAME_MAXLEN = 31
PDLM_TAG_LDH_PulsePowerTable = 0x00000021
PDLM_TAG_PulsePowerHiLimit = 0x0000004A
PDLM_TAG_PulsePowerPermille = 0x000000B4
PDLM_TAG_PulseShape = 0x000000B5
PDLM_TAG_PulsePower = 0x000000B8
PDLM_TAG_PulsePowerNanowatt = 0x000000BC
PDLM_TAG_Frequency = 0x000000A8
PDLM_TAG_TempScale = 0x0000009F
PDLM_TAG_TargetTemp = 0x00000098
PDLM_TAG_TargetTempRaw = 0x00000090
PDLM_TAG_TriggerLevel = 0x00000048
PDLM_TAG_PulseShape = 0x000000B5
PDLM_TAG_NONE = 0x00000000
PDLM_TAGTYPE_INT = 0x00110001
PDLM_TAGTYPE_UINT = 0x00010001
PDLM_TAGTYPE_UINT_ENUM = 0x00010002
PDLM_TAGTYPE_SINGLE = 0x01000001

PDLM_TEMPERATURESCALE_CELSIUS = 0
PDLM_TEMPERATURESCALE_FAHRENHEIT = 1
PDLM_TEMPERATURESCALE_KELVIN = 2

PDLM_ERROR_NONE = 0
PDLM_ERROR_DEVICE_BUSY_OR_BLOCKED = -17
PDLM_ERROR_USB_INAPPROPRIATE_DEVICE = -18
PDLM_ERROR_DEVICE_ALREADY_OPENED = -27
PDLM_ERROR_OPEN_DEVICE_FAILED = -28
PDLM_ERROR_USB_UNKNOWN_DEVICE = -29
PDLM_ERROR_UNKNOWN_TAG = -34
PDLM_ERROR_VALUE_NOT_AVAILABLE = -40

ERRORS_ON_OPEN_ALLOWED = [ PDLM_ERROR_DEVICE_BUSY_OR_BLOCKED,
                           PDLM_ERROR_USB_INAPPROPRIATE_DEVICE,
                           PDLM_ERROR_DEVICE_ALREADY_OPENED,
                           PDLM_ERROR_OPEN_DEVICE_FAILED,
                           PDLM_ERROR_USB_UNKNOWN_DEVICE ]
ERRORS_ALLOWED_ON_TAG_DESCR = [ PDLM_ERROR_UNKNOWN_TAG ]
ERRORS_ALLOWED_ON_EXT_TRIG = [ PDLM_ERROR_VALUE_NOT_AVAILABLE ]

UNIT_PREFIX = ["a", "f", "p", "n", "\u03BC", "m", "", "k", "M", "G"]
TEMPSCALE_UNITS = ["°C", "°F", "K"]
PREFIX_ZEROIDX = 6
STD_BUFFER_LEN = 1024

PDLM_LASER_MODE_CW = 0x00000000
PDLM_LASER_MODE_PULSE = 0x00000001

MODE = {
    PDLM_LASER_MODE_CW : 'cw',
    PDLM_LASER_MODE_PULSE : 'pulsed'
}

PDLM_GATEIMP_10k_OHM = 0x00000000
PDLM_GATEIMP_50_OHM = 0x00000001

GATE_IMPEDANCE = {
    PDLM_GATEIMP_10k_OHM: 10000,  # 10 kOhm
    PDLM_GATEIMP_50_OHM: 50  # 50 Ohm
}

PULSE_SHAPE = {
    0: 'Broadened',
    1: 'Narrow',
    2: 'Sub-threshold, LED domain',
    3: 'Unknown'
}

TRIGGER_MODE = {
    0: 'INTERNAL',
    1: 'EXTERNAL_FALLING_EDGE',
    2: 'EXTERNAL_RISING_EDGE'
}
    

try:
    pdlm_lib = ct.WinDLL('PDLM_Lib.dll')
except:
    dir_path = r"C:\ProgramData\Taiko PDL M1\API\x64"
    pdlm_lib = ct.WinDLL(os.path.join(dir_path, "PDLM_Lib.dll"))


def check_ret_value(func_name, ret, allowed_ret_vals, error_msg):
    if (ret != PDLM_ERROR_NONE) and (ret not in allowed_ret_vals):
        if error_msg:
            print("[ERROR] %s" % error_msg)
        error_text = ct.create_string_buffer(b"", STD_BUFFER_LEN)
        buffer_len = ct.c_uint(STD_BUFFER_LEN)
        pdlm_lib.PDLM_DecodeError(ret, error_text, byref(buffer_len))
        error_text = error_text.value.decode("utf-8")
        print("[ERROR] %d occured in \"%s\"" % (ret, func_name))
        print("[ERROR] %s" % error_text)
        pdlm_lib.PDLM_CloseDevice(USB_idx)
        exit(ret)
    return ret

def calc_temp(scale_idx, raw_value):
    if scale_idx == PDLM_TEMPERATURESCALE_FAHRENHEIT:
        return (0.18*raw_value + 32)
    if scale_idx == PDLM_TEMPERATURESCALE_CELSIUS:
        return 0.1*raw_value
    if scale_idx == PDLM_TEMPERATURESCALE_KELVIN:
        return (0.1*raw_value + 273.1)


def enumerate_devices():
    USB_idx = -1
    for i in range(PDLM_MAX_USBDEVICES):
        buf = ct.create_string_buffer(b"", STD_BUFFER_LEN)
        ret = check_ret_value("PDLM_OpenGetSerNumAndClose",
                              pdlm_lib.PDLM_OpenGetSerNumAndClose(ct.c_int(i), buf),
                              ERRORS_ON_OPEN_ALLOWED, None)
        if (ret != PDLM_ERROR_USB_UNKNOWN_DEVICE) and \
                (ret != PDLM_ERROR_OPEN_DEVICE_FAILED):
            if ret == PDLM_ERROR_NONE and USB_idx < 0:
                USB_idx = i
            switcher = {
                PDLM_ERROR_DEVICE_BUSY_OR_BLOCKED: "(busy)",
                PDLM_ERROR_USB_INAPPROPRIATE_DEVICE: "(inappropriate)",
                PDLM_ERROR_DEVICE_ALREADY_OPENED: "(already opened)",
                PDLM_ERROR_NONE: "(ready to run)"
            }
            print("  USB-Index %d: PDL-M1 with serial number %s found %s" %
                  (i, buf.value.decode("utf-8"), switcher.get(ret)))
    if USB_idx < 0:
        print("  No PDL-M1 found.")

def open_device(USB_idx):
    """Open the PDL and return its serial number

    Parameters
    ----------
    USB_idx : int
        laser identifier

    Returns
    -------
    str
        serial number of the Taiko
    """
    buf = ct.create_string_buffer(b"", STD_BUFFER_LEN)
    USB_idx = ct.c_int(USB_idx)

    check_ret_value("PDLM_OpenDevice",
                    pdlm_lib.PDLM_OpenDevice(USB_idx, buf),
                    [], "  Can't open device.")

    logger.debug("  USB-Index %d: PDL-M1 opened.\n" % USB_idx.value)
    return buf.value.decode("utf-8")

def close_device(USB_idx):
    USB_idx = ct.c_int(USB_idx)

    check_ret_value("PDLM_CloseDevice",
                    pdlm_lib.PDLM_CloseDevice(USB_idx),
                    [], "  Can't close device.")
    logger.debug('Closed device %d' % USB_idx.value)

def log_status(USB_idx):
    ui_status = ct.c_uint()
    check_ret_value("PDLM_GetSystemStatus",
                    pdlm_lib.PDLM_GetSystemStatus(USB_idx, byref(ui_status)),
                    [], "  Can't read system status.")
    # Interpret the system status code:
    logger.debug("  Laser is%slocked by On/Off Button" %
          (" " if (PDLM_DEVSTATE_LOCKED_BY_ON_OFF_BUTTON & ui_status.value > 0) else " NOT "))
    logger.debug("  Laser is%slocked by key\n" %
          (" " if (PDLM_DEVSTATE_KEYLOCK & ui_status.value > 0) else " NOT "))

def get_fast_gate(USB_idx):
    enabled = ct.c_uint()
    check_ret_value("PDLM_GetFastGate",
                    pdlm_lib.PDLM_GetFastGate(USB_idx, byref(enabled)),
                    [], "  Can't read system status.")
    impedance = ct.c_uint()
    check_ret_value("PDLM_GetFastGateImp",
                    pdlm_lib.PDLM_GetFastGateImp(USB_idx, byref(impedance)),
                    [], "  Can't read system status.")
    
    return bool(enabled.value), GATE_IMPEDANCE[impedance.value]

def set_fast_gate(USB_idx, enable=True, high_impedance=True):
    enable = ct.c_uint(enable)
    check_ret_value("PDLM_SetFastGate",
                    pdlm_lib.PDLM_SetFastGate(USB_idx, enable),
                    [], "  Can't read system status.")
    if high_impedance:
        impedance = PDLM_GATEIMP_10k_OHM
    else:
        impedance = PDLM_GATEIMP_50_OHM
    check_ret_value("PDLM_SetFastGateImp",
                    pdlm_lib.PDLM_SetFastGateImp(USB_idx, impedance),
                    [], "  Can't read system status.")

def set_emission_mode(USB_idx, mode):
    """set emission mode

    Parameters
    ----------
    USB_idx : int
        laser identifier
    mode : int
        PDLM_LASER_MODE_CW = 0x00000000
        PDLM_LASER_MODE_PULSE = 0x00000001
    """
    check_ret_value("PDLM_SetLaserMode",
                    pdlm_lib.PDLM_SetLaserMode(USB_idx, mode),
                    [], "  Can't set emission mode.")

def get_emission_mode(USB_idx):
    mode = ct.c_uint()
    check_ret_value("PDLM_SetLaserMode",
                    pdlm_lib.PDLM_GetLaserMode(USB_idx, byref(mode)),
                    [], "  Can't set emission mode.")
    return MODE[mode.value]

def get_cw_power_limits(USB_idx):
    """Returns power limits for CW mode in units of Watts

    Parameters
    ----------
    USB_idx : int
        laser identifier

    Returns
    -------
    min_range : float
        minimum power output [W]
    max_range : float
        maximum power output [W]
    """
    min_power = ct.c_float()
    max_power = ct.c_float()
    check_ret_value("PDLM_GetCwPowerLimits",
                    pdlm_lib.PDLM_GetCwPowerLimits(USB_idx, byref(min_power),
                    byref(max_power)),
                    [], "  Can't get CW power limits.")
    return min_power.value, max_power.value

def get_pulsed_power_limits(USB_idx):
    """Returns pusled power limits in units of Watts

    Parameters
    ----------
    USB_idx : int
        laser identifier

    Returns
    -------
    min_range : float
        minimum power output [W]
    max_range : float
        maximum power output [W]
    """
    min_power = ct.c_float()
    max_power = ct.c_float()
    check_ret_value("PDLM_GetPulsePowerLimits",
                    pdlm_lib.PDLM_GetPulsePowerLimits(USB_idx, byref(min_power),
                    byref(max_power)),
                    [], "  Can't get CW power limits.")
    return min_power.value, max_power.value

def get_cw_power(USB_idx):
    """

    Parameters
    ----------
    USB_idx : int
        laser identifier

    Returns
    -------
    power : float
        CW power setting [W]
    """
    power = ct.c_float()
    check_ret_value("PDLM_GetCwPower",
                    pdlm_lib.PDLM_GetCwPower(USB_idx, byref(power)),
                    [], "  Can't get CW power.")
    return power.value

def set_cw_power(USB_idx, power):
    """

    Parameters
    ----------
    USB_idx : int
        laser identifier
    power : float
        CW power [W] to set laser output to
    """
    power = ct.c_float(power)
    check_ret_value("PDLM_SetCwPower",
                    pdlm_lib.PDLM_SetCwPower(USB_idx, power),
                    [], "  Can't set CW power.")

def get_pulsed_power(USB_idx):
    """

    Parameters
    ----------
    USB_idx : int
        laser identifier

    Returns
    -------
    power : float
        Time-averaged power output [W], useful for
        pulsed mode with internal trigger.
    """
    power = ct.c_float()
    check_ret_value("PDLM_GetPulsePower",
                    pdlm_lib.PDLM_GetPulsePower(USB_idx, byref(power)),
                    [], "  Can't get pulsed power.")
    return power.value

def set_pulsed_power(USB_idx, power):
    """

    Parameters
    ----------
    USB_idx : int
        laser identifier
    power : float
        Power [W] to set laser output to. Time-averaged power, useful for
        pulsed mode with internal trigger.
    """
    power = ct.c_float(power)
    check_ret_value("PDLM_SetCwPower",
                    pdlm_lib.PDLM_SetPulsePower(USB_idx, power),
                    [], "  Can't set pulsed power.")

def get_pulse_shape(USB_idx):
    """return whether pulse shape is within spec or not

    Parameters
    ----------
    USB_idx : int
        laser identifier

    Returns
    -------
    description : str
        description of pulse shape
    """
    shape = ct.c_uint()
    check_ret_value("PDLM_GetPulseShape",
                    pdlm_lib.PDLM_GetPulseShape(USB_idx, byref(shape)),
                    [], "  Can't get pulse shape.")
    return PULSE_SHAPE[shape.value]

def get_trigger_mode(USB_idx):
    mode = ct.c_uint()
    check_ret_value("PDLM_GetTriggerMode",
                    pdlm_lib.PDLM_GetTriggerMode(USB_idx, byref(mode)),
                    [], "  Can't set emission mode.")
    return TRIGGER_MODE[mode.value]

def get_ext_trigger_freq(USB_idx):
    """

    Returns
    -------
    int
        Frequency [Hz] of external trigger (roughly, inaccurate below 8kHz)
    """
    freq = ct.c_uint()
    check_ret_value("PDLM_GetExtTriggerFrequency",
                    pdlm_lib.PDLM_GetExtTriggerFrequency(USB_idx, byref(freq)),
                    [], "  Can't get ext trig freq.")
    return freq.value

def get_frequency(USB_idx):
    """

    Returns
    -------
    int
        Frequency [Hz] of internally triggered pulses or bursts
    """
    freq = ct.c_uint()
    check_ret_value("PDLM_GetFrequency",
                    pdlm_lib.PDLM_GetFrequency(USB_idx, byref(freq)),
                    [], "  Can't get ext trig freq.")
    return freq.value


def set_softlock(USB_idx, enable):
    """

    Parameters
    ----------
    USB_idx : int
        laser identifier
    enable : bool
        whether to enable or disable softlock (i.e. turn on / off laser)
    """
    enable = ct.c_int(enable)
    check_ret_value("PDLM_SetSoftLock",
                    pdlm_lib.PDLM_SetSoftLock(USB_idx, enable),
                    [], "  Can't set pulsed power.")

def get_softlock(USB_idx):
    """

    Parameters
    ----------
    USB_idx : int
        laser identifier
    
    Returns
    -------
    enabled : bool
        whether the laser softlock is on/off
    """
    enabled = ct.c_int()
    check_ret_value("PDLM_GetSoftLock",
                    pdlm_lib.PDLM_GetSoftLock(USB_idx, byref(enabled)),
                    [], "  error.")
    return bool(enabled.value)

def get_lock_status(USB_idx):
    """

    Parameters
    ----------
    USB_idx : int
        laser identifier
    
    Returns
    -------
    enabled : bool
        whether the laser is locked or unlocked, for whatever reason
    """
    enabled = ct.c_int()
    check_ret_value("PDLM_GetLocked",
                    pdlm_lib.PDLM_GetLocked(USB_idx, byref(enabled)),
                    [], "  error.")
    return bool(enabled.value)
