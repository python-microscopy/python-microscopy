#!/usr/bin/python

###############
# freeSpace.py
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
import os
import platform
import ctypes
import warnings

def get_free_space_bytes(folder):
    """ Return folder/drive free space (in bytes)
    """
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        total_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, ctypes.pointer(total_bytes), ctypes.pointer(free_bytes))
        return free_bytes.value
    else:
        stats = os.statvfs(folder)
        return stats.f_bfree*stats.f_frsize
    
def get_free_space(folder):
    """ Return folder/drive free space (in bytes)
    """
    warnings.warn(DeprecationWarning('Use get_free_space_bytes() instead'))
    return get_free_space_bytes(folder)

def disk_usage(folder):
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        total_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, ctypes.pointer(total_bytes), ctypes.pointer(free_bytes))
        free = free_bytes.value
        total = total_bytes.value
    else:
        stats = os.statvfs(folder)
        total = stats.f_blocks * stats.f_frsize
        free = stats.f_bavail * stats.f_frsize
    used = total - free
    return total, used, free