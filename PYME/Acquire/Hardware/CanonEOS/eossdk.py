#!/usr/bin/python

###############
# eossdk.py
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
"""Wrapper for EDSDK.h

Generated with:
C:\Python27\Scripts\ctypesgen.py -l EDSDK -o eos.py EDSDK.h EDSDKErrors.h EDSDKTypes.h

Do not modify this file.
"""

__docformat__ =  'restructuredtext'

# Begin preamble

import ctypes, os, sys
from ctypes import *

_int_types = (c_int16, c_int32)
if hasattr(ctypes, 'c_int64'):
    # Some builds of ctypes apparently do not have c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (c_int64,)
for t in _int_types:
    if sizeof(t) == sizeof(c_size_t):
        c_ptrdiff_t = t
del t
del _int_types

class c_void(Structure):
    # c_void_p is a buggy return type, converting to int, so
    # POINTER(None) == c_void_p is actually written as
    # POINTER(c_void), so it can be treated as a real pointer.
    _fields_ = [('dummy', c_int)]

def POINTER(obj):
    p = ctypes.POINTER(obj)

    # Convert None to a real NULL pointer to work around bugs
    # in how ctypes handles None on 64-bit platforms
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

class UserString:
    def __init__(self, seq):
        if isinstance(seq, basestring):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq)
    def __str__(self): return str(self.data)
    def __repr__(self): return repr(self.data)
    def __int__(self): return int(self.data)
    def __long__(self): return long(self.data)
    def __float__(self): return float(self.data)
    def __complex__(self): return complex(self.data)
    def __hash__(self): return hash(self.data)

    def __cmp__(self, string):
        if isinstance(string, UserString):
            return cmp(self.data, string.data)
        else:
            return cmp(self.data, string)
    def __contains__(self, char):
        return char in self.data

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.__class__(self.data[index])
    def __getslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, basestring):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other))
    def __radd__(self, other):
        if isinstance(other, basestring):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other) + self.data)
    def __mul__(self, n):
        return self.__class__(self.data*n)
    __rmul__ = __mul__
    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self): return self.__class__(self.data.capitalize())
    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))
    def count(self, sub, start=0, end=sys.maxsize):
        return self.data.count(sub, start, end)
    def decode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())
    def encode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())
    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self.data.endswith(suffix, start, end)
    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))
    def find(self, sub, start=0, end=sys.maxsize):
        return self.data.find(sub, start, end)
    def index(self, sub, start=0, end=sys.maxsize):
        return self.data.index(sub, start, end)
    def isalpha(self): return self.data.isalpha()
    def isalnum(self): return self.data.isalnum()
    def isdecimal(self): return self.data.isdecimal()
    def isdigit(self): return self.data.isdigit()
    def islower(self): return self.data.islower()
    def isnumeric(self): return self.data.isnumeric()
    def isspace(self): return self.data.isspace()
    def istitle(self): return self.data.istitle()
    def isupper(self): return self.data.isupper()
    def join(self, seq): return self.data.join(seq)
    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))
    def lower(self): return self.__class__(self.data.lower())
    def lstrip(self, chars=None): return self.__class__(self.data.lstrip(chars))
    def partition(self, sep):
        return self.data.partition(sep)
    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))
    def rfind(self, sub, start=0, end=sys.maxsize):
        return self.data.rfind(sub, start, end)
    def rindex(self, sub, start=0, end=sys.maxsize):
        return self.data.rindex(sub, start, end)
    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))
    def rpartition(self, sep):
        return self.data.rpartition(sep)
    def rstrip(self, chars=None): return self.__class__(self.data.rstrip(chars))
    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)
    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)
    def splitlines(self, keepends=0): return self.data.splitlines(keepends)
    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self.data.startswith(prefix, start, end)
    def strip(self, chars=None): return self.__class__(self.data.strip(chars))
    def swapcase(self): return self.__class__(self.data.swapcase())
    def title(self): return self.__class__(self.data.title())
    def translate(self, *args):
        return self.__class__(self.data.translate(*args))
    def upper(self): return self.__class__(self.data.upper())
    def zfill(self, width): return self.__class__(self.data.zfill(width))

class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""
    def __init__(self, string=""):
        self.data = string
    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")
    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + sub + self.data[index+1:]
    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + self.data[index+1:]
    def __setslice__(self, start, end, sub):
        start = max(start, 0); end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start]+sub.data+self.data[end:]
        elif isinstance(sub, basestring):
            self.data = self.data[:start]+sub+self.data[end:]
        else:
            self.data =  self.data[:start]+str(sub)+self.data[end:]
    def __delslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]
    def immutable(self):
        return UserString(self.data)
    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, basestring):
            self.data += other
        else:
            self.data += str(other)
        return self
    def __imul__(self, n):
        self.data *= n
        return self

class String(MutableString, Union):

    _fields_ = [('raw', POINTER(c_char)),
                ('data', c_char_p)]

    def __init__(self, obj=""):
        if isinstance(obj, (str, unicode, UserString)):
            self.data = str(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(POINTER(c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj)

        # Convert from c_char_p
        elif isinstance(obj, c_char_p):
            return obj

        # Convert from POINTER(c_char)
        elif isinstance(obj, POINTER(c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(cast(obj, POINTER(c_char)))

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)
    from_param = classmethod(from_param)

def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)

# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to c_void_p.
def UNCHECKED(type):
    if (hasattr(type, "_type_") and isinstance(type._type_, str)
        and type._type_ != "P"):
        return type
    else:
        return c_void_p

# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self,func,restype,argtypes):
        self.func=func
        self.func.restype=restype
        self.argtypes=argtypes
    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func
    def __call__(self,*args):
        fixed_args=[]
        i=0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i+=1
        return self.func(*fixed_args+list(args[i:]))

# End preamble

_libs = {}
_libdirs = []

# Begin loader

# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import os.path, re, sys, glob
import ctypes
import ctypes.util

def _environ_path(name):
    if name in os.environ:
        return os.environ[name].split(":")
    else:
        return []

class LibraryLoader(object):
    def __init__(self):
        self.other_dirs=[]

    def load_library(self,libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            if os.path.exists(path):
                return self.load(path)

        raise ImportError("%s not found." % libname)

    def load(self,path):
        """Given a path to a library, load it."""
        try:
            # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
            # of the default RTLD_LOCAL.  Without this, you end up with
            # libraries not being loadable, resulting in "Symbol not found"
            # errors
            if sys.platform == 'darwin':
                return ctypes.CDLL(path, ctypes.RTLD_GLOBAL)
            else:
                return ctypes.cdll.LoadLibrary(path)
        except OSError,e:
            raise ImportError(e)

    def getpaths(self,libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # FIXME / TODO return '.' and os.path.dirname(__file__)
            for path in self.getplatformpaths(libname):
                yield path

            path = ctypes.util.find_library(libname)
            if path: yield path

    def getplatformpaths(self, libname):
        return []

# Darwin (Mac OS X)

class DarwinLibraryLoader(LibraryLoader):
    name_formats = ["lib%s.dylib", "lib%s.so", "lib%s.bundle", "%s.dylib",
                "%s.so", "%s.bundle", "%s"]

    def getplatformpaths(self,libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [format % libname for format in self.name_formats]

        for dir in self.getdirs(libname):
            for name in names:
                yield os.path.join(dir,name)

    def getdirs(self,libname):
        """Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        """

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [os.path.expanduser('~/lib'),
                                          '/usr/local/lib', '/usr/lib']

        dirs = []

        if '/' in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))

        dirs.extend(self.other_dirs)
        dirs.append(".")
        dirs.append(os.path.dirname(__file__))

        if hasattr(sys, 'frozen') and sys.frozen == 'macosx_app':
            dirs.append(os.path.join(
                os.environ['RESOURCEPATH'],
                '..',
                'Frameworks'))

        dirs.extend(dyld_fallback_library_path)

        return dirs

# Posix

class PosixLibraryLoader(LibraryLoader):
    _ld_so_cache = None

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = []
        for name in ("LD_LIBRARY_PATH",
                     "SHLIB_PATH", # HPUX
                     "LIBPATH", # OS/2, AIX
                     "LIBRARY_PATH", # BE/OS
                    ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))
        directories.extend(self.other_dirs)
        directories.append(".")
        directories.append(os.path.dirname(__file__))

        try: directories.extend([dir.strip() for dir in open('/etc/ld.so.conf')])
        except IOError: pass

        directories.extend(['/lib', '/usr/lib', '/lib64', '/usr/lib64'])

        cache = {}
        lib_re = re.compile(r'lib(.*)\.s[ol]')
        ext_re = re.compile(r'\.s[ol]$')
        for dir in directories:
            try:
                for path in glob.glob("%s/*.s[ol]*" % dir):
                    file = os.path.basename(path)

                    # Index by filename
                    if file not in cache:
                        cache[file] = path

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        if library not in cache:
                            cache[library] = path
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname)
        if result: yield result

        path = ctypes.util.find_library(libname)
        if path: yield os.path.join("/lib",path)

# Windows

class _WindowsLibrary(object):
    def __init__(self, path):
        self.cdll = ctypes.cdll.LoadLibrary(path)
        self.windll = ctypes.windll.LoadLibrary(path)

    def __getattr__(self, name):
        try: return getattr(self.windll,name)
        except AttributeError:
            try: return getattr(self.cdll,name)
            except AttributeError:
                raise

class WindowsLibraryLoader(LibraryLoader):
    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll"]

    def load_library(self, libname):
        try:
            result = LibraryLoader.load_library(self, libname)
        except ImportError:
            result = None
            if os.path.sep not in libname:
                for name in self.name_formats:
                    try:
                        result = getattr(ctypes.cdll, name % libname)
                        if result:
                            break
                    except WindowsError:
                        result = None
            if result is None:
                try:
                    result = getattr(ctypes.cdll, libname)
                except WindowsError:
                    result = None
            if result is None:
                raise ImportError("%s not found." % libname)
        return result

    def load(self, path):
        return _WindowsLibrary(path)

    def getplatformpaths(self, libname):
        if os.path.sep not in libname:
            for name in self.name_formats:
                dll_in_current_dir = os.path.abspath(name % libname)
                if os.path.exists(dll_in_current_dir):
                    yield dll_in_current_dir
                path = ctypes.util.find_library(name % libname)
                if path:
                    yield path

# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin":   DarwinLibraryLoader,
    "cygwin":   WindowsLibraryLoader,
    "win32":    WindowsLibraryLoader
}

loader = loaderclass.get(sys.platform, PosixLibraryLoader)()

def add_library_search_dirs(other_dirs):
    loader.other_dirs = other_dirs

load_library = loader.load_library

del loaderclass

# End loader

add_library_search_dirs([])

# Begin libraries

_libs["EDSDK"] = load_library("EDSDK")

# 1 libraries
# End libraries

# No modules

WCHAR = c_wchar # c:\\python27\\mingw\\bin\\../lib/gcc/mingw32/4.5.2/../../../../include/winnt.h: 105

EdsVoid = None # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 81

EdsBool = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 82

EdsChar = c_char # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 84

EdsInt8 = c_char # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 85

EdsUInt8 = c_ubyte # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 86

EdsInt16 = c_short # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 87

EdsUInt16 = c_ushort # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 88

EdsInt32 = c_long # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 89

EdsUInt32 = c_ulong # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 90

EdsInt64 = c_longlong # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 101

EdsUInt64 = c_ulonglong # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 102

EdsFloat = c_float # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 105

EdsDouble = c_double # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 106

EdsError = EdsUInt32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 112

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 117
class struct___EdsObject(Structure):
    pass

EdsBaseRef = POINTER(struct___EdsObject) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 117
#EdsBaseRef = POINTER(c_void)

EdsCameraListRef = EdsBaseRef # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 119

EdsCameraRef = EdsBaseRef # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 120

EdsVolumeRef = EdsBaseRef # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 121

EdsDirectoryItemRef = EdsBaseRef # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 122

EdsStreamRef = EdsBaseRef # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 124

EdsImageRef = EdsStreamRef # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 125

EdsEvfImageRef = EdsBaseRef # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 127

enum_anon_242 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Unknown = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Bool = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_String = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Int8 = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_UInt8 = 6 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Int16 = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_UInt16 = 7 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Int32 = 8 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_UInt32 = 9 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Int64 = 10 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_UInt64 = 11 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Float = 12 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Double = 13 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_ByteBlock = 14 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Rational = 20 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Point = 21 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Rect = 22 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Time = 23 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Bool_Array = 30 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Int8_Array = 31 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Int16_Array = 32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Int32_Array = 33 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_UInt8_Array = 34 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_UInt16_Array = 35 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_UInt32_Array = 36 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_Rational_Array = 37 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_FocusInfo = 101 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

kEdsDataType_PictureStyleDesc = (kEdsDataType_FocusInfo + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

EdsDataType = enum_anon_242 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 165

EdsPropertyID = EdsUInt32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 170

EdsCameraCommand = EdsUInt32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 306

enum_anon_243 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 324

kEdsCameraCommand_EvfAf_OFF = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 324

kEdsCameraCommand_EvfAf_ON = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 324

EdsEvfAf = enum_anon_243 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 324

enum_anon_244 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 333

kEdsCameraCommand_ShutterButton_OFF = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 333

kEdsCameraCommand_ShutterButton_Halfway = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 333

kEdsCameraCommand_ShutterButton_Completely = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 333

kEdsCameraCommand_ShutterButton_Halfway_NonAF = 65537 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 333

kEdsCameraCommand_ShutterButton_Completely_NonAF = 65539 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 333

EdsShutterButton = enum_anon_244 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 333

EdsCameraStatusCommand = EdsUInt32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 337

EdsPropertyEvent = EdsUInt32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 349

EdsObjectEvent = EdsUInt32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 378

EdsStateEvent = EdsUInt32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 472

enum_anon_245 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 537

kEdsEvfDriveLens_Near1 = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 537

kEdsEvfDriveLens_Near2 = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 537

kEdsEvfDriveLens_Near3 = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 537

kEdsEvfDriveLens_Far1 = 32769 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 537

kEdsEvfDriveLens_Far2 = 32770 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 537

kEdsEvfDriveLens_Far3 = 32771 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 537

EdsEvfDriveLens = enum_anon_245 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 537

enum_anon_246 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 547

kEdsEvfDepthOfFieldPreview_OFF = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 547

kEdsEvfDepthOfFieldPreview_ON = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 547

EdsEvfDepthOfFieldPreview = enum_anon_246 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 547

enum_anon_247 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 559

kEdsSeek_Cur = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 559

kEdsSeek_Begin = (kEdsSeek_Cur + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 559

kEdsSeek_End = (kEdsSeek_Begin + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 559

EdsSeekOrigin = enum_anon_247 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 559

enum_anon_248 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 571

kEdsAccess_Read = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 571

kEdsAccess_Write = (kEdsAccess_Read + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 571

kEdsAccess_ReadWrite = (kEdsAccess_Write + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 571

kEdsAccess_Error = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 571

EdsAccess = enum_anon_248 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 571

enum_anon_249 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 584

kEdsFileCreateDisposition_CreateNew = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 584

kEdsFileCreateDisposition_CreateAlways = (kEdsFileCreateDisposition_CreateNew + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 584

kEdsFileCreateDisposition_OpenExisting = (kEdsFileCreateDisposition_CreateAlways + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 584

kEdsFileCreateDisposition_OpenAlways = (kEdsFileCreateDisposition_OpenExisting + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 584

kEdsFileCreateDisposition_TruncateExsisting = (kEdsFileCreateDisposition_OpenAlways + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 584

EdsFileCreateDisposition = enum_anon_249 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 584

enum_anon_250 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 599

kEdsImageType_Unknown = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 599

kEdsImageType_Jpeg = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 599

kEdsImageType_CRW = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 599

kEdsImageType_RAW = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 599

kEdsImageType_CR2 = 6 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 599

EdsImageType = enum_anon_250 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 599

enum_anon_251 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Large = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Middle = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Small = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Middle1 = 5 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Middle2 = 6 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Small1 = 14 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Small2 = 15 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Small3 = 16 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

kEdsImageSize_Unknown = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

EdsImageSize = enum_anon_251 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 616

enum_anon_252 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 629

kEdsCompressQuality_Normal = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 629

kEdsCompressQuality_Fine = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 629

kEdsCompressQuality_Lossless = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 629

kEdsCompressQuality_SuperFine = 5 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 629

kEdsCompressQuality_Unknown = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 629

EdsCompressQuality = enum_anon_252 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 629

enum_anon_253 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LJ = 1113871 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_M1J = 84999951 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_M2J = 101777167 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SJ = 34668303 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LJF = 1310479 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LJN = 1244943 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MJF = 18087695 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MJN = 18022159 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SJF = 34864911 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SJN = 34799375 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_S1JF = 236191503 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_S1JN = 236125967 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_S2JF = 252968719 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_S3JF = 269745935 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LR = 6618895 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRLJF = 6553619 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRLJN = 6553618 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRMJF = 6553875 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRMJN = 6553874 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRSJF = 6554131 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRSJN = 6554130 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRS1JF = 6557203 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRS1JN = 6557202 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRS2JF = 6557459 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRS3JF = 6557715 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRLJ = 6553616 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRM1J = 6554896 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRM2J = 6555152 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_LRSJ = 6554128 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MR = 23396111 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRLJF = 23330835 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRLJN = 23330834 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRMJF = 23331091 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRMJN = 23331090 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRSJF = 23331347 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRSJN = 23331346 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRS1JF = 23334419 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRS1JN = 23334418 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRS2JF = 23334675 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRS3JF = 23334931 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRLJ = 23330832 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRM1J = 23332112 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRM2J = 23332368 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_MRSJ = 23331344 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SR = 40173327 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRLJF = 40108051 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRLJN = 40108050 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRMJF = 40108307 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRMJN = 40108306 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRSJF = 40108563 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRSJN = 40108562 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRS1JF = 40111635 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRS1JN = 40111634 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRS2JF = 40111891 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRS3JF = 40112147 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRLJ = 40108048 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRM1J = 40109328 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRM2J = 40109584 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_SRSJ = 40108560 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality_Unknown = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

EdsImageQuality = enum_anon_253 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 708

enum_anon_254 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LJ = 2031631 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_M1J = 85917711 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_M2J = 102694927 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_SJ = 35586063 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LJF = 1245184 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LJN = 1179648 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_MJF = 18022400 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_MJN = 17956864 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_SJF = 34799616 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_SJN = 34734080 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LR = 2359296 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LRLJF = 2359315 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LRLJN = 2359314 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LRMJF = 2359571 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LRMJN = 2359570 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LRSJF = 2359827 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LRSJN = 2359826 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LR2 = 3080207 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LR2LJ = 3080223 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LR2M1J = 3081503 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LR2M2J = 3081759 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_LR2SJ = 3080735 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

kEdsImageQualityForLegacy_Unknown = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

EdsImageQualityForLegacy = enum_anon_254 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 738

enum_anon_255 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 752

kEdsImageSrc_FullView = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 752

kEdsImageSrc_Thumbnail = (kEdsImageSrc_FullView + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 752

kEdsImageSrc_Preview = (kEdsImageSrc_Thumbnail + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 752

kEdsImageSrc_RAWThumbnail = (kEdsImageSrc_Preview + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 752

kEdsImageSrc_RAWFullView = (kEdsImageSrc_RAWThumbnail + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 752

EdsImageSource = enum_anon_255 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 752

enum_anon_256 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

kEdsTargetImageType_Unknown = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

kEdsTargetImageType_Jpeg = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

kEdsTargetImageType_TIFF = 7 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

kEdsTargetImageType_TIFF16 = 8 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

kEdsTargetImageType_RGB = 9 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

kEdsTargetImageType_RGB16 = 10 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

kEdsTargetImageType_DIB = 11 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

EdsTargetImageType = enum_anon_256 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 768

enum_anon_257 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 779

kEdsProgressOption_NoReport = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 779

kEdsProgressOption_Done = (kEdsProgressOption_NoReport + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 779

kEdsProgressOption_Periodically = (kEdsProgressOption_Done + 1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 779

EdsProgressOption = enum_anon_257 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 779

enum_anon_258 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 793

kEdsFileAttribute_Normal = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 793

kEdsFileAttribute_ReadOnly = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 793

kEdsFileAttribute_Hidden = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 793

kEdsFileAttribute_System = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 793

kEdsFileAttribute_Archive = 32 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 793

EdsFileAttributes = enum_anon_258 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 793

enum_anon_259 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_Empty = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_Low = 9 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_Half = 49 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_Normal = 80 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_Hi = 69 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_Quarter = 19 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_Error = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_BCLevel = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

kEdsBatteryLevel2_AC = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

EdsBatteryLevel2 = enum_anon_259 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 810

enum_anon_260 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 821

kEdsSaveTo_Camera = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 821

kEdsSaveTo_Host = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 821

kEdsSaveTo_Both = (kEdsSaveTo_Camera | kEdsSaveTo_Host) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 821

EdsSaveTo = enum_anon_260 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 821

enum_anon_261 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 833

kEdsStorageType_Non = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 833

kEdsStorageType_CF = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 833

kEdsStorageType_SD = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 833

kEdsStorageType_HD = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 833

EdsStorageType = enum_anon_261 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 833

enum_anon_262 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Auto = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Daylight = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Cloudy = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Tangsten = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Fluorescent = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Strobe = 5 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_WhitePaper = 6 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Shade = 8 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_ColorTemp = 9 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_PCSet1 = 10 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_PCSet2 = 11 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_PCSet3 = 12 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_WhitePaper2 = 15 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_WhitePaper3 = 16 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_WhitePaper4 = 18 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_WhitePaper5 = 19 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_PCSet4 = 20 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_PCSet5 = 21 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Click = (-1) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

kEdsWhiteBalance_Pasted = (-2) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

EdsWhiteBalance = enum_anon_262 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 861

enum_anon_263 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 871

kEdsPhotoEffect_Off = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 871

kEdsPhotoEffect_Monochrome = 5 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 871

EdsPhotoEffect = enum_anon_263 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 871

enum_anon_264 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

kEdsColorMatrix_Custom = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

kEdsColorMatrix_1 = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

kEdsColorMatrix_2 = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

kEdsColorMatrix_3 = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

kEdsColorMatrix_4 = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

kEdsColorMatrix_5 = 5 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

kEdsColorMatrix_6 = 6 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

kEdsColorMatrix_7 = 7 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

EdsColorMatrix = enum_anon_264 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 887

enum_anon_265 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 900

kEdsFilterEffect_None = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 900

kEdsFilterEffect_Yellow = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 900

kEdsFilterEffect_Orange = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 900

kEdsFilterEffect_Red = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 900

kEdsFilterEffect_Green = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 900

EdsFilterEffect = enum_anon_265 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 900

enum_anon_266 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 913

kEdsTonigEffect_None = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 913

kEdsTonigEffect_Sepia = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 913

kEdsTonigEffect_Blue = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 913

kEdsTonigEffect_Purple = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 913

kEdsTonigEffect_Green = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 913

EdsTonigEffect = enum_anon_266 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 913

enum_anon_267 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 924

kEdsColorSpace_sRGB = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 924

kEdsColorSpace_AdobeRGB = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 924

kEdsColorSpace_Unknown = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 924

EdsColorSpace = enum_anon_267 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 924

enum_anon_268 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_Standard = 129 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_Portrait = 130 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_Landscape = 131 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_Neutral = 132 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_Faithful = 133 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_Monochrome = 134 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_Auto = 135 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_User1 = 33 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_User2 = 34 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_User3 = 35 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_PC1 = 65 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_PC2 = 66 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

kEdsPictureStyle_PC3 = 67 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

EdsPictureStyle = enum_anon_268 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 945

enum_anon_269 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 956

kEdsTransferOption_ByDirectTransfer = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 956

kEdsTransferOption_ByRelease = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 956

kEdsTransferOption_ToDesktop = 256 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 956

EdsTransferOption = enum_anon_269 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 956

enum_anon_270 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Program = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Tv = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Av = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Manual = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Bulb = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_A_DEP = 5 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_DEP = 6 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Custom = 7 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Lock = 8 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Green = 9 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_NightPortrait = 10 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Sports = 11 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Portrait = 12 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Landscape = 13 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Closeup = 14 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_FlashOff = 15 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_CreativeAuto = 19 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Movie = 20 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_PhotoInMovie = 21 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

kEdsAEMode_Unknown = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

EdsAEMode = enum_anon_270 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 984

enum_anon_271 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 997

kEdsBracket_AEB = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 997

kEdsBracket_ISOB = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 997

kEdsBracket_WBB = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 997

kEdsBracket_FEB = 8 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 997

kEdsBracket_Unknown = 4294967295L # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 997

EdsBracket = enum_anon_271 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 997

enum_anon_272 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1006

kEdsEvfOutputDevice_TFT = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1006

kEdsEvfOutputDevice_PC = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1006

EdsEvfOutputDevice = enum_anon_272 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1006

enum_anon_273 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1016

kEdsEvfZoom_Fit = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1016

kEdsEvfZoom_x5 = 5 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1016

kEdsEvfZoom_x10 = 10 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1016

EdsEvfZoom = enum_anon_273 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1016

enum_anon_274 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1026

Evf_AFMode_Quick = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1026

Evf_AFMode_Live = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1026

Evf_AFMode_LiveFace = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1026

EdsEvfAFMode = enum_anon_274 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1026

enum_anon_275 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

kEdsStroboModeInternal = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

kEdsStroboModeExternalETTL = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

kEdsStroboModeExternalATTL = 2 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

kEdsStroboModeExternalTTL = 3 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

kEdsStroboModeExternalAuto = 4 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

kEdsStroboModeExternalManual = 5 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

kEdsStroboModeManual = 6 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

EdsStroboMode = enum_anon_275 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1040

enum_anon_276 = c_int # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1049

kEdsETTL2ModeEvaluative = 0 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1049

kEdsETTL2ModeAverage = 1 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1049

EdsETTL2Mode = enum_anon_276 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1049

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1062
class struct_tagEdsPoint(Structure):
    pass

struct_tagEdsPoint.__slots__ = [
    'x',
    'y',
]
struct_tagEdsPoint._fields_ = [
    ('x', EdsInt32),
    ('y', EdsInt32),
]

EdsPoint = struct_tagEdsPoint # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1062

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1072
class struct_tagEdsSize(Structure):
    pass

struct_tagEdsSize.__slots__ = [
    'width',
    'height',
]
struct_tagEdsSize._fields_ = [
    ('width', EdsInt32),
    ('height', EdsInt32),
]

EdsSize = struct_tagEdsSize # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1072

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1082
class struct_tagEdsRect(Structure):
    pass

struct_tagEdsRect.__slots__ = [
    'point',
    'size',
]
struct_tagEdsRect._fields_ = [
    ('point', EdsPoint),
    ('size', EdsSize),
]

EdsRect = struct_tagEdsRect # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1082

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1091
class struct_tagEdsRational(Structure):
    pass

struct_tagEdsRational.__slots__ = [
    'numerator',
    'denominator',
]
struct_tagEdsRational._fields_ = [
    ('numerator', EdsInt32),
    ('denominator', EdsUInt32),
]

EdsRational = struct_tagEdsRational # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1091

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1106
class struct_tagEdsTime(Structure):
    pass

struct_tagEdsTime.__slots__ = [
    'year',
    'month',
    'day',
    'hour',
    'minute',
    'second',
    'milliseconds',
]
struct_tagEdsTime._fields_ = [
    ('year', EdsUInt32),
    ('month', EdsUInt32),
    ('day', EdsUInt32),
    ('hour', EdsUInt32),
    ('minute', EdsUInt32),
    ('second', EdsUInt32),
    ('milliseconds', EdsUInt32),
]

EdsTime = struct_tagEdsTime # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1106

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1117
class struct_tagEdsDeviceInfo(Structure):
    pass

struct_tagEdsDeviceInfo.__slots__ = [
    'szPortName',
    'szDeviceDescription',
    'deviceSubType',
    'reserved',
]
struct_tagEdsDeviceInfo._fields_ = [
    ('szPortName', EdsChar * 256),
    ('szDeviceDescription', EdsChar * 256),
    ('deviceSubType', EdsUInt32),
    ('reserved', EdsUInt32),
]

EdsDeviceInfo = struct_tagEdsDeviceInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1117

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1130
class struct_tagEdsVolumeInfo(Structure):
    pass

struct_tagEdsVolumeInfo.__slots__ = [
    'storageType',
    'access',
    'maxCapacity',
    'freeSpaceInBytes',
    'szVolumeLabel',
]
struct_tagEdsVolumeInfo._fields_ = [
    ('storageType', EdsUInt32),
    ('access', EdsAccess),
    ('maxCapacity', EdsUInt64),
    ('freeSpaceInBytes', EdsUInt64),
    ('szVolumeLabel', EdsChar * 256),
]

EdsVolumeInfo = struct_tagEdsVolumeInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1130

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1145
class struct_tagEdsDirectoryItemInfo(Structure):
    pass

struct_tagEdsDirectoryItemInfo.__slots__ = [
    'size',
    'isFolder',
    'groupID',
    'option',
    'szFileName',
    'format',
]
struct_tagEdsDirectoryItemInfo._fields_ = [
    ('size', EdsUInt32),
    ('isFolder', EdsBool),
    ('groupID', EdsUInt32),
    ('option', EdsUInt32),
    ('szFileName', EdsChar * 256),
    ('format', EdsUInt32),
]

EdsDirectoryItemInfo = struct_tagEdsDirectoryItemInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1145

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1160
class struct_tagEdsImageInfo(Structure):
    pass

struct_tagEdsImageInfo.__slots__ = [
    'width',
    'height',
    'numOfComponents',
    'componentDepth',
    'effectiveRect',
    'reserved1',
    'reserved2',
]
struct_tagEdsImageInfo._fields_ = [
    ('width', EdsUInt32),
    ('height', EdsUInt32),
    ('numOfComponents', EdsUInt32),
    ('componentDepth', EdsUInt32),
    ('effectiveRect', EdsRect),
    ('reserved1', EdsUInt32),
    ('reserved2', EdsUInt32),
]

EdsImageInfo = struct_tagEdsImageInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1160

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1171
class struct_tagEdsSaveImageSetting(Structure):
    pass

struct_tagEdsSaveImageSetting.__slots__ = [
    'JPEGQuality',
    'iccProfileStream',
    'reserved',
]
struct_tagEdsSaveImageSetting._fields_ = [
    ('JPEGQuality', EdsUInt32),
    ('iccProfileStream', EdsStreamRef),
    ('reserved', EdsUInt32),
]

EdsSaveImageSetting = struct_tagEdsSaveImageSetting # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1171

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1183
class struct_tagEdsPropertyDesc(Structure):
    pass

struct_tagEdsPropertyDesc.__slots__ = [
    'form',
    'access',
    'numElements',
    'propDesc',
]
struct_tagEdsPropertyDesc._fields_ = [
    ('form', EdsInt32),
    ('access', EdsInt32),
    ('numElements', EdsInt32),
    ('propDesc', EdsInt32 * 128),
]

EdsPropertyDesc = struct_tagEdsPropertyDesc # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1183

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1197
class struct_tagEdsPictureStyleDesc(Structure):
    pass

struct_tagEdsPictureStyleDesc.__slots__ = [
    'contrast',
    'sharpness',
    'saturation',
    'colorTone',
    'filterEffect',
    'toningEffect',
]
struct_tagEdsPictureStyleDesc._fields_ = [
    ('contrast', EdsInt32),
    ('sharpness', EdsUInt32),
    ('saturation', EdsInt32),
    ('colorTone', EdsInt32),
    ('filterEffect', EdsUInt32),
    ('toningEffect', EdsUInt32),
]

EdsPictureStyleDesc = struct_tagEdsPictureStyleDesc # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1197

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1210
class struct_tagEdsFrameDesc(Structure):
    pass

struct_tagEdsFrameDesc.__slots__ = [
    'valid',
    'selected',
    'justFocus',
    'rect',
    'reserved',
]
struct_tagEdsFrameDesc._fields_ = [
    ('valid', EdsUInt32),
    ('selected', EdsUInt32),
    ('justFocus', EdsUInt32),
    ('rect', EdsRect),
    ('reserved', EdsUInt32),
]

EdsFocusPoint = struct_tagEdsFrameDesc # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1210

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1219
class struct_tagEdsFocusInfo(Structure):
    pass

struct_tagEdsFocusInfo.__slots__ = [
    'imageRect',
    'pointNumber',
    'focusPoint',
    'executeMode',
]
struct_tagEdsFocusInfo._fields_ = [
    ('imageRect', EdsRect),
    ('pointNumber', EdsUInt32),
    ('focusPoint', EdsFocusPoint * 128),
    ('executeMode', EdsUInt32),
]

EdsFocusInfo = struct_tagEdsFocusInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1219

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1231
class struct_tagEdsUsersetData(Structure):
    pass

struct_tagEdsUsersetData.__slots__ = [
    'valid',
    'dataSize',
    'szCaption',
    'data',
]
struct_tagEdsUsersetData._fields_ = [
    ('valid', EdsUInt32),
    ('dataSize', EdsUInt32),
    ('szCaption', EdsChar * 32),
    ('data', EdsUInt8 * 1),
]

EdsUsersetData = struct_tagEdsUsersetData # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1231

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1242
class struct_tagEdsCapacity(Structure):
    pass

struct_tagEdsCapacity.__slots__ = [
    'numberOfFreeClusters',
    'bytesPerSector',
    'reset',
]
struct_tagEdsCapacity._fields_ = [
    ('numberOfFreeClusters', EdsInt32),
    ('bytesPerSector', EdsInt32),
    ('reset', EdsBool),
]

EdsCapacity = struct_tagEdsCapacity # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1242

EdsProgressCallback = CFUNCTYPE(UNCHECKED(EdsError), EdsUInt32, POINTER(EdsVoid), POINTER(EdsBool)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1251

EdsCameraAddedHandler = CFUNCTYPE(UNCHECKED(EdsError), POINTER(EdsVoid)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1259

EdsPropertyEventHandler = CFUNCTYPE(UNCHECKED(EdsError), EdsPropertyEvent, EdsPropertyID, EdsUInt32, POINTER(EdsVoid)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1265

EdsObjectEventHandler = CFUNCTYPE(UNCHECKED(EdsError), EdsObjectEvent, EdsBaseRef, POINTER(EdsVoid)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1274

EdsStateEventHandler = CFUNCTYPE(UNCHECKED(EdsError), EdsStateEvent, EdsUInt32, POINTER(EdsVoid)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1282

EdsReadStream = CFUNCTYPE(UNCHECKED(EdsError), POINTER(None), EdsUInt32, POINTER(EdsVoid), POINTER(EdsUInt32)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1289

EdsWriteStream = CFUNCTYPE(UNCHECKED(EdsError), POINTER(None), EdsUInt32, POINTER(EdsVoid), POINTER(EdsUInt32)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1290

EdsSeekStream = CFUNCTYPE(UNCHECKED(EdsError), POINTER(None), EdsInt32, EdsSeekOrigin) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1291

EdsTellStream = CFUNCTYPE(UNCHECKED(EdsError), POINTER(None), POINTER(EdsInt32)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1292

EdsGetStreamLength = CFUNCTYPE(UNCHECKED(EdsError), POINTER(None), POINTER(EdsUInt32)) # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1293

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1304
class struct_anon_277(Structure):
    pass

struct_anon_277.__slots__ = [
    'context',
    'read',
    'write',
    'seek',
    'tell',
    'getLength',
]
struct_anon_277._fields_ = [
    ('context', POINTER(None)),
    ('read', POINTER(EdsReadStream)),
    ('write', POINTER(EdsWriteStream)),
    ('seek', POINTER(EdsSeekStream)),
    ('tell', POINTER(EdsTellStream)),
    ('getLength', POINTER(EdsGetStreamLength)),
]

EdsIStream = struct_anon_277 # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1304

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 76
if hasattr(_libs['EDSDK'], 'EdsInitializeSDK'):
    EdsInitializeSDK = _libs['EDSDK'].EdsInitializeSDK
    EdsInitializeSDK.argtypes = []
    EdsInitializeSDK.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 93
if hasattr(_libs['EDSDK'], 'EdsTerminateSDK'):
    EdsTerminateSDK = _libs['EDSDK'].EdsTerminateSDK
    EdsTerminateSDK.argtypes = []
    EdsTerminateSDK.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 118
if hasattr(_libs['EDSDK'], 'EdsRetain'):
    EdsRetain = _libs['EDSDK'].EdsRetain
    EdsRetain.argtypes = [EdsBaseRef]
    EdsRetain.restype = EdsUInt32

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 134
if hasattr(_libs['EDSDK'], 'EdsRelease'):
    EdsRelease = _libs['EDSDK'].EdsRelease
    EdsRelease.argtypes = [EdsBaseRef]
    EdsRelease.restype = EdsUInt32

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 160
if hasattr(_libs['EDSDK'], 'EdsGetChildCount'):
    EdsGetChildCount = _libs['EDSDK'].EdsGetChildCount
    EdsGetChildCount.argtypes = [EdsBaseRef, POINTER(EdsUInt32)]
    EdsGetChildCount.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 179
if hasattr(_libs['EDSDK'], 'EdsGetChildAtIndex'):
    EdsGetChildAtIndex = _libs['EDSDK'].EdsGetChildAtIndex
    EdsGetChildAtIndex.argtypes = [EdsBaseRef, EdsInt32, POINTER(EdsBaseRef)]
    EdsGetChildAtIndex.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 197
if hasattr(_libs['EDSDK'], 'EdsGetParent'):
    EdsGetParent = _libs['EDSDK'].EdsGetParent
    EdsGetParent.argtypes = [EdsBaseRef, POINTER(EdsBaseRef)]
    EdsGetParent.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 231
if hasattr(_libs['EDSDK'], 'EdsGetPropertySize'):
    EdsGetPropertySize = _libs['EDSDK'].EdsGetPropertySize
    EdsGetPropertySize.argtypes = [EdsBaseRef, EdsPropertyID, EdsInt32, POINTER(EdsDataType), POINTER(EdsUInt32)]
    EdsGetPropertySize.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 257
if hasattr(_libs['EDSDK'], 'EdsGetPropertyData'):
    EdsGetPropertyData = _libs['EDSDK'].EdsGetPropertyData
    EdsGetPropertyData.argtypes = [EdsBaseRef, EdsPropertyID, EdsInt32, EdsUInt32, POINTER(EdsVoid)]
    EdsGetPropertyData.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 282
if hasattr(_libs['EDSDK'], 'EdsSetPropertyData'):
    EdsSetPropertyData = _libs['EDSDK'].EdsSetPropertyData
    EdsSetPropertyData.argtypes = [EdsBaseRef, EdsPropertyID, EdsInt32, EdsUInt32, POINTER(EdsVoid)]
    EdsSetPropertyData.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 305
if hasattr(_libs['EDSDK'], 'EdsGetPropertyDesc'):
    EdsGetPropertyDesc = _libs['EDSDK'].EdsGetPropertyDesc
    EdsGetPropertyDesc.argtypes = [EdsBaseRef, EdsPropertyID, POINTER(EdsPropertyDesc)]
    EdsGetPropertyDesc.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 332
if hasattr(_libs['EDSDK'], 'EdsGetCameraList'):
    EdsGetCameraList = _libs['EDSDK'].EdsGetCameraList
    EdsGetCameraList.argtypes = [POINTER(EdsCameraListRef)]
    EdsGetCameraList.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 361
if hasattr(_libs['EDSDK'], 'EdsGetDeviceInfo'):
    EdsGetDeviceInfo = _libs['EDSDK'].EdsGetDeviceInfo
    EdsGetDeviceInfo.argtypes = [EdsCameraRef, POINTER(EdsDeviceInfo)]
    EdsGetDeviceInfo.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 379
if hasattr(_libs['EDSDK'], 'EdsOpenSession'):
    EdsOpenSession = _libs['EDSDK'].EdsOpenSession
    EdsOpenSession.argtypes = [EdsCameraRef]
    EdsOpenSession.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 395
if hasattr(_libs['EDSDK'], 'EdsCloseSession'):
    EdsCloseSession = _libs['EDSDK'].EdsCloseSession
    EdsCloseSession.argtypes = [EdsCameraRef]
    EdsCloseSession.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 414
if hasattr(_libs['EDSDK'], 'EdsSendCommand'):
    EdsSendCommand = _libs['EDSDK'].EdsSendCommand
    EdsSendCommand.argtypes = [EdsCameraRef, EdsCameraCommand, EdsInt32]
    EdsSendCommand.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 435
if hasattr(_libs['EDSDK'], 'EdsSendStatusCommand'):
    EdsSendStatusCommand = _libs['EDSDK'].EdsSendStatusCommand
    EdsSendStatusCommand.argtypes = [EdsCameraRef, EdsCameraStatusCommand, EdsInt32]
    EdsSendStatusCommand.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 466
if hasattr(_libs['EDSDK'], 'EdsSetCapacity'):
    EdsSetCapacity = _libs['EDSDK'].EdsSetCapacity
    EdsSetCapacity.argtypes = [EdsCameraRef, EdsCapacity]
    EdsSetCapacity.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 492
if hasattr(_libs['EDSDK'], 'EdsGetVolumeInfo'):
    EdsGetVolumeInfo = _libs['EDSDK'].EdsGetVolumeInfo
    EdsGetVolumeInfo.argtypes = [EdsVolumeRef, POINTER(EdsVolumeInfo)]
    EdsGetVolumeInfo.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 508
if hasattr(_libs['EDSDK'], 'EdsFormatVolume'):
    EdsFormatVolume = _libs['EDSDK'].EdsFormatVolume
    EdsFormatVolume.argtypes = [EdsVolumeRef]
    EdsFormatVolume.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 534
if hasattr(_libs['EDSDK'], 'EdsGetDirectoryItemInfo'):
    EdsGetDirectoryItemInfo = _libs['EDSDK'].EdsGetDirectoryItemInfo
    EdsGetDirectoryItemInfo.argtypes = [EdsDirectoryItemRef, POINTER(EdsDirectoryItemInfo)]
    EdsGetDirectoryItemInfo.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 556
if hasattr(_libs['EDSDK'], 'EdsDeleteDirectoryItem'):
    EdsDeleteDirectoryItem = _libs['EDSDK'].EdsDeleteDirectoryItem
    EdsDeleteDirectoryItem.argtypes = [EdsDirectoryItemRef]
    EdsDeleteDirectoryItem.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 579
if hasattr(_libs['EDSDK'], 'EdsDownload'):
    EdsDownload = _libs['EDSDK'].EdsDownload
    EdsDownload.argtypes = [EdsDirectoryItemRef, EdsUInt32, EdsStreamRef]
    EdsDownload.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 599
if hasattr(_libs['EDSDK'], 'EdsDownloadCancel'):
    EdsDownloadCancel = _libs['EDSDK'].EdsDownloadCancel
    EdsDownloadCancel.argtypes = [EdsDirectoryItemRef]
    EdsDownloadCancel.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 619
if hasattr(_libs['EDSDK'], 'EdsDownloadComplete'):
    EdsDownloadComplete = _libs['EDSDK'].EdsDownloadComplete
    EdsDownloadComplete.argtypes = [EdsDirectoryItemRef]
    EdsDownloadComplete.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 639
if hasattr(_libs['EDSDK'], 'EdsDownloadThumbnail'):
    EdsDownloadThumbnail = _libs['EDSDK'].EdsDownloadThumbnail
    EdsDownloadThumbnail.argtypes = [EdsDirectoryItemRef, EdsStreamRef]
    EdsDownloadThumbnail.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 660
if hasattr(_libs['EDSDK'], 'EdsGetAttribute'):
    EdsGetAttribute = _libs['EDSDK'].EdsGetAttribute
    EdsGetAttribute.argtypes = [EdsDirectoryItemRef, POINTER(EdsFileAttributes)]
    EdsGetAttribute.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 680
if hasattr(_libs['EDSDK'], 'EdsSetAttribute'):
    EdsSetAttribute = _libs['EDSDK'].EdsSetAttribute
    EdsSetAttribute.argtypes = [EdsDirectoryItemRef, EdsFileAttributes]
    EdsSetAttribute.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 715
if hasattr(_libs['EDSDK'], 'EdsCreateFileStream'):
    EdsCreateFileStream = _libs['EDSDK'].EdsCreateFileStream
    EdsCreateFileStream.argtypes = [POINTER(EdsChar), EdsFileCreateDisposition, EdsAccess, POINTER(EdsStreamRef)]
    EdsCreateFileStream.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 737
if hasattr(_libs['EDSDK'], 'EdsCreateMemoryStream'):
    EdsCreateMemoryStream = _libs['EDSDK'].EdsCreateMemoryStream
    EdsCreateMemoryStream.argtypes = [EdsUInt32, POINTER(EdsStreamRef)]
    EdsCreateMemoryStream.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 761
if hasattr(_libs['EDSDK'], 'EdsCreateFileStreamEx'):
    EdsCreateFileStreamEx = _libs['EDSDK'].EdsCreateFileStreamEx
    EdsCreateFileStreamEx.argtypes = [POINTER(WCHAR), EdsFileCreateDisposition, EdsAccess, POINTER(EdsStreamRef)]
    EdsCreateFileStreamEx.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 789
if hasattr(_libs['EDSDK'], 'EdsCreateMemoryStreamFromPointer'):
    EdsCreateMemoryStreamFromPointer = _libs['EDSDK'].EdsCreateMemoryStreamFromPointer
    EdsCreateMemoryStreamFromPointer.argtypes = [POINTER(EdsVoid), EdsUInt32, POINTER(EdsStreamRef)]
    EdsCreateMemoryStreamFromPointer.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 816
if hasattr(_libs['EDSDK'], 'EdsGetPointer'):
    EdsGetPointer = _libs['EDSDK'].EdsGetPointer
    EdsGetPointer.argtypes = [EdsStreamRef, POINTER(POINTER(EdsVoid))]
    EdsGetPointer.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 839
if hasattr(_libs['EDSDK'], 'EdsRead'):
    EdsRead = _libs['EDSDK'].EdsRead
    EdsRead.argtypes = [EdsStreamRef, EdsUInt32, POINTER(EdsVoid), POINTER(EdsUInt32)]
    EdsRead.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 863
if hasattr(_libs['EDSDK'], 'EdsWrite'):
    EdsWrite = _libs['EDSDK'].EdsWrite
    EdsWrite.argtypes = [EdsStreamRef, EdsUInt32, POINTER(EdsVoid), POINTER(EdsUInt32)]
    EdsWrite.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 892
if hasattr(_libs['EDSDK'], 'EdsSeek'):
    EdsSeek = _libs['EDSDK'].EdsSeek
    EdsSeek.argtypes = [EdsStreamRef, EdsInt32, EdsSeekOrigin]
    EdsSeek.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 912
if hasattr(_libs['EDSDK'], 'EdsGetPosition'):
    EdsGetPosition = _libs['EDSDK'].EdsGetPosition
    EdsGetPosition.argtypes = [EdsStreamRef, POINTER(EdsUInt32)]
    EdsGetPosition.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 930
if hasattr(_libs['EDSDK'], 'EdsGetLength'):
    EdsGetLength = _libs['EDSDK'].EdsGetLength
    EdsGetLength.argtypes = [EdsStreamRef, POINTER(EdsUInt32)]
    EdsGetLength.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 954
if hasattr(_libs['EDSDK'], 'EdsCopyData'):
    EdsCopyData = _libs['EDSDK'].EdsCopyData
    EdsCopyData.argtypes = [EdsStreamRef, EdsUInt32, EdsStreamRef]
    EdsCopyData.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 990
if hasattr(_libs['EDSDK'], 'EdsSetProgressCallback'):
    EdsSetProgressCallback = _libs['EDSDK'].EdsSetProgressCallback
    EdsSetProgressCallback.argtypes = [EdsBaseRef, EdsProgressCallback, EdsProgressOption, POINTER(EdsVoid)]
    EdsSetProgressCallback.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1026
if hasattr(_libs['EDSDK'], 'EdsCreateImageRef'):
    EdsCreateImageRef = _libs['EDSDK'].EdsCreateImageRef
    EdsCreateImageRef.argtypes = [EdsStreamRef, POINTER(EdsImageRef)]
    EdsCreateImageRef.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1061
if hasattr(_libs['EDSDK'], 'EdsGetImageInfo'):
    EdsGetImageInfo = _libs['EDSDK'].EdsGetImageInfo
    EdsGetImageInfo.argtypes = [EdsImageRef, EdsImageSource, POINTER(EdsImageInfo)]
    EdsGetImageInfo.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1106
if hasattr(_libs['EDSDK'], 'EdsGetImage'):
    EdsGetImage = _libs['EDSDK'].EdsGetImage
    EdsGetImage.argtypes = [EdsImageRef, EdsImageSource, EdsTargetImageType, EdsRect, EdsSize, EdsStreamRef]
    EdsGetImage.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1139
if hasattr(_libs['EDSDK'], 'EdsSaveImage'):
    EdsSaveImage = _libs['EDSDK'].EdsSaveImage
    EdsSaveImage.argtypes = [EdsImageRef, EdsTargetImageType, EdsSaveImageSetting, EdsStreamRef]
    EdsSaveImage.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1165
if hasattr(_libs['EDSDK'], 'EdsCacheImage'):
    EdsCacheImage = _libs['EDSDK'].EdsCacheImage
    EdsCacheImage.argtypes = [EdsImageRef, EdsBool]
    EdsCacheImage.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1182
if hasattr(_libs['EDSDK'], 'EdsReflectImageProperty'):
    EdsReflectImageProperty = _libs['EDSDK'].EdsReflectImageProperty
    EdsReflectImageProperty.argtypes = [EdsImageRef]
    EdsReflectImageProperty.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1198
if hasattr(_libs['EDSDK'], 'EdsCreateEvfImageRef'):
    EdsCreateEvfImageRef = _libs['EDSDK'].EdsCreateEvfImageRef
    EdsCreateEvfImageRef.argtypes = [EdsStreamRef, POINTER(EdsEvfImageRef)]
    EdsCreateEvfImageRef.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1223
if hasattr(_libs['EDSDK'], 'EdsDownloadEvfImage'):
    EdsDownloadEvfImage = _libs['EDSDK'].EdsDownloadEvfImage
    EdsDownloadEvfImage.argtypes = [EdsCameraRef, EdsEvfImageRef]
    EdsDownloadEvfImage.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1253
if hasattr(_libs['EDSDK'], 'EdsSetCameraAddedHandler'):
    EdsSetCameraAddedHandler = _libs['EDSDK'].EdsSetCameraAddedHandler
    EdsSetCameraAddedHandler.argtypes = [EdsCameraAddedHandler, POINTER(EdsVoid)]
    EdsSetCameraAddedHandler.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1278
if hasattr(_libs['EDSDK'], 'EdsSetPropertyEventHandler'):
    EdsSetPropertyEventHandler = _libs['EDSDK'].EdsSetPropertyEventHandler
    EdsSetPropertyEventHandler.argtypes = [EdsCameraRef, EdsPropertyEvent, EdsPropertyEventHandler, POINTER(EdsVoid)]
    EdsSetPropertyEventHandler.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1307
if hasattr(_libs['EDSDK'], 'EdsSetObjectEventHandler'):
    EdsSetObjectEventHandler = _libs['EDSDK'].EdsSetObjectEventHandler
    EdsSetObjectEventHandler.argtypes = [EdsCameraRef, EdsObjectEvent, EdsObjectEventHandler, POINTER(EdsVoid)]
    EdsSetObjectEventHandler.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1335
if hasattr(_libs['EDSDK'], 'EdsSetCameraStateEventHandler'):
    EdsSetCameraStateEventHandler = _libs['EDSDK'].EdsSetCameraStateEventHandler
    EdsSetCameraStateEventHandler.argtypes = [EdsCameraRef, EdsStateEvent, EdsStateEventHandler, POINTER(EdsVoid)]
    EdsSetCameraStateEventHandler.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1343
if hasattr(_libs['EDSDK'], 'EdsCreateStream'):
    EdsCreateStream = _libs['EDSDK'].EdsCreateStream
    EdsCreateStream.argtypes = [POINTER(EdsIStream), POINTER(EdsStreamRef)]
    EdsCreateStream.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 1360
if hasattr(_libs['EDSDK'], 'EdsGetEvent'):
    EdsGetEvent = _libs['EDSDK'].EdsGetEvent
    EdsGetEvent.argtypes = []
    EdsGetEvent.restype = EdsError

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 51
try:
    EDS_MAX_NAME = 256
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 52
try:
    EDS_TRANSFER_BLOCK_SIZE = 512
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 174
try:
    kEdsPropID_Unknown = 65535
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 176
try:
    kEdsPropID_ProductName = 2
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 177
try:
    kEdsPropID_OwnerName = 4
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 178
try:
    kEdsPropID_MakerName = 5
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 179
try:
    kEdsPropID_DateTime = 6
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 180
try:
    kEdsPropID_FirmwareVersion = 7
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 181
try:
    kEdsPropID_BatteryLevel = 8
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 182
try:
    kEdsPropID_CFn = 9
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 183
try:
    kEdsPropID_SaveTo = 11
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 184
try:
    kEdsPropID_CurrentStorage = 12
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 185
try:
    kEdsPropID_CurrentFolder = 13
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 186
try:
    kEdsPropID_MyMenu = 14
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 188
try:
    kEdsPropID_BatteryQuality = 16
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 190
try:
    kEdsPropID_BodyIDEx = 21
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 191
try:
    kEdsPropID_HDDirectoryStructure = 32
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 197
try:
    kEdsPropID_ImageQuality = 256
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 198
try:
    kEdsPropID_JpegQuality = 257
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 199
try:
    kEdsPropID_Orientation = 258
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 200
try:
    kEdsPropID_ICCProfile = 259
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 201
try:
    kEdsPropID_FocusInfo = 260
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 202
try:
    kEdsPropID_DigitalExposure = 261
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 203
try:
    kEdsPropID_WhiteBalance = 262
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 204
try:
    kEdsPropID_ColorTemperature = 263
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 205
try:
    kEdsPropID_WhiteBalanceShift = 264
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 206
try:
    kEdsPropID_Contrast = 265
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 207
try:
    kEdsPropID_ColorSaturation = 266
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 208
try:
    kEdsPropID_ColorTone = 267
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 209
try:
    kEdsPropID_Sharpness = 268
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 210
try:
    kEdsPropID_ColorSpace = 269
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 211
try:
    kEdsPropID_ToneCurve = 270
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 212
try:
    kEdsPropID_PhotoEffect = 271
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 213
try:
    kEdsPropID_FilterEffect = 272
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 214
try:
    kEdsPropID_ToningEffect = 273
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 215
try:
    kEdsPropID_ParameterSet = 274
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 216
try:
    kEdsPropID_ColorMatrix = 275
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 217
try:
    kEdsPropID_PictureStyle = 276
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 218
try:
    kEdsPropID_PictureStyleDesc = 277
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 219
try:
    kEdsPropID_PictureStyleCaption = 512
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 224
try:
    kEdsPropID_Linear = 768
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 225
try:
    kEdsPropID_ClickWBPoint = 769
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 226
try:
    kEdsPropID_WBCoeffs = 770
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 232
try:
    kEdsPropID_GPSVersionID = 2048
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 233
try:
    kEdsPropID_GPSLatitudeRef = 2049
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 234
try:
    kEdsPropID_GPSLatitude = 2050
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 235
try:
    kEdsPropID_GPSLongitudeRef = 2051
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 236
try:
    kEdsPropID_GPSLongitude = 2052
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 237
try:
    kEdsPropID_GPSAltitudeRef = 2053
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 238
try:
    kEdsPropID_GPSAltitude = 2054
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 239
try:
    kEdsPropID_GPSTimeStamp = 2055
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 240
try:
    kEdsPropID_GPSSatellites = 2056
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 241
try:
    kEdsPropID_GPSStatus = 2057
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 242
try:
    kEdsPropID_GPSMapDatum = 2066
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 243
try:
    kEdsPropID_GPSDateStamp = 2077
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 249
try:
    kEdsPropID_AtCapture_Flag = 2147483648L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 255
try:
    kEdsPropID_AEMode = 1024
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 256
try:
    kEdsPropID_DriveMode = 1025
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 257
try:
    kEdsPropID_ISOSpeed = 1026
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 258
try:
    kEdsPropID_MeteringMode = 1027
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 259
try:
    kEdsPropID_AFMode = 1028
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 260
try:
    kEdsPropID_Av = 1029
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 261
try:
    kEdsPropID_Tv = 1030
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 262
try:
    kEdsPropID_ExposureCompensation = 1031
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 263
try:
    kEdsPropID_FlashCompensation = 1032
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 264
try:
    kEdsPropID_FocalLength = 1033
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 265
try:
    kEdsPropID_AvailableShots = 1034
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 266
try:
    kEdsPropID_Bracket = 1035
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 267
try:
    kEdsPropID_WhiteBalanceBracket = 1036
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 268
try:
    kEdsPropID_LensName = 1037
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 269
try:
    kEdsPropID_AEBracket = 1038
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 270
try:
    kEdsPropID_FEBracket = 1039
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 271
try:
    kEdsPropID_ISOBracket = 1040
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 272
try:
    kEdsPropID_NoiseReduction = 1041
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 273
try:
    kEdsPropID_FlashOn = 1042
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 274
try:
    kEdsPropID_RedEye = 1043
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 275
try:
    kEdsPropID_FlashMode = 1044
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 276
try:
    kEdsPropID_LensStatus = 1046
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 277
try:
    kEdsPropID_Artist = 1048
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 278
try:
    kEdsPropID_Copyright = 1049
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 279
try:
    kEdsPropID_DepthOfField = 1051
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 280
try:
    kEdsPropID_EFCompensation = 1054
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 285
try:
    kEdsPropID_Evf_OutputDevice = 1280
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 286
try:
    kEdsPropID_Evf_Mode = 1281
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 287
try:
    kEdsPropID_Evf_WhiteBalance = 1282
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 288
try:
    kEdsPropID_Evf_ColorTemperature = 1283
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 289
try:
    kEdsPropID_Evf_DepthOfFieldPreview = 1284
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 292
try:
    kEdsPropID_Evf_Zoom = 1287
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 293
try:
    kEdsPropID_Evf_ZoomPosition = 1288
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 294
try:
    kEdsPropID_Evf_FocusAid = 1289
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 295
try:
    kEdsPropID_Evf_Histogram = 1290
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 296
try:
    kEdsPropID_Evf_ImagePosition = 1291
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 297
try:
    kEdsPropID_Evf_HistogramStatus = 1292
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 298
try:
    kEdsPropID_Evf_AFMode = 1294
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 300
try:
    kEdsPropID_Evf_CoordinateSystem = 1344
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 301
try:
    kEdsPropID_Evf_ZoomRect = 1345
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 310
try:
    kEdsCameraCommand_TakePicture = 0
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 311
try:
    kEdsCameraCommand_ExtendShutDownTimer = 1
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 312
try:
    kEdsCameraCommand_BulbStart = 2
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 313
try:
    kEdsCameraCommand_BulbEnd = 3
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 314
try:
    kEdsCameraCommand_DoEvfAf = 258
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 315
try:
    kEdsCameraCommand_DriveLensEvf = 259
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 316
try:
    kEdsCameraCommand_DoClickWBEvf = 260
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 318
try:
    kEdsCameraCommand_PressShutterButton = 4
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 341
try:
    kEdsCameraStatusCommand_UILock = 0
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 342
try:
    kEdsCameraStatusCommand_UIUnLock = 1
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 343
try:
    kEdsCameraStatusCommand_EnterDirectTransfer = 2
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 344
try:
    kEdsCameraStatusCommand_ExitDirectTransfer = 3
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 355
try:
    kEdsPropertyEvent_All = 256
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 364
try:
    kEdsPropertyEvent_PropertyChanged = 257
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 376
try:
    kEdsPropertyEvent_PropertyDescChanged = 258
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 384
try:
    kEdsObjectEvent_All = 512
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 391
try:
    kEdsObjectEvent_VolumeInfoChanged = 513
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 400
try:
    kEdsObjectEvent_VolumeUpdateItems = 514
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 407
try:
    kEdsObjectEvent_FolderUpdateItems = 515
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 420
try:
    kEdsObjectEvent_DirItemCreated = 516
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 428
try:
    kEdsObjectEvent_DirItemRemoved = 517
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 434
try:
    kEdsObjectEvent_DirItemInfoChanged = 518
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 440
try:
    kEdsObjectEvent_DirItemContentChanged = 519
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 451
try:
    kEdsObjectEvent_DirItemRequestTransfer = 520
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 459
try:
    kEdsObjectEvent_DirItemRequestTransferDT = 521
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 467
try:
    kEdsObjectEvent_DirItemCancelTransferDT = 522
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 469
try:
    kEdsObjectEvent_VolumeAdded = 524
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 470
try:
    kEdsObjectEvent_VolumeRemoved = 525
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 478
try:
    kEdsStateEvent_All = 768
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 484
try:
    kEdsStateEvent_Shutdown = 769
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 492
try:
    kEdsStateEvent_JobStatusChanged = 770
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 502
try:
    kEdsStateEvent_WillSoonShutDown = 771
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 508
try:
    kEdsStateEvent_ShutDownTimerUpdate = 772
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 512
try:
    kEdsStateEvent_CaptureError = 773
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 518
try:
    kEdsStateEvent_InternalError = 774
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 521
try:
    kEdsStateEvent_AfResult = 777
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 524
try:
    kEdsStateEvent_BulbExposureTime = 784
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 34
try:
    EDS_ISSPECIFIC_MASK = 2147483648L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 35
try:
    EDS_COMPONENTID_MASK = 2130706432L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 36
try:
    EDS_RESERVED_MASK = 16711680L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 37
try:
    EDS_ERRORID_MASK = 65535L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 42
try:
    EDS_CMP_ID_CLIENT_COMPONENTID = 16777216L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 43
try:
    EDS_CMP_ID_LLSDK_COMPONENTID = 33554432L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 44
try:
    EDS_CMP_ID_HLSDK_COMPONENTID = 50331648L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 49
try:
    EDS_ERR_OK = 0L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 55
try:
    EDS_ERR_UNIMPLEMENTED = 1L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 56
try:
    EDS_ERR_INTERNAL_ERROR = 2L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 57
try:
    EDS_ERR_MEM_ALLOC_FAILED = 3L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 58
try:
    EDS_ERR_MEM_FREE_FAILED = 4L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 59
try:
    EDS_ERR_OPERATION_CANCELLED = 5L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 60
try:
    EDS_ERR_INCOMPATIBLE_VERSION = 6L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 61
try:
    EDS_ERR_NOT_SUPPORTED = 7L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 62
try:
    EDS_ERR_UNEXPECTED_EXCEPTION = 8L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 63
try:
    EDS_ERR_PROTECTION_VIOLATION = 9L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 64
try:
    EDS_ERR_MISSING_SUBCOMPONENT = 10L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 65
try:
    EDS_ERR_SELECTION_UNAVAILABLE = 11L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 68
try:
    EDS_ERR_FILE_IO_ERROR = 32L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 69
try:
    EDS_ERR_FILE_TOO_MANY_OPEN = 33L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 70
try:
    EDS_ERR_FILE_NOT_FOUND = 34L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 71
try:
    EDS_ERR_FILE_OPEN_ERROR = 35L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 72
try:
    EDS_ERR_FILE_CLOSE_ERROR = 36L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 73
try:
    EDS_ERR_FILE_SEEK_ERROR = 37L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 74
try:
    EDS_ERR_FILE_TELL_ERROR = 38L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 75
try:
    EDS_ERR_FILE_READ_ERROR = 39L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 76
try:
    EDS_ERR_FILE_WRITE_ERROR = 40L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 77
try:
    EDS_ERR_FILE_PERMISSION_ERROR = 41L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 78
try:
    EDS_ERR_FILE_DISK_FULL_ERROR = 42L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 79
try:
    EDS_ERR_FILE_ALREADY_EXISTS = 43L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 80
try:
    EDS_ERR_FILE_FORMAT_UNRECOGNIZED = 44L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 81
try:
    EDS_ERR_FILE_DATA_CORRUPT = 45L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 82
try:
    EDS_ERR_FILE_NAMING_NA = 46L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 85
try:
    EDS_ERR_DIR_NOT_FOUND = 64L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 86
try:
    EDS_ERR_DIR_IO_ERROR = 65L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 87
try:
    EDS_ERR_DIR_ENTRY_NOT_FOUND = 66L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 88
try:
    EDS_ERR_DIR_ENTRY_EXISTS = 67L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 89
try:
    EDS_ERR_DIR_NOT_EMPTY = 68L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 92
try:
    EDS_ERR_PROPERTIES_UNAVAILABLE = 80L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 93
try:
    EDS_ERR_PROPERTIES_MISMATCH = 81L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 94
try:
    EDS_ERR_PROPERTIES_NOT_LOADED = 83L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 97
try:
    EDS_ERR_INVALID_PARAMETER = 96L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 98
try:
    EDS_ERR_INVALID_HANDLE = 97L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 99
try:
    EDS_ERR_INVALID_POINTER = 98L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 100
try:
    EDS_ERR_INVALID_INDEX = 99L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 101
try:
    EDS_ERR_INVALID_LENGTH = 100L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 102
try:
    EDS_ERR_INVALID_FN_POINTER = 101L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 103
try:
    EDS_ERR_INVALID_SORT_FN = 102L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 106
try:
    EDS_ERR_DEVICE_NOT_FOUND = 128L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 107
try:
    EDS_ERR_DEVICE_BUSY = 129L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 108
try:
    EDS_ERR_DEVICE_INVALID = 130L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 109
try:
    EDS_ERR_DEVICE_EMERGENCY = 131L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 110
try:
    EDS_ERR_DEVICE_MEMORY_FULL = 132L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 111
try:
    EDS_ERR_DEVICE_INTERNAL_ERROR = 133L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 112
try:
    EDS_ERR_DEVICE_INVALID_PARAMETER = 134L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 113
try:
    EDS_ERR_DEVICE_NO_DISK = 135L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 114
try:
    EDS_ERR_DEVICE_DISK_ERROR = 136L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 115
try:
    EDS_ERR_DEVICE_CF_GATE_CHANGED = 137L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 116
try:
    EDS_ERR_DEVICE_DIAL_CHANGED = 138L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 117
try:
    EDS_ERR_DEVICE_NOT_INSTALLED = 139L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 118
try:
    EDS_ERR_DEVICE_STAY_AWAKE = 140L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 119
try:
    EDS_ERR_DEVICE_NOT_RELEASED = 141L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 122
try:
    EDS_ERR_STREAM_IO_ERROR = 160L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 123
try:
    EDS_ERR_STREAM_NOT_OPEN = 161L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 124
try:
    EDS_ERR_STREAM_ALREADY_OPEN = 162L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 125
try:
    EDS_ERR_STREAM_OPEN_ERROR = 163L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 126
try:
    EDS_ERR_STREAM_CLOSE_ERROR = 164L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 127
try:
    EDS_ERR_STREAM_SEEK_ERROR = 165L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 128
try:
    EDS_ERR_STREAM_TELL_ERROR = 166L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 129
try:
    EDS_ERR_STREAM_READ_ERROR = 167L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 130
try:
    EDS_ERR_STREAM_WRITE_ERROR = 168L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 131
try:
    EDS_ERR_STREAM_PERMISSION_ERROR = 169L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 132
try:
    EDS_ERR_STREAM_COULDNT_BEGIN_THREAD = 170L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 133
try:
    EDS_ERR_STREAM_BAD_OPTIONS = 171L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 134
try:
    EDS_ERR_STREAM_END_OF_STREAM = 172L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 137
try:
    EDS_ERR_COMM_PORT_IS_IN_USE = 192L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 138
try:
    EDS_ERR_COMM_DISCONNECTED = 193L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 139
try:
    EDS_ERR_COMM_DEVICE_INCOMPATIBLE = 194L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 140
try:
    EDS_ERR_COMM_BUFFER_FULL = 195L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 141
try:
    EDS_ERR_COMM_USB_BUS_ERR = 196L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 144
try:
    EDS_ERR_USB_DEVICE_LOCK_ERROR = 208L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 145
try:
    EDS_ERR_USB_DEVICE_UNLOCK_ERROR = 209L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 148
try:
    EDS_ERR_STI_UNKNOWN_ERROR = 224L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 149
try:
    EDS_ERR_STI_INTERNAL_ERROR = 225L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 150
try:
    EDS_ERR_STI_DEVICE_CREATE_ERROR = 226L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 151
try:
    EDS_ERR_STI_DEVICE_RELEASE_ERROR = 227L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 152
try:
    EDS_ERR_DEVICE_NOT_LAUNCHED = 228L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 154
try:
    EDS_ERR_ENUM_NA = 240L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 155
try:
    EDS_ERR_INVALID_FN_CALL = 241L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 156
try:
    EDS_ERR_HANDLE_NOT_FOUND = 242L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 157
try:
    EDS_ERR_INVALID_ID = 243L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 158
try:
    EDS_ERR_WAIT_TIMEOUT_ERROR = 244L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 161
try:
    EDS_ERR_SESSION_NOT_OPEN = 8195
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 162
try:
    EDS_ERR_INVALID_TRANSACTIONID = 8196
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 163
try:
    EDS_ERR_INCOMPLETE_TRANSFER = 8199
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 164
try:
    EDS_ERR_INVALID_STRAGEID = 8200
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 165
try:
    EDS_ERR_DEVICEPROP_NOT_SUPPORTED = 8202
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 166
try:
    EDS_ERR_INVALID_OBJECTFORMATCODE = 8203
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 167
try:
    EDS_ERR_SELF_TEST_FAILED = 8209
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 168
try:
    EDS_ERR_PARTIAL_DELETION = 8210
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 169
try:
    EDS_ERR_SPECIFICATION_BY_FORMAT_UNSUPPORTED = 8212
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 170
try:
    EDS_ERR_NO_VALID_OBJECTINFO = 8213
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 171
try:
    EDS_ERR_INVALID_CODE_FORMAT = 8214
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 172
try:
    EDS_ERR_UNKNOWN_VENDOR_CODE = 8215
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 173
try:
    EDS_ERR_CAPTURE_ALREADY_TERMINATED = 8216
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 174
try:
    EDS_ERR_INVALID_PARENTOBJECT = 8218
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 175
try:
    EDS_ERR_INVALID_DEVICEPROP_FORMAT = 8219
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 176
try:
    EDS_ERR_INVALID_DEVICEPROP_VALUE = 8220
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 177
try:
    EDS_ERR_SESSION_ALREADY_OPEN = 8222
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 178
try:
    EDS_ERR_TRANSACTION_CANCELLED = 8223
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 179
try:
    EDS_ERR_SPECIFICATION_OF_DESTINATION_UNSUPPORTED = 8224
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 182
try:
    EDS_ERR_UNKNOWN_COMMAND = 40961
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 183
try:
    EDS_ERR_OPERATION_REFUSED = 40965
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 184
try:
    EDS_ERR_LENS_COVER_CLOSE = 40966
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 185
try:
    EDS_ERR_LOW_BATTERY = 41217
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 186
try:
    EDS_ERR_OBJECT_NOTREADY = 41218
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 187
try:
    EDS_ERR_CANNOT_MAKE_OBJECT = 41220
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 190
try:
    EDS_ERR_TAKE_PICTURE_AF_NG = 36097L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 191
try:
    EDS_ERR_TAKE_PICTURE_RESERVED = 36098L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 192
try:
    EDS_ERR_TAKE_PICTURE_MIRROR_UP_NG = 36099L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 193
try:
    EDS_ERR_TAKE_PICTURE_SENSOR_CLEANING_NG = 36100L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 194
try:
    EDS_ERR_TAKE_PICTURE_SILENCE_NG = 36101L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 195
try:
    EDS_ERR_TAKE_PICTURE_NO_CARD_NG = 36102L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 196
try:
    EDS_ERR_TAKE_PICTURE_CARD_NG = 36103L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 197
try:
    EDS_ERR_TAKE_PICTURE_CARD_PROTECT_NG = 36104L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 198
try:
    EDS_ERR_TAKE_PICTURE_MOVIE_CROP_NG = 36105L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 199
try:
    EDS_ERR_TAKE_PICTURE_STROBO_CHARGE_NG = 36106L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKErrors.h: 202
try:
    EDS_ERR_LAST_GENERIC_ERROR_PLUS_ONE = 245L
except:
    pass

# C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\EDSDK.h: 46
try:
    oldif = 0
except:
    pass

__EdsObject = struct___EdsObject # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 117

tagEdsPoint = struct_tagEdsPoint # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1062

tagEdsSize = struct_tagEdsSize # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1072

tagEdsRect = struct_tagEdsRect # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1082

tagEdsRational = struct_tagEdsRational # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1091

tagEdsTime = struct_tagEdsTime # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1106

tagEdsDeviceInfo = struct_tagEdsDeviceInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1117

tagEdsVolumeInfo = struct_tagEdsVolumeInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1130

tagEdsDirectoryItemInfo = struct_tagEdsDirectoryItemInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1145

tagEdsImageInfo = struct_tagEdsImageInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1160

tagEdsSaveImageSetting = struct_tagEdsSaveImageSetting # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1171

tagEdsPropertyDesc = struct_tagEdsPropertyDesc # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1183

tagEdsPictureStyleDesc = struct_tagEdsPictureStyleDesc # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1197

tagEdsFrameDesc = struct_tagEdsFrameDesc # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1210

tagEdsFocusInfo = struct_tagEdsFocusInfo # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1219

tagEdsUsersetData = struct_tagEdsUsersetData # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1231

tagEdsCapacity = struct_tagEdsCapacity # C:\\PYME\\PYME\\Acquire\\Hardware\\CanonEOS\\/EDSDKTypes.h: 1242

# No inserted files

