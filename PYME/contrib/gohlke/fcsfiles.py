# -*- coding: utf-8 -*-
# fcsfiles.py

# Copyright (c) 2012-2014, Christoph Gohlke
# Copyright (c) 2012-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read fluorescence correlation spectroscopy (FCS) data files.

Currently only Carl Zeiss(r) ConfoCor(r) RAW and ASCII measurement data files
are supported.

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2013.01.18

Requirements
------------
* `CPython 2.7 or 3.3 <http://www.python.org>`_
* `Numpy 1.7 <http://www.numpy.org>`_

Notes
-----
"Carl Zeiss" and "ConfoCor" are registered trademarks of Carl Zeiss, Inc.

The API is not stable yet and is expected to change between revisions.

The use of this implementation may be subject to patent or license
restrictions.

This module does *not* read flow cytometry standard FCS files.

"""

from __future__ import division, print_function

import os
import struct

import numpy

__version__ = '2013.01.18'
__docformat__ = 'restructuredtext en'
__all__ = ('ConfoCor3Fcs', 'ConfoCor3Raw', 'ConfoCor2Raw')


class ConfoCor3Fcs(dict):
    """Carl Zeiss ConfoCor3 ASCII data file.

    No specification is available. The encoding is 'Windows-1252'.

    Examples
    --------
    >>> fcs = ConfoCor3Fcs('fluorescein.fcs')
    >>> fcs['FcsData']['FcsEntry'][0]['FcsDataSet']['CountRateArray'].shape
    (60000, 2)

    """
    _header = "Carl Zeiss ConfoCor3 - measurement data file - version 3.0 ANSI"

    def __init__(self, filename):
        """Read file content and parse into dictionary."""
        dict.__init__(self)
        filename = os.path.abspath(filename)
        self._filepath, self._filename = os.path.split(filename)
        with open(filename, 'U') as fh:
            # encoding = 'Windows-1252'
            header = fh.read(63)
            if header != self._header:
                raise ValueError("not a ConfoCor3 measurement data file")
            fh.readline()
            current = self
            stack = []
            array = None
            key = None
            for line in fh:
                line = line.lstrip()
                if line[0].isdigit():
                    if array:
                        array.append(line)
                    else:
                        array = [line]
                    continue
                if array:
                    shape = tuple(int(s) for s in current[key].split())
                    array = ''.join(array)
                    array = numpy.fromstring(array, dtype='float32', sep=' ')
                    array = array.reshape(*shape)
                    current[key] = array
                    array = None
                elif array == '':
                    # work around https://github.com/numpy/numpy/issues/1714
                    shape = tuple(int(s) for s in current[key].split())
                    current[key] = numpy.zeros(shape)
                    array = None
                if line.startswith('BEGIN'):
                    stack.append(current)
                    _, key, val = line.split()
                    key = key.strip()
                    val = {'_value': int(val.strip())}
                    if key[-1].isdigit():
                        key = key[:-1]
                        if key[-1].isdigit():
                            key = key[:-1]
                        if key in current:
                            current[key].append(val)
                        else:
                            current[key] = [val]
                        current = current[key][-1]
                    else:
                        current[key] = val
                        current = current[key]
                elif line.startswith('END'):
                    current = stack.pop()
                elif '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    if key.endswith('ArraySize'):
                        continue
                    if key.endswith('Array'):
                        array = ''
                    value = value.strip()
                    if value:
                        for type_t in (int, float, str):
                            try:
                                value = type_t(value)
                                break
                            except (ValueError, TypeError):
                                pass
                    if key[-1].isdigit():
                        key = key[:-1]
                        if key[-1].isdigit():
                            key = key[:-1]
                        if key in current:
                            current[key].append(value)
                        else:
                            current[key] = [value]
                    else:
                        current[key] = value

    def __str__(self):
        """Return string close to original file format."""
        result = [self._header]

        def append(key, value, indent='', index=''):
            """Recursively append formatted keys and values to result."""
            if index != '':
                index = str(index + 1)
            if isinstance(value, dict):
                result.append("%sBEGIN %s%s %i" % (
                    indent, key, index, value['_value']))
                for k, v in sorted(value.items(), key=sortkey):
                    append(k, v, indent+'\t')
                result.append("%sEND" % indent)
            elif isinstance(value, (list, tuple)):
                for i, val in enumerate(value):
                    append(key, val, indent, i)
            elif isinstance(value, numpy.ndarray):
                size = value.shape[0]
                if size != 1:
                    result.append("%s%sSize = %i" % (indent, key, size))
                result.append("%s%s = %s" % (
                    indent, key, ' '.join(str(i) for i in value.shape)))
                for i in range(size):
                    result.append("%s%s" % (
                        indent, '\t '.join('%.8f' % v for v in value[i])))
            elif key != '_value':
                result.append("%s%s%s = %s" % (indent, key, index, value))

        def sortkey(item):
            """Sort dictionary items by key string and value type."""
            key, value = item
            key = key.lower()
            if isinstance(value, (list, tuple)):
                return '~' + key
            if isinstance(value, numpy.ndarray):
                return '~~' + key
            if isinstance(value, dict):
                return '~~~' + key
            return key

        for key, val in sorted(self.items(), key=sortkey):
            append(key, val)
        return '\n'.join(result)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class ConfoCor3Raw(object):
    """Carl Zeiss ConfoCor3 raw data file.

    Based on "Confocor3 Raw Data Specification. 2 May 2007. by efg, Stowers
    Institute for Medical Research."

    Attributes
    ----------
    measurement_identifier : str
        Hex form of 4 32-bit little endian integers without leading 0.
        It is the same for all channels of the same repetition.
    measurement_position : int
        Zero based measurement position.
    kinetic_index : int
        Zero based kinetic index.
    repetition_number : int
        Zero based repetition_number.
    frequency : int
        Sampling frequency in Hz.

    Examples
    --------
    >>> fcs = ConfoCor3Raw('R1_P1_K1_Ch1.raw')
    >>> fcs.filename()
    'f5ee4f36488fca2f89cb6b8626111006_R1_P1_K1_Ch1.raw'
    >>> fcs.frequency
    20000000
    >>> times = fcs.asarray()
    >>> times[10858]
    1199925494
    >>> times, bincounts = fcs.asarray(bins=1000)
    >>> times.shape
    (1000,)
    >>> bincounts[618]
    23
    >>> fcs.close()

    """
    _header = b"Carl Zeiss ConfoCor3 - raw data file - version 3.000 - Channel"

    def __init__(self, filename):
        """Read file header."""
        filename = os.path.abspath(filename)
        self._filepath, self._filename = os.path.split(filename)
        self._fh = open(filename, 'rb')
        header = self._fh.read(64)
        if not header.startswith(self._header):
            self._fh.close()
            raise ValueError("not a ConfoCor3 raw data file")
        self.file_identifier = header
        self.channel = int(header.rsplit(b' ', 1)[-1]) - 1
        (measureid,
         self.measurement_position,
         self.kinetic_index,
         self.repetition_number,
         self.frequency,
         _) = struct.unpack('<16sIIII32s', self._fh.read(64))
        measureid = struct.unpack('<IIII', measureid)
        measureid = ''.join(hex(int(i))[2:] for i in measureid)
        self.measurement_identifier = measureid.replace('L', '')

    def filename(self):
        """Return normalized file name from file content."""
        return "%s_R%i_P%i_K%i_Ch%i.raw" % (self.measurement_identifier,
                                            self.repetition_number + 1,
                                            self.measurement_position + 1,
                                            self.kinetic_index + 1,
                                            self.channel + 1)

    def asarray(self, count=-1, skip=0, **kwargs):
        """Read data from file, perform optional binning, and return as array.

        Parameters
        ----------
        count : int (optional)
            Number of data words to process. Default: -1 (all words).
        skip : int (optional)
            Number of data words to skip at beginning of stream. Default: 0.
        kwargs : dict
            Optional argument to the `fcs_bincount` function, specifying
            size or number of bins.

        Returns
        -------
        times : ndarray
            The lower range of the bins in units of seconds, or, if ``binsize``
            is 0, the times of the events in detector clock units.
        bincounts : ndarray
            The number of events in each bin.

        """
        self._fh.seek(128 + skip*4)
        times = numpy.fromfile(self._fh, dtype='<u4', count=count)
        times = times.astype('u8')
        times = numpy.cumsum(times, out=times)
        if kwargs:
            result = fcs_bincount((times, ), self.frequency, **kwargs)
            return result[0], result[1][0]
        else:
            return times

    def __str__(self):
        """Return string with information about file."""
        return '\n'.join((
            self._filename.capitalize(),
            " (ConfoCor3 Raw Data)",
            "* Measurement identifier: %s" % self.measurement_identifier,
            "* Sampling frequency: %i Hz" % self.frequency,
            "* Repetition number %i" % self.repetition_number,
            "* Measurement position: %i" % self.measurement_position,
            "* Kinetic index: %i" % self.kinetic_index,
            "* Channel number: %i" % self.channel))

    def close(self):
        """Close open file."""
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ConfoCor2Raw(object):
    """Carl Zeiss ConfoCor2 raw data file.

    The LSM 510 META - ConfoCor 2 system: an integrated imaging and
    spectroscopic platform for single-molecule detection.
    Weisshart K, Jungel V, and Briddon SJ.
    Curr Pharm Biotechnol. 2004 Apr;5(2):135-54.

    ConfoCor 2 - Description of the Raw Data Format. 03.05.01.
    www.zeiss.at/C12567BE00472A5C/EmbedTitelIntern/RawData/$File/RawData.pdf

    Carl Zeiss Advanced Imaging Microscopy claims a patent on the file format
    and data compression scheme.

    Attributes
    ----------
    frequency : int
        Sampling frequency in Hz.

    Examples
    --------
    >>> fcs = ConfoCor2Raw('Sample488.011')
    >>> fcs.frequency
    20000000
    >>> ch0, ch1 = fcs.asarray()
    >>> ch1[4812432]
    999999833
    >>> times, ch0, ch1 = fcs.asarray(bins=1000)
    >>> times.shape
    (1000,)
    >>> ch1[428]
    10095
    >>> fcs.close()

    """
    _header = b"ConfoCor 2 - Raw data file 1.0"

    def __init__(self, filename):
        """Read file content and parse into a dictionary."""
        filename = os.path.abspath(filename)
        self._filepath, self._filename = os.path.split(filename)
        self._fh = open(filename, 'rb')
        header = self._fh.read(30)
        if not header.startswith(self._header):
            self._fh.close()
            raise ValueError("not a ConfoCor 2 raw data file")
        self.file_identifier = header
        self.channels = 2
        self.frequency = 20000000

    def asarray(self, count=-1, skip=0, **kwargs):
        """Read data from file, perform optional binning, and return as arrays.

        Parameters
        ----------
        count : int (optional)
            Number of data words to process. Default: -1 (all words).
        skip : int (optional)
            Number of data words to skip at beginning of stream. Default: 0.
        kwargs : dict
            Optional argument to the `fcs_bincount` function, specifying
            size or number of bins.

        Returns
        -------
        times : ndarray
            If binning, the lower range of the bins in units of seconds.
            Otherwise (default), the times of the events in detector clock
            units for each channel.
        Bincounts : ndarray
            If binning, the number of events in each bin for each channel.

        """
        self._fh.seek(30 + skip*2)
        data = numpy.fromfile(self._fh, dtype='u1', count=count*2)
        # accumulate clock time
        times = numpy.empty((len(data)//2, 4), dtype='u8')
        times[:, 0] = data[::2]
        times[:, 1:] = 1
        times = times.flatten()
        numpy.cumsum(times, out=times)
        # extract events for each channel
        data = numpy.repeat(data[1::2], 8).reshape(-1, 8)
        data &= numpy.array([1, 2, 4, 8, 16, 32, 64, 128], dtype='u1')
        ch0 = numpy.take(times, numpy.where(data[:, 0::2].flatten() != 0)[0])
        ch1 = numpy.take(times, numpy.where(data[:, 1::2].flatten() != 0)[0])
        del data
        del times
        if kwargs:
            result = fcs_bincount((ch0, ch1), self.frequency, **kwargs)
            return result[0], result[1][0], result[1][1]
        else:
            return ch0, ch1

    def __str__(self):
        """Return string with information about file."""
        return '\n'.join((
            self._filename.capitalize(),
            " (ConfoCor2 Raw Data)",
            "* Sampling frequency: %i Hz" % self.frequency))

    def close(self):
        """Close open file."""
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def fcs_bincount(data, frequency, binsize=None, bins=None, binspm=None):
    """Count number of events in bins.

    Parameters
    ----------
    data : tuple of ndarray
        For each channel, the times of events in detector clock units.
    frequency : int
        Sampling frequency in Hz.
    binsize : int
        Time interval, as a multiple of the sampling frequency, in which
        to bin events.
    bins : int
        If `binsize` is None, the number of bins.
    binspm : int
        If `binsize` and `bins` are None, the number of bins per minute (60s).

    Returns
    -------
    times : ndarray
        The lower range of the bins in units of seconds.
    bincounts : tuple of ndarray
        For each channel, the number of events in each bin.

    """
    assert type(data) in (tuple, list)
    if binsize is None:
        if bins is None:
            if binspm is None:
                raise ValueError('missing parameter binsize, bins, or binspm')
            binsize = (60 * frequency) // binspm
        else:
            size = max((ch[-1] if ch.size else 0) for ch in data)
            binsize = size // bins + 1
    for ch in data:
        ch //= binsize
    size = int(max((ch[-1] if ch.size else 0) for ch in data) + 1)
    times = numpy.linspace(0, size * binsize / float(frequency), size,
                           endpoint=False)
    # FIXME: work around https://github.com/numpy/numpy/issues/823
    if size < 2**31:
        # use 32 bit signed int if possible
        data = (ch.astype('i4') for ch in data)
    else:
        # else use a signed view; fails on 32 bit
        data = (ch.view('i8') for ch in data)
    bincounts = tuple(numpy.bincount(ch, minlength=size) for ch in data)
    return times, bincounts


if __name__ == "__main__":
    import doctest
    numpy.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
