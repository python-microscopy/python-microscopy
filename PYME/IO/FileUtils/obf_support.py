# obf_support.py from https://github.com/jkfindeisen/python-mix/tree/main/imspector/obf
# under the following license:
# MIT License

# Copyright (c) 2021 jkfindeisen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
  Pure Python read only support for OBF files. The OBF file format originates from the Department of NanoBiophotonics
  of the Max Planck Institute for Biophysical Chemistry in Göttingen, Germany. A specification can be found at
  https://github.com/AbberiorInstruments/ImspectorDocs/blob/master/docs/fileformat.rst This implementation is similar
  to the File and Stack API of specpy (https://pypi.org/project/specpy/). Can also read MSR files (the OBF part of it).

  Documentation:

      Include in your project as "import obf_support"

      File
      - Open an OBF file with "obf = obf_support.File(path_to_file)", that will read all meta data (including stack meta data)
      - Access the following attributes: format_version, description, stacks
      - Close with "obf.close()" (optional, is also closed automatically on deletion of the File object)

      Stack
      - Each Stack has attributes: format_version, name, description, shape, lengths, offsets, data_type, data
      - data returns a NumPy array containing the stack data (the stack data is loaded from the file lazily, i.e. when the
        attribute is accessed the first time)

  Example: see obf_support_example.py

  Implementation notes:

      - Relies on the struct module (https://docs.python.org/3.9/library/struct.html).
      - In particular see the format characters of the struct module (https://docs.python.org/3.9/library/struct.html#format-characters).
      - Opened issue at https://github.com/AbberiorInstruments/ImspectorDocs/issues/9 about
        + Constant OMAS_BF_MAX_DIMENSIONS is not explained, the value is 15.
        + Data type of OMAS_DT is not specified, it's an enum type in C++, which is stored as uint32.
      - In the future maybe:
        + Writing to OBF would in principle be possible (using the struct module)
        + Read part (slice) of data (use flush points), currently data is read all at once (impractical for very large files)
        + may still crash if not all data is written in a stack (haven't seen such a stack yet)
        + if there is a problem with a stack (like unknown data type), we could simply ignore the stack and print a warning instead

  Author: Jan Keller-Findeisen (https://github.com/jkfindeisen), May-July 2021, MIT licensed (see LICENSE)
"""

from __future__ import annotations
from collections import namedtuple
import struct
import zlib
import math
import numpy as np

# single long value
long_fmt = '<Q'
long_len = struct.calcsize(long_fmt)
long_unpack = struct.Struct(long_fmt).unpack_from

# file header = char[10], uint32, uint64, uint32
file_header_fmt = '<10sIQI'
file_header_len = struct.calcsize(file_header_fmt)
file_header_unpack = struct.Struct(file_header_fmt).unpack_from
FILE_MAGIC_HEADER = b'OMAS_BF\n\xff\xff'

# stack header = char[16], uint32, uint32, uint32[15], double[15], double[15], uint32, uint32, uint32, uint32, uint32, uint64, uint64, uint64
stack_header_fmt = '<16s17I30d5I3Q'
stack_header_len = struct.calcsize(stack_header_fmt)
stack_header_unpack = struct.Struct(stack_header_fmt).unpack_from
STACK_MAGIC_HEADER = b'OMAS_BF_STACK\n\xff\xff'

# stack footer version 1 = uint32, uint32[15], uint32[15]
stack_footer_v1_fmt = '<31I'
stack_footer_v1_len = struct.calcsize(stack_footer_v1_fmt)
stack_footer_v1_unpack = struct.Struct(stack_footer_v1_fmt).unpack_from

# stack footer version 1A = version 1 + uint32 (we only store the difference)
stack_footer_v1a_fmt = '<I'
stack_footer_v1a_len = struct.calcsize(stack_footer_v1a_fmt)
stack_footer_v1a_unpack = struct.Struct(stack_footer_v1a_fmt).unpack_from

# OBF_SI_FRACTION = int32[2] = '2i'
# OBF_SI_UNIT = OBF_SI_FRACTION[9], double = '18id'

# stack footer version 2 = version1A + OBF_SI_UNIT, OBF_SI_UNIT[15]
stack_footer_v2_fmt = '<' + '18id' * 16
stack_footer_v2_len = struct.calcsize(stack_footer_v2_fmt)
stack_footer_v2_unpack = struct.Struct(stack_footer_v2_fmt).unpack_from

# stack footer version 3 = version 2 + uint64, uint64
stack_footer_v3_fmt = '<2Q'
stack_footer_v3_len = struct.calcsize(stack_footer_v3_fmt)
stack_footer_v3_unpack = struct.Struct(stack_footer_v3_fmt).unpack_from

# stack footer version 4 = version 3 + uint64
stack_footer_v4_fmt = '<Q'
stack_footer_v4_len = struct.calcsize(stack_footer_v4_fmt)
stack_footer_v4_unpack = struct.Struct(stack_footer_v4_fmt).unpack_from

# stack footer version 5 = version 4 + uin64, uint32
stack_footer_v5_fmt = '<QI'
stack_footer_v5_len = struct.calcsize(stack_footer_v5_fmt)
stack_footer_v5_unpack = struct.Struct(stack_footer_v5_fmt).unpack_from

# stack footer version 5a = version 5 + uint64
stack_footer_v5a_fmt = '<Q'
stack_footer_v5a_len = struct.calcsize(stack_footer_v5a_fmt)
stack_footer_v5a_unpack = struct.Struct(stack_footer_v5a_fmt).unpack_from

# stack footer version 6 = version 5a + uint64, uint64
stack_footer_v6_fmt = '<2Q'
stack_footer_v6_len = struct.calcsize(stack_footer_v6_fmt)
stack_footer_v6_unpack = struct.Struct(stack_footer_v6_fmt).unpack_from

# mapping of OMAS_DT to NumPy data types (see https://numpy.org/doc/stable/user/basics.types.html)
omas_data_types = {0x00000001: np.uint8, 0x00000002: np.int8, 0x00000004: np.uint16, 0x00000008: np.int16,
                   0x00000010: np.uint32, 0x00000020: np.int32, 0x00000040: np.float32, 0x00000080: np.float64,
                   0x00001000: np.uint64, 0x00002000: np.int64, 0x00010000: np.bool_}

# named tuple used in SIUnit
Fraction = namedtuple('Fraction', ('numerator', 'denominator'))


class File:
    """
    OBF file access.

    Reads the file header, stack header and stack footer of all stacks contained in the file.

    Attributes:
        - format_version
        - description
        - stacks (list of Stack)
    """

    def __init__(self, file_path: str):
        """
        Create a new OBF file access object by providing a file path:
        obf = File(file_path)

        :param file_path: path of the OBF file
        """
        # we cannot use "with open as" because we read the data stacks content later
        try:
            # open the file at the given file path
            self._file = open(file_path, 'rb')
            self._file.seek(0, 2)  # seek to the end of the file
            file_size = self._file.tell()
            self._file.seek(0)

            # read the obf file header
            data = self._file.read(file_header_len)
            magic_header, self.format_version, first_stack_pos, description_len = file_header_unpack(data)
            if magic_header != FILE_MAGIC_HEADER:
                raise RuntimeError('Magic file header not found.')

            # read file description
            self.description = self._read_string(description_len)

            # read file meta data
            if self.format_version >= 2:
                # read meta data position
                file_meta_data_pos = long_unpack(self._file.read(long_len))[0]
                self._file.seek(file_meta_data_pos)

                self.meta = {}
                key = self._read_string()
                while len(key) > 0:
                    value = self._read_string()
                    self.meta[key] = value
                    key = self._read_string()

            # read all the stacks
            next_stack_pos = first_stack_pos
            self.stacks = []
            while next_stack_pos != 0:

                # create new stack
                stack = Stack(self)

                # seek to position of next stack header
                self._file.seek(next_stack_pos)

                # read stack header
                data = self._file.read(stack_header_len)
                values = stack_header_unpack(data)
                if values[0] != STACK_MAGIC_HEADER:
                    raise RuntimeError('Magic stack header not found.')

                # interpret stack header
                stack.format_version = values[1]
                stack.rank = values[2]
                stack.shape = values[3:17][:stack.rank]
                stack.lengths = values[18:32][:stack.rank]
                stack.offsets = values[33:47][:stack.rank]
                value = values[48]
                if value not in omas_data_types:
                    raise RuntimeError('Unsupported data type {}.'.format(value))
                else:
                    stack.data_type = omas_data_types[value]
                stack._compression_type = values[49]
                # compression_level = values[50] # relatively uninteresting, we ignore it
                name_length = values[51]
                description_length = values[52]
                stack._data_length = values[54]  # data_len_disk
                next_stack_pos = values[55]
                stack.name = self._read_string(name_length)
                stack.description = self._read_string(description_length)

                stack._data_pos = self._file.tell()
                footer_pos = stack._data_pos + stack._data_length

                # additionally we compute a dimensionality of a stack which is the number of elements in shape minus
                # trailing single value dimensions; helps finding the 2D image stacks for example
                dimensionality = len(stack.shape)
                while dimensionality > 1 and stack.shape[dimensionality - 1] == 1:
                    dimensionality -= 1
                stack.dimensionality = dimensionality

                # default footer
                footer = {
                    'stack_end_used_disk': file_size,
                    'samples_written': np.prod(stack.shape),
                    'chunk_positions': [[0, 0]]
                }

                # read and interpret stack footer (for format version >= 1)
                if stack.format_version >= 1:
                    self._file.seek(footer_pos)

                    # read version 1 part
                    data = self._file.read(stack_footer_v1_len)
                    values = stack_footer_v1_unpack(data)
                    footer_length = values[0]
                    footer['has_col_positions'] = values[1:15][:stack.rank]
                    footer['has_col_labels'] = values[16:30][:stack.rank]

                    if stack.format_version >= 2:
                        # read version 1A part
                        data = self._file.read(stack_footer_v1a_len)
                        values = stack_footer_v1a_unpack(data)
                        footer['metadata_length'] = values[0]

                        # read version 2 part
                        data = self._file.read(stack_footer_v2_len)
                        values = stack_footer_v2_unpack(data)
                        stack.si_value = SIUnit(values[0:19])
                        stack.si_dimensions = []
                        for i in range(stack.rank):
                            stack.si_dimensions.append(SIUnit(values[i * 19:(i + 1) * 19]))

                    if stack.format_version >= 3:
                        # read version 3 part
                        data = self._file.read(stack_footer_v3_len)
                        values = stack_footer_v3_unpack(data)
                        footer['num_flush_points'] = values[0]
                        footer['flush_block_size'] = values[1]

                    if stack.format_version >= 4:
                        # read version 4 part
                        data = self._file.read(stack_footer_v4_len)
                        values = stack_footer_v4_unpack(data)
                        footer['tag_dictionary_length'] = values[0]

                    if stack.format_version >= 5:
                        # read version 5 part
                        data = self._file.read(stack_footer_v5_len)
                        values = stack_footer_v5_unpack(data)
                        footer['min_format_version'] = values[1]

                    if stack.format_version >= 6:
                        # read version 5a part
                        data = self._file.read(stack_footer_v5a_len)
                        values = stack_footer_v5a_unpack(data)
                        footer['stack_end_used_disk'] = values[0]

                        # read version 6 part
                        data = self._file.read(stack_footer_v6_len)
                        values = stack_footer_v6_unpack(data)
                        footer['samples_written'] = values[0]
                        footer['num_chunk_positions'] = values[1]

                    # omit possible footer entries from later versions
                    self._file.seek(footer_pos + footer_length)

                    # read label strings
                    stack.labels = [self._read_string() for _ in range(stack.rank)]

                    # read col positions
                    if 'has_col_positions' in footer:
                        stack.col_positions = {}
                        for axis, has_them in enumerate(footer['has_col_positions']):
                            if has_them:
                                # read doubles as positions
                                fmt = '<{}d'.format(stack.shape[axis])
                                data = self._file.read(struct.calcsize(fmt))
                                values = struct.unpack_from(fmt, data)
                                stack.col_positions[axis] = values

                    # read col labels
                    if 'has_col_labels' in footer:
                        stack.col_labels = {}
                        for axis, has_them in enumerate(footer['has_col_labels']):
                            if has_them:
                                # read labels
                                labels = []
                                for _ in range(stack.shape[axis]):
                                    label = self._read_string()
                                    labels.append(label)
                                stack.col_labels[axis] = labels

                    # read metadata
                    if 'metadata_length' in footer:
                        stack.metadata = self._read_string(footer['metadata_length'])

                    # read flush positions
                    if 'num_flush_points' in footer:
                        length = footer['num_flush_points']
                        fmt = '<{}Q'.format(length)
                        data = self._file.read(struct.calcsize(fmt))
                        values = struct.unpack_from(fmt, data)
                        footer['flush_positions'] = values

                    # read tag dictionary
                    if 'tag_dictionary_length' in footer:
                        stack.tag_dictionary = {}
                        length = footer['tag_dictionary_length']
                        if length > 0:
                            # read key, value pairs until len(key) is zero
                            key = self._read_string()
                            while len(key) > 0:
                                value = self._read_string()
                                stack.tag_dictionary[key] = value
                                key = self._read_string()

                    # read chunk positions
                    chunk_positions = [[0, 0]]
                    for _ in range(footer.get('num_chunk_positions', 0)):
                        fmt = '<2Q'
                        data = self._file.read(struct.calcsize(fmt))
                        values = struct.unpack_from(fmt, data)
                        chunk_positions.append(values)
                    footer['chunk_positions'] = chunk_positions

                stack.footer = footer

                # append stack to list
                self.stacks.append(stack)
        except:
            self.close()
            raise

    def find_stack_by_name(self, name_part: str) -> list[Stack]:
        """
        Small convenience method. Will return all stacks in this OBF file where string is contained in the stack name.
        """
        return [stack for stack in self.stacks if name_part in stack.name]

    def close(self):
        """
        Closes the file if it isn't closed already.
        """
        if not self._file.closed:
            self._file.close()

    def _read_string(self, length: int = None) -> str:
        """
        For internal use only.
        :param length: Number of bytes to read, if none is given, reads the length first.
        :return: Decoded string
        """
        if length is None:
            fmt = '<I'
            data = self._file.read(struct.calcsize(fmt))
            length = struct.unpack_from(fmt, data)[0]

        fmt = '<{}s'.format(length)
        data = self._file.read(struct.calcsize(fmt))
        string = struct.unpack_from(fmt, data)[0]  # unpack always returns a tuple
        try:
            string = string.decode('utf-8')
        except UnicodeDecodeError:
            # fallback encoding for very old (<2008) files
            string = string.decode('iso-8859-1')

        return string

    def _read_stack(self, stack: Stack):
        """
        Internal function. Reads the data array from a stack from the OBF file as a NumPy array and stores it as the
        _data attribute of the stack. If called a second time, will re-read the stack.

        Supporting the chunked/interleaved storage of stacks with stack format version 6, this is a bit more
        elaborate and also includes a bit of heuristic estimation of the number of compressed bytes
        that need to be read. The problem is that the length of the last chunk of chunked data does not seem to be
        properly specified, so the total number of bytes contained in a compressed data stack is unknown and
        we need to workaround it.

        :param stack: A Stack object containing all meta data
        """
        try:
            # read the whole stack data (works for stack format versions <= 5)
            # self._file.seek(stack._data_pos)
            # data = self._file.read(stack._data_length)

            # with chunks (works for min_format_version 6 and also below)
            pos = 0
            idx = 0
            seek_pos = stack._data_pos
            self._file.seek(seek_pos)
            data = []
            if stack._compression_type == 1 and stack.footer['samples_written'] > 0:
                # if compressed and not empty: we are not completely sure about the length of the written data
                if 'num_chunk_positions' in stack.footer:
                    # stack format version >= 6 (with chunks)
                    bytes_written = min(stack.footer['samples_written']*stack.data_type().itemsize+16, stack.footer['chunk_positions'][-1][0] + stack.footer['stack_end_used_disk'] - stack.footer['chunk_positions'][-1][1])  # this is a bit heuristic and not documented but I don't want to read too much
                    # stack.footer['chunk_positions'][-1][0] + stack.footer['stack_end_used_disk'] - stack.footer['chunk_positions'][-1][1] is the maximal number of bytes between the begin of the last chunk and the end of the data
                    # stack.footer['samples_written']*stack.data_type().itemsize+16 is the size of the uncompressed data plus a small overhead for the zip header that is also divisible by all data type sizes in bytes
                else:
                    # stack format version < 6 (without chunks)
                    bytes_written = stack._data_length
            else:
                # if not compressed or empty, we know the number of bytes exactly
                bytes_written = stack.footer['samples_written'] * stack.data_type().itemsize

            # read (using the algorithm outlined in the format description)==
            while pos < bytes_written:
                bytes_to_read = bytes_written - pos
                if idx < len(stack.footer['chunk_positions']):
                    if pos + bytes_to_read > stack.footer['chunk_positions'][idx][0]:  # chunk_positions[0] = logical offset, [1] = file offset
                        bytes_to_read = stack.footer['chunk_positions'][idx][0] - pos
                        seek_pos = stack.footer['chunk_positions'][idx][1] + stack._data_pos
                        idx += 1
                if bytes_to_read > 0:
                    data.append(self._file.read(bytes_to_read))
                self._file.seek(seek_pos)
                pos += bytes_to_read

            data = b"".join(data)  # is there a more efficient way to concatenate byte arrays?

            # if compressed, uncompress
            if stack._compression_type == 1:
                zobj = zlib.decompressobj()
                data = zobj.decompress(data)
                # data = zlib.decompress(data) # that gave "zlib.error: Error -5 while decompressing data: incomplete or truncated stream" sometimes

            # convert to numpy array
            array = np.frombuffer(data, dtype=stack.data_type)
            array = array[:stack.footer['samples_written']]  # not sure if this is needed anymore

            # reshape (with reversed shape and then reverse order of dimensions)
            array = np.reshape(array, stack.shape[::-1])
            array = np.transpose(array)

            # store
            stack._data = array
        except:
            self.close()
            raise

    def __del__(self):
        """
        Make sure that the file is closed upon deletion.
        """
        self.close()


class Stack:
    """
    A Stack class, holds attributes about stacks
    """

    def __init__(self, file: File):
        """
        Initialize with a File object.
        """
        self.file = file
        self._data = None

    def __getattr__(self, name: str):
        """
        Computes a few convenience attributes on the fly as well as lazy loading of the data
        :param name: either "pixel_sizes" or "data"
        """
        if name == 'pixel_sizes':
            # if a dimension is 0, the pixel size is NaN in that direction
            pixel_sizes = [length / n if n > 0 else math.nan for length, n in zip(self.lengths, self.shape)]
            return pixel_sizes
        elif name == 'data':
            if self._data is None:
                # first time data is called, load it
                self.file._read_stack(self)
            return self._data


class SIUnit:
    """
    OMAS_SIUNIT reimplementation
    exponents = Meters (M), Kilograms (KG), Seconds (S), Amperes (A), Kelvin (K), Moles (MOL), Candela (CD), Radian (R), Steradian (SR)
    """

    unitnames = ['m', 'kg', 's', 'A', 'K', 'mol', 'CD', 'R', 'SR']

    def __init__(self, values):
        """
        Initialize with 19 double values.
        """
        if len(values) != 19:
            raise RuntimeError('SI unit needs 19 values to initialize.')
        self.exponents = []
        for i in range(9):
            self.exponents.append(Fraction(values[i * 2], values[i * 2 + 1]))
        self.scalefactor = values[18]

    def __str__(self) -> str:
        """
        Some kind of meaningful string representation. Could still be improved.
        """
        s = '{} '.format(self.scalefactor)
        for exponent, unit in zip(self.exponents, SIUnit.unitnames):
            if exponent.numerator != 0:
                s += '{}^({}/{})'.format(unit, *exponent)
        return s
