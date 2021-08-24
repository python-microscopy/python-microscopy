from __future__ import print_function
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

"""This is a reverse-engineered reader for DCIMG files, based very loosely on:
https://github.com/StuartLittlefair/dcimg/blob/master/dcimg/Raw.py

"""

#best guess at the session header
SESSION_HEADER_DTYPE = [('session_length', 'i4'), 
                        ('pad1', 'i4'), 
                        ('psuedo_offset', 'i4'),
                        ('pad2', '5i4'),
                        ('num_frames', 'i4'),
                        ('pixel_type', 'i4'),
                        ('mystery1', 'i4'),
                        ('num_columns', 'i4'),
                        ('bytes_per_row', 'i4'),
                        ('num_rows', 'i4'),
                        ('bytes_per_image', 'i4'),
                        ('pad3', '2i4'),
                        ('offset_to_data', 'i4'),
                        ('offset_to_footer', 'i4')]

SESSION_HEADER_BYTES = np.zeros(1, dtype=SESSION_HEADER_DTYPE).nbytes
#immediately folloing this structure, there are a few more non-zero entries at dwords +2, +3, & +4, which I ca't make sense of

#newer versions of the dcimg format have a different header 
SESSION_HEADER_DTYPE_INT2P24 = [('session_length', 'i4'),
                        ('pad1', '5i4'),
                        ('psuedo_offset', 'i4'),
                        ('pad2', '8i4'),        # there were mysteries number 1, 144, 65537 in padding
                        ('num_frames', 'i4'),
                        ('pixel_type', 'i4'),
                        ('mystery1', 'i4'),
                        ('num_columns', 'i4'),
                        ('num_rows', 'i4'),      # num_rows switched position
                        ('bytes_per_row', 'i4'),
                        ('bytes_per_image', 'i4'),
                        ('pad3', '2i4'),
                        ('offset_to_data', 'i4'),
                        ('pad4', '4i4'),
                        ('bytes_per_frame', 'i4')]  # end_of_a_frame = bytes_per_image + 32 bytes - DB: Is this an offset, or the frame size?

SESSION_HEADER_BYTES_INT2P24 = np.zeros(1, dtype=SESSION_HEADER_DTYPE_INT2P24).nbytes

class DCIMGFile(object):
    def __init__(self, filename):
        from PYME.IO import unifiedIO
        with unifiedIO.openFile(filename, 'rb') as hfile:
            header_bytes = hfile.read(728)  #Read enough of the header to open both verssions (previously 232) 

        self._info = self._parse_header(header_bytes)

        self.frames_with_footer = np.memmap(filename, dtype='<u2', mode='r', offset=int(self._info['session0_data']),
                                            shape=(int(self._info['bytes_per_frame']/self._info['bytes_per_pixel']), int(self._info['num_frames'])), order='F')
        # In the case of initial offset = 16777216, the 32 bytes footer included 4 pixel values to correct for [0 65535 0 65535] within the frame

    def get_frame(self, ind):
        """Get the frame at the given index, discarding a footer if needed

        Parameters
        ----------

        ind : int
            The index of the frame to retrieve
        """
        frame_wo_footer = self.frames_with_footer[0:(int(self._info['bytes_per_image']/self._info['bytes_per_pixel'])), ind]
        raw_frame_data = frame_wo_footer.reshape([self._info['num_columns_raw'], self._info['num_rows']], order='F')
        return raw_frame_data[self._info['row_offset']:, :]  # drop pixels outside defined FOV

    def get_slice_shape(self):
        return (self._info['num_columns'], self._info['num_rows'])

    def get_num_slices(self):
        return self._info['num_frames']

    def _parse_header(self, header):
        info = {}

        if not header.startswith(b'DCIMG'):
            raise RuntimeError("Not a valid DCIMG file")

        # the file header starts with an itentifying string and then seems to consist (mostly) of uint32 integers, aligned on 
        # 4-byte boundaries. 

        # after the identifying string, the first non-zero entry is at byte 8
        # it is unclear exactly what this means, but it is 7 in the example data. The number of frames is roughly 7 4-byte dwords away, 
        # so this might be an offset. That said, it could also be a version number or pretty much anything.
        # We have encountered two cases format_version = 0x7 or 0x1000000, which have differences in both session_head and frame data format
        # Could this be some form of "flags" register?
        info['format_version'] = int(np.fromstring(header[8:12], 'uint32'))

        if not (info['format_version'] in [0x7, 0x1000000]):
            # in example data this is always 7. Warn if this is not the case, as further assumptions might be invalid
            print("Warning: format_version is %d  rather than 0x7 or 0x1000000 as expected" % info['format_version'])

        # the next non-zero value is 6 dwords further into the file at byte 32. This is most likely to do with sessions. DCIMG files
        # support multiple "sessions", each of which contains a series of frames. I have no multi-session data to test on, so cannot
        # ascertain exactly what this value means. Reasonable candidates are either the number of sessions, or the current/first session ID

        info['num_sessions'] = int(np.fromstring(header[32:36], 'uint32'))
        if not info['num_sessions'] == 1:
            print("Warning: it appears that there are %d sessions. We only support one session" % info['num_sessions'])

        # the next entry is the number of frames, most likey in the first session. We do not attempt to support multiple sessions
        info['num_frames'] = int(np.fromstring(header[36:40], 'uint32'))

        # and the next entry is an offset to the beginning of what I'm guessing is the first session
        info['session0_offset'] = int(np.fromstring(header[40:44], 'uint32'))

        # this is followed by the filesize in bytes
        info['filesize'] = int(np.fromstring(header[48:52], 'uint32'))

        # NB because there is zero-padding after these offset and size values, it's possible they are long (64 bit) integers instead
        # of uint32. None of the example data breaks the 4GB limit that would require long offsets.

        # The filesize is repeated starting at byte 64, for unknown reasons
        info['filesize2'] = int(np.fromstring(header[64:68], 'uint32'))

        # the next non-zero value is at byte 84, and has the value 1024 in the example data. The meaning is unknown.
        info['mystery1'] = int(np.fromstring(header[84:88], 'uint32'))

        # read the 1st session header  - vormat varies depending on file format
        if info['format_version'] == 7: 
            session_head = np.fromstring(header[info['session0_offset']:(info['session0_offset']+SESSION_HEADER_BYTES)], 
                                        dtype=SESSION_HEADER_DTYPE)[0]

        elif info['format_version'] >= 0x1000000:
            if info['format_version'] > 0x1000000:
                print("Warning: you are using version {}, but only version {} is guaranteed to work.".format(info['format_version'], 0x1000000))
            session_head = np.fromstring(header[info['session0_offset']:(info['session0_offset']+SESSION_HEADER_BYTES_INT2P24)],
                                    dtype=SESSION_HEADER_DTYPE_INT2P24)[0]

            # each image include data and 32 bytes footer
        else:
            raise RuntimeError("Unknown file version: %0X" % info['format_version'])

        info['pixel_type'] = session_head['pixel_type']
        info['num_columns'] = session_head['num_columns']
        info['bytes_per_row'] = session_head['bytes_per_row']
        info['bytes_per_pixel'] = int(info['bytes_per_row']/info['num_columns'])
        info['num_rows'] = session_head['num_rows']
        info['session0_data'] = info['session0_offset'] + session_head['offset_to_data']
        #info['session0_footer'] = info['session0_offset'] + session_head['offset_to_footer']

        info['bytes_per_image'] = session_head['bytes_per_image']
        try:
            #if the bytes per frame is different to the bytes per image (e.g. if there is a frame footer)
            info['bytes_per_frame'] = session_head['bytes_per_frame']
        except ValueError:
            info['bytes_per_frame'] = session_head['bytes_per_image']

        # depending on ROI position and size, dcimg occasionally contain extra bytes per 'row', though this ends up as
        # per column once we convert from fortran ordering. You have to drop the first 'row_offset' rows of a 2D image,
        # i.e. image = raw[row_offset:, :]
        info['num_columns_raw'] = int(info['bytes_per_row'] / info['bytes_per_pixel'])
        info['row_offset'] = 0
        if info['bytes_per_row'] % info['num_columns'] != 0:
            logger.debug('Handling %d extra bytes per row.' % info['bytes_per_row'] % info['num_columns'])
            info['row_offset'] = info['num_columns_raw'] - info['num_columns']

        return info

