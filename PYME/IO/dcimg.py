import numpy as np
import os

'''This is a reverse-engineered reader for DCIMG files, based very loosely on:
https://github.com/StuartLittlefair/dcimg/blob/master/dcimg/Raw.py

'''

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

class DCIMGFile(object):
	def __init__(self, filename):
		with open(filename, 'rb') as hfile:
			header_bytes = hfile.read(232)

		self._info = self._parse_header(header_bytes)

		self.data = np.memmap(filename, dtype='<u2', mode='r', 
			  						offset=self._info['session0_data'], 
			  						shape=(self._info['num_columns'],self._info['num_rows'],self._info['num_frames']), order='F')

	def _parse_header(self, header):
		info = {}

		if not header.startswith('DCIMG'):
			raise RuntimeError("Not a valid DCIMG file")

		#the file header starts with an itentifying string and then seems to consist (mostly) of uint32 integers, aligned on 
		#4-byte boundaries. 

		#after the identifying string, the first non-zero entry is at byte 8
		#it is unclear exactly what this means, but it is 7 in the example data. The number of frames is roughly 7 4-byte dwords away, 
		#so this might be an offset. That said, it could also be a version number or pretty much anything.
		info['initial_offset'] = int(np.fromstring(header[8:12], 'uint32'))

		if not (info['initial_offset'] == 7):
			#in example data this is always 7. Warn if this is not the case, as further assumptions might be invalid
			print "Warning: Initial offset is %d rather than 7 as expected" % info['initial_offset']

		#the next non-zero value is 6 dwords further into the file at byte 32. This is most likely to do with sessions. DCIMG files
		#support multiple "sessions", each of which contains a series of frames. I have no multi-session data to test on, so cannot
		#ascertain exactly what this value means. Reasonable candidates are either the number of sessions, or the current/first session ID

		info['num_sessions'] = int(np.fromstring(header[32:36], 'uint32'))
		if not info['num_sessions'] == 1:
			print "Warning: it appears that there are %d sessions. We only support one session" % info['num_sessions']

		#the next entry is the number of frames, most likey in the first session. We do not attempt to support multiple sessions
		info['num_frames'] = int(np.fromstring(header[36:40], 'uint32'))

		#and the next entry is an offset to the beginning of what I'm guessing is the first session
		info['session0_offset'] = int(np.fromstring(header[40:44], 'uint32'))

		#this is followed by the filesize in bytes
		info['filesize'] = int(np.fromstring(header[48:52], 'uint32'))

		#NB because there is zero-padding after these offset and size values, it's possible they are long (64 bit) integers instead
		#of uint32. None of the example data breaks the 4GB limit that would require long offsets.

		#The filesize is repeated starting at byte 64, for unknown reasons
		info['filesize2'] = int(np.fromstring(header[64:68], 'uint32'))

		#the next non-zero value is at byte 84, and has the value 1024 in the example data. The meaning is unknown.
		info['mystery1'] = int(np.fromstring(header[84:88], 'uint32'))

		#read the 1st session header 
		session_head = np.fromstring(header[info['session0_offset']:(info['session0_offset']+SESSION_HEADER_BYTES)], 
									dtype=SESSION_HEADER_DTYPE)

		info['pixel_type'] = session_head['pixel_type']
		info['num_columns'] = session_head['num_columns']
		info['bytes_per_row'] = session_head['bytes_per_row']
		info['bytes_per_pixel'] = info['bytes_per_row']/info['num_columns']
		info['num_rows'] = session_head['num_rows']
		info['bytes_per_image'] = session_head['bytes_per_image']

		info['session0_data'] = info['session0_offset'] + session_head['offset_to_data']
		info['session0_footer'] = info['session0_offset'] + session_head['offset_to_footer']

		return info

