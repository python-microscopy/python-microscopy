# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:28:15 2015

@author: david

Defines a 'wire' format for transmitting or saving image frame data. 
"""


import numpy as np

FILE_FORMAT_ID = 'BD'
FORMAT_VERSION = 0

DATA_FMT_UINT8 = 0
DATA_FMT_UINT16 = 1
DATA_FMT_FLOAT32 = 2

DATA_FMTS = ['u1', 'u2', 'f4']

DATA_COMP_RAW = 0
DATA_COMP_HUFFCODE = 1

##############
# Definition of file header
#
# Most of the entries should be fairly self explanatory, with the following 
# deserving a bit more explanation:  
#
# ID: a 2-character string that we can test to see if the file type is consistent
# Version: the version of this format the file uses
# DataFormat: what the data type of individual pixels is
# DataCompression: whether the data is compressed, and which algorithm is used
# SequenceID: A unique identifier for the sequence to which this frame belongs.
#             The most important property of this number is that it is unique to 
#             each sequence. A reasonable method of generation would be to use
#             a unix-format integer timestamp for the first dword, and a random
#             integer for the second. A hash of the first n image pixels could 
#             also be used.
# FrameNum: The position of this frame within the sequence
# FrameTimestamp: Space to save camera derived frame timestamps, if available
# Depth: As envisaged, the format is expected to contain individual 2D frames, with
#        multiple frames being pulled together in a higher level container to 
#        construct a sequence or stack. Depth is included just because it doesn't
#        take a significant ammount of extra space, but gives us flexibility for
#        the future.

header_dtype = [('ID', 'S2'), ('Version', 'u1') , ('DataFormat', 'u1'), ('DataCompression', 'u1'), ('RESERVED0', 'S3'), ('SequenceID', 'i8'), 
                ('FrameNum', 'u4'), ('Width', 'u4'), ('Height', 'u4'), ('Depth', 'u4'), 
                ('FrameTimestamp', 'u8'), ('RESERVED1', 'u8')]
                
HEADER_LENGTH = np.zeros(1, header_dtype).nbytes            

def dumps(data, sequenceID=0, frameNum=0, frameTimestamp=0, compression = 'none'):
    '''dump an image frame (supplied as a numpy array) into a string in PZF format
    
    Parameters
    ==========

    data:  The frame as a 2D (or optionally 3D) numpt array    
    
    sequenceID:  A unique identifier for the sequence to which this frame belongs.
                 This will let us connect the frame with it's metadata even if 
                 they end up in different directories etc ...
                 
    frameNum:  The position of this frame within the sequence
    
    frameTimestamp:  A timestamp for the frame (if provided by the camera)
    
    compression:  compression method to use (at the moment only huffman is supported)
    '''
    
    header = np.zeros(1, header_dtype)
    
    header['ID'] = FILE_FORMAT_ID
    header['Version'] = FORMAT_VERSION
    
    header['FrameNum'] = frameNum
    header['SequenceID'] = sequenceID
    header['FrameTimestamp'] = frameTimestamp 
    
    if data.dtype == 'uint8':
        header['DataFormat'] = DATA_FMT_UINT8
    elif data.dtype == 'uint16':
        header['DataFormat'] = DATA_FMT_UINT16
    elif data.dtype == 'float32':
        header['DataFormat'] = DATA_FMT_FLOAT32
    else:
        raise RuntimeError('Unsupported data type')
    
    
    header['Width'] = data.shape[0]
    header['Height'] = data.shape[1]
    if data.ndim > 2:
        header['Depth'] = data.shape[2]
    else:
        header['Depth'] = 1
    
    if compression == 'huffmann':
        header['DataCompression'] = DATA_COMP_HUFFCODE
        
        raise RuntimeError('Compression not implemented yet')
        dataString = bcl.compress(data.tostring())
    else:
        dataString = data.tostring()
        
    return header.tostring() + dataString
 

   
def loads(datastring):
    header = np.fromstring(datastring[:HEADER_LENGTH], header_dtype)
    
    if not header['ID'] == FILE_FORMAT_ID:
        raise RuntimeError("Invalid format: This doesn't appear to be a PZF file")
    
    if header['DataCompression'] == DATA_COMP_RAW:
        #no need to decompress
        data = np.fromstring(datastring[HEADER_LENGTH:], DATA_FMTS[header['DataFormat']])
    else:
        raise RuntimeError('Compression not yet implemented')
    
    data = data.reshape([header['Width'], header['Height'], header['Depth']])
    
    return data, header