# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:28:15 2015

@author: david

Defines a 'wire' format for transmitting or saving image frame data. 
"""


import numpy as np

#try:
from PYME.bcl import bcl

from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

import threading

NUM_COMP_THREADS = 2#cpu_count()

compPool = ThreadPool(NUM_COMP_THREADS)

def ChunkedHuffmanCompress(data):
    num_chunks = NUM_COMP_THREADS
    
    chunk_size = int(np.ceil(float(len(data))/num_chunks))
    raw_chunks = [data[j*chunk_size:(j+1)*chunk_size].data for j in range(num_chunks)]
    
    #compPool = ThreadPool(NUM_COMP_THREADS)
    
    comp_chunk_d = {}
    
    def _compChunk(c, j):
        comp_chunk_d[j] = bcl.HuffmanCompress(c)
    
    threads = [threading.Thread(target = _compChunk, args=(rc, j)) for j, rc in enumerate(raw_chunks)]
            
    for p in threads:
        #print p
        p.start()

    for p in threads:
        p.join()

    
    #comp_chunks = compPool.map(bcl.HuffmanCompress, raw_chunks) 
    
    s = np.array([num_chunks], 'u2').tostring()
    
    for j, r in enumerate(raw_chunks):
        c = comp_chunk_d[j]
        s += np.array([len(c), len(r)], 'u4').tostring()
        s += c.tostring()
        
    return s

def ChunkedHuffmanCompress_o(data):
    num_chunks = NUM_COMP_THREADS
    
    chunk_size = int(np.ceil(float(len(data))/num_chunks))
    raw_chunks = [data[j*chunk_size:(j+1)*chunk_size].data for j in range(num_chunks)]
    
    #compPool = ThreadPool(NUM_COMP_THREADS)
    
    comp_chunks = compPool.map(bcl.HuffmanCompress, raw_chunks) 
    
    s = np.array([num_chunks], 'u2').tostring()
    
    for c, r in zip(comp_chunks, raw_chunks):
        s += np.array([len(c), len(r)], 'u4').tostring()
        s += c.tostring()
        
    return s

def _chunkDecompress(args):
    chunk, length = args
    return bcl.HuffmanDecompress(np.fromstring(chunk, 'u1'), length)
    
def ChunkedHuffmanDecompress(datastring):
    num_chunks = np.fromstring(datastring[:2], 'u2')
    
    #compPool = ThreadPool(NUM_COMP_THREADS)
    
    sp = 2
    
    comp_chunks = []
    for i in range(num_chunks):
        chunk_len, raw_len = np.fromstring(datastring[sp:(sp+8)], 'u4')
        sp += 8
        comp_chunks.append((datastring[sp:(sp+ chunk_len)], raw_len))
        sp += chunk_len
        
    
    decomp_chunks = compPool.map(_chunkDecompress, comp_chunks) 
    
    data = np.hstack(decomp_chunks)
    
    #print data.shape #, comp_chunks
    return data

#except ImportError:
#    pass

FILE_FORMAT_ID = 'BD'
FORMAT_VERSION = 1

DATA_FMT_UINT8 = 0
DATA_FMT_UINT16 = 1
DATA_FMT_FLOAT32 = 2

DATA_FMTS = ['u1', 'u2', 'f4']
DATA_FMTS_SIZES = [1,2,4]

DATA_COMP_RAW = 0
DATA_COMP_HUFFCODE = 1
DATA_COMP_HUFFCODE_CHUNKS = 2

DATA_QUANT_NONE = 0
DATA_QUANT_SQRT = 1

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

#header_dtype = [('ID', 'S2'), ('Version', 'u1') , ('DataFormat', 'u1'), ('DataCompression', 'u1'), ('RESERVED0', 'S3'), ('SequenceID', 'i8'), 
#                ('FrameNum', 'u4'), ('Width', 'u4'), ('Height', 'u4'), ('Depth', 'u4'), 
#                ('FrameTimestamp', 'u8'), ('RESERVED1', 'u8')]
                
header_dtype = [('ID', 'S2'), ('Version', 'u1') , ('DataFormat', 'u1'), ('DataCompression', 'u1'), 
                ('DataQuantization', 'u1'),('RESERVED0', 'S2'), ('SequenceID', 'i8'), 
                ('FrameNum', 'u4'), ('Width', 'u4'), ('Height', 'u4'), ('Depth', 'u4'), 
                ('FrameTimestamp', 'u8'), ('QuantOffset', 'f4'), ('QuantScale', 'f4')]
                
HEADER_LENGTH = np.zeros(1, header_dtype).nbytes            

def dumps(data, sequenceID=0, frameNum=0, frameTimestamp=0, compression = DATA_COMP_RAW, quantization=DATA_QUANT_NONE, quantizationOffset=0, quantizationScale=1):
    '''dump an image frame (supplied as a numpy array) into a string in PZF format
    
    Parameters
    ==========

    data:  The frame as a 2D (or optionally 3D) numpt array    
    
    sequenceID:  A unique identifier for the sequence to which this frame belongs.
                 This will let us connect the frame with it's metadata even if 
                 they end up in different directories etc ...
                 
    frameNum:  The position of this frame within the sequence
    
    frameTimestamp:  A timestamp for the frame (if provided by the camera)
    
    compression:  compression method to use - one of: PZFFormat.DATA_COMP_RAW,
                  PZFFormat.DATA_COMP_HUFFCODE, or PZFFormat.DATA_COMP_HUFFCODE_CHUNKS
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
        
    if quantization == DATA_QUANT_SQRT:
        #sqrt- quantize the data
        header['DataQuantization'] = DATA_QUANT_SQRT
        header['QuantOffset'] = quantizationOffset
        header['QuantScale'] = quantizationScale
        
        qs = 1.0/quantizationScale
        data = (np.sqrt(np.maximum(data-quantizationOffset,0))*qs).astype('uint8')
    
    if compression == DATA_COMP_HUFFCODE:
        header['DataCompression'] = DATA_COMP_HUFFCODE
        
        dataString = bcl.HuffmanCompress(data.data).tostring()
    elif compression == DATA_COMP_HUFFCODE_CHUNKS:
        header['DataCompression'] = DATA_COMP_HUFFCODE_CHUNKS
        
        dataString = ChunkedHuffmanCompress(data)
    else:
        dataString = data.tostring()
        
    return header.tostring() + dataString
 

   
def loads(datastring):
    header = np.fromstring(datastring[:HEADER_LENGTH], header_dtype)
    
    if not header['ID'] == FILE_FORMAT_ID:
        raise RuntimeError("Invalid format: This doesn't appear to be a PZF file")
        
    w, h, d = header['Width'], header['Height'], header['Depth']
    
    data_s = datastring[HEADER_LENGTH:]
    
    if header['DataCompression'] == DATA_COMP_RAW:
        #no need to decompress
        data = np.fromstring(data_s, 'u1')
    elif header['DataCompression'] == DATA_COMP_HUFFCODE:
        data = bcl.HuffmanDecompress(np.fromstring(data_s, 'u1'), len(data_s))
    elif header['DataCompression'] == DATA_COMP_HUFFCODE_CHUNKS:
        data = ChunkedHuffmanDecompress(data_s)
    else:
        raise RuntimeError('Compression type not understood')
        
    if header['DataQuantization'] == DATA_QUANT_SQRT:
        #un-quantize data
        data = data*header['QuantScale']
        data = (data*data + header['QuantOffset']).astype(DATA_FMTS[header['DataFormat']])
    
    data = data.view(DATA_FMTS[header['DataFormat']]).reshape([w,h,d])
    
    return data, header