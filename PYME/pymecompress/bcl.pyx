import numpy as np
from cython.view cimport array as cvarray
cimport cython
from libc.stdint cimport uint16_t, uint8_t

cdef extern from "huffman.h":
    int Huffman_Compress( unsigned char *inp, unsigned char *out, unsigned int insize ) nogil
    int Huffman_Compress_( unsigned char *inp, unsigned char *out, unsigned int insize ) nogil
    void Huffman_Uncompress( unsigned char *inp, unsigned char *out, unsigned int insize, unsigned int outsize ) nogil

cdef extern from "quantize.h":
    void quantize_u16(uint16_t *data, uint8_t * out, int size, float offset, float scale) nogil
    void quantize_u16_avx( uint16_t * data, uint8_t * out, int size, float offset, float scale) nogil
    
@cython.boundscheck(False)    
def HuffmanCompress(unsigned char[:] data):
    out = np.zeros(int(data.shape[0]*1.01 + 320),'uint8')
    cdef unsigned char [:] ov = out
    cdef int dsize = data.shape[0]
    with nogil:
        
        nb = Huffman_Compress(&data[0], &ov[0], dsize)
    return out[:nb]

@cython.boundscheck(False)
def HuffmanCompressQuant(unsigned short[:] data, float offset, float scale):
    out = np.zeros(int(data.shape[0]*1.01 + 320),'uint8')
    quant = np.zeros(data.shape[0], 'uint8')
    cdef unsigned char [:] ov = out
    cdef unsigned char [:] qv = quant
    cdef int dsize = data.shape[0]
    with nogil:
        quantize_u16_avx(&data[0], &qv[0], dsize, offset, scale)
        nb = Huffman_Compress(&qv[0], &ov[0], dsize)
    return out[:nb]

@cython.boundscheck(False)    
def HuffmanCompressOrig(unsigned char[:] data):
    out = np.zeros(int(data.shape[0]*1.01 + 320),'uint8')
    cdef unsigned char [:] ov = out
    cdef int dsize = data.shape[0]
    with nogil:
        
        nb = Huffman_Compress_(&data[0], &ov[0], dsize)
    return out[:nb]

@cython.boundscheck(False)   
def HuffmanDecompress(unsigned char[:] data, unsigned int outsize):
    out = np.zeros(outsize,'uint8')
    cdef unsigned char [:] ov = out
    cdef int insize = data.shape[0]
    #cdef int outsize = outsize
    with nogil:
        
        Huffman_Uncompress(&data[0], &ov[0], insize, outsize)
    return out
    
    