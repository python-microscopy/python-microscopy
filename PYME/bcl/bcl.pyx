import numpy as np
from cython.view cimport array as cvarray

cdef extern from "huffman.h":
    int Huffman_Compress( unsigned char *inp, unsigned char *out, unsigned int insize ) nogil
    void Huffman_Uncompress( unsigned char *inp, unsigned char *out, unsigned int insize, unsigned int outsize ) nogil
    
    
def HuffmanCompress(unsigned char[:] data):
    out = np.zeros(int(data.shape[0]*1.01 + 320),'uint8')
    cdef unsigned char [:] ov = out
    cdef int dsize = data.shape[0]
    with nogil:
        
        nb = Huffman_Compress(&data[0], &ov[0], dsize)
    return out[:nb]
    
def HuffmanDecompress(unsigned char[:] data, unsigned int outsize):
    out = np.zeros(outsize,'uint8')
    cdef unsigned char [:] ov = out
    cdef int insize = data.shape[0]
    #cdef int outsize = outsize
    with nogil:
        
        Huffman_Uncompress(&data[0], &ov[0], insize, outsize)
    return out
    
    