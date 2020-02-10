import numpy as np
import pytest
#pytestmark = pytest.mark.skip("Segfaults on Linux")


def test_compression_lossless_uint16():
    from pymecompress import bcl
    test_data = np.random.poisson(100, 10000).reshape(100,100).astype('uint16')

    result = bcl.HuffmanDecompress(bcl.HuffmanCompress(np.frombuffer(test_data.data, dtype='uint8')),
                                   test_data.nbytes).view(test_data.dtype).reshape(test_data.shape)

    assert np.allclose(result, test_data)

def test_compression_lossless_uint8():
    from pymecompress import bcl
    test_data = np.random.poisson(100, 10000).reshape(100,100).astype('uint8')

    result = bcl.HuffmanDecompress(bcl.HuffmanCompress(np.frombuffer(test_data.data, dtype='uint8')),
                                   test_data.nbytes).view(test_data.dtype).reshape(test_data.shape)

    assert np.allclose(result, test_data)

def test_PZFFormat_raw_uint16():
    from PYME.IO import PZFFormat
    test_data = np.random.poisson(100, 10000).reshape(100,100).astype('uint16')

    result, header = PZFFormat.loads(PZFFormat.dumps(test_data))

    print(test_data, result.squeeze())
    print(header, result.dtype)

    assert np.allclose(result.squeeze(), test_data.squeeze())
    
def test_PZFFormat_raw_uint16_F():
    from PYME.IO import PZFFormat
    test_data = np.random.poisson(100, 10000).reshape(100,100).astype('uint16').copy(order='F')

    print(test_data.flags)
    
    result, header = PZFFormat.loads(PZFFormat.dumps(test_data))

    #print result
    print(header, result.dtype)

    print(test_data, result.squeeze())

    assert np.allclose(result.squeeze(), test_data.squeeze())

def test_PZFFormat_raw_uint8():
    from PYME.IO import PZFFormat
    test_data = np.random.poisson(50, 100).reshape(10,10).astype('uint8')

    result, header = PZFFormat.loads(PZFFormat.dumps(test_data))

    #print result.squeeze(), test_data, result.shape, test_data.shape

    assert np.allclose(result.squeeze(), test_data.squeeze())

def test_PZFFormat_lossless_uint16():
    from PYME.IO import PZFFormat
    test_data = np.random.poisson(100, 10000).reshape(100,100).astype('uint16')

    result, header = PZFFormat.loads(PZFFormat.dumps(test_data, compression = PZFFormat.DATA_COMP_HUFFCODE))

    #print result

    assert np.allclose(result.squeeze(), test_data.squeeze())

def test_PZFFormat_lossy_uint16():
    from PYME.IO import PZFFormat
    test_data = np.random.poisson(100, 100).reshape(10,10).astype('uint16')

    result, header = PZFFormat.loads(PZFFormat.dumps(test_data,
                                                     compression = PZFFormat.DATA_COMP_HUFFCODE,
                                                     quantization = PZFFormat.DATA_QUANT_SQRT,
                                                     quantizationOffset=0, quantizationScale=1))

    #print result
    test_quant = (np.round(np.sqrt(test_data.astype('f'))).astype('i'))**2

    #print(test_quant.squeeze() - result.squeeze())
    #print(test_data.squeeze())
    #print(test_quant.squeeze())
    #print(result.squeeze())

    assert np.allclose(result.squeeze(), test_quant.squeeze())
    
def test_PZFFormat_lossy_uint16_qs():
    from PYME.IO import PZFFormat
    test_data = np.random.poisson(100, 100).reshape(10,10).astype('uint16')
    
    qs = .2

    result, header = PZFFormat.loads(PZFFormat.dumps(test_data,
                                                     compression = PZFFormat.DATA_COMP_HUFFCODE,
                                                     quantization = PZFFormat.DATA_QUANT_SQRT,
                                                     quantizationOffset=0, quantizationScale=qs))

    #print result
    test_quant = ((np.round(np.sqrt(test_data.astype('f'))/qs).astype('i')*qs)**2).astype('i')

    
    print(test_data.min(), test_data.max(), result.min(), result.max(), test_quant.min(), test_quant.max())
    
    #print(test_quant.squeeze() - result.squeeze())
    #print(test_data.squeeze())
    #print(test_quant.squeeze())
    #print(result.squeeze())
    
    print(result.squeeze() - test_quant.squeeze())

    assert np.allclose(result.squeeze(), test_quant.squeeze())

def test_PZFFormat_lossless_uint8():
    from PYME.IO import PZFFormat
    test_data = np.random.poisson(50, 100).reshape(10,10).astype('uint8')

    result, header = PZFFormat.loads(PZFFormat.dumps(test_data, compression = PZFFormat.DATA_COMP_HUFFCODE))

    #print result.squeeze(), test_data, result.shape, test_data.shape

    assert np.allclose(result.squeeze(), test_data.squeeze())