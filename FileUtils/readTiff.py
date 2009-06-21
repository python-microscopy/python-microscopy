import Image
import numpy as np


def read3DTiff(filename):
    im = Image.open(filename)

    im.seek(0)
    #print im.size

    ima = np.array(im.getdata(), 'int16').newbyteorder('BE')
    #ima = np.array(im)

    print ima.dtype
    
    #print ima.shape

    ima = ima.reshape((im.size[1], im.size[0], 1))

    pos = im.tell()

    try:
        while True:
            pos += 1
            im.seek(pos)

            ima = np.concatenate((ima, np.array(im.getdata(), 'int16').newbyteorder('BE').reshape((im.size[1], im.size[0], 1))), 2)

    except EOFError:
        pass

    return ima
