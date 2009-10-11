import sys
import os

import KdfStackToHdf5


if __name__ == '__main__':
    pixelsize=0.09

    fns = sys.argv[1:]

    for inFile in fns:
        outFile = os.path.splitext(inFile)[0] + '.h5'
        KdfStackToHdf5.convertFile(inFile, outFile, pixelsize=pixelsize)