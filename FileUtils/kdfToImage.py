import Image
import read_kdf
import sys

if not (len(sys.argv) == 3):
    raise 'Usage: kdfToImage infile outfile'

inFile = sys.argv[1]
outFile = sys.argv[2]

im = read_kdf.ReadKdfData(inFile).squeeze()

mode = ''

if (im.dtype.__str__() == 'uint16'):
    mode = 'I;16'
elif (im.dtype.__str__() == 'float32'):
    mode = 'F'
else:
    raise 'Error data type <%s> not supported' % im.dtype

Image.fromarray(im, mode).save(outFile)
