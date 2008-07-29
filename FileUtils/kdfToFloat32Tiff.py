import Image
import read_kdf

if not (len(sys.argv) == 3):
    raise 'Usage: procStack inDir resDir'

inDir = sys.argv[1]
outDir = sys.argv[2]
