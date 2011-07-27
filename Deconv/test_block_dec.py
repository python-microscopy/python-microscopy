from PYME import mProfile
mProfile.profileOn(['tq_block_dec.py', 'dec.py'])

from PYME.DSView import View3D, ViewIm3D
from PYME.DSView.image import ImageStack

image = ImageStack(filename = '/data/zar_cropped.tif')

data = image.data[:,:,:]
psf, vs = np.load('/home/david/Desktop/id.psf')
import Pyro.core
import os
tq = Pyro.core.getProxyForURI('PYRONAME://' + os.environ['PYME_TASKQUEUENAME'])
from PYME.Deconv import tq_block_dec
bd = tq_block_dec.blocking_deconv(tq, data, psf, 'foo', blocksize={'y': 128, 'x': 128, 'z': 256}, blockoverlap={'y': 10, 'x': 10, 'z': 50})
bd.go()
#bd.push_deconv_tasks()
bd.fake_push_deconv()
#bd.pull_and_deblock()

mProfile.report()
