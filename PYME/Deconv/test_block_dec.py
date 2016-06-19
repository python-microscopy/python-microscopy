#!/usr/bin/python

###############
# test_block_dec.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#
################
__test__ = False
if __name__ == '__main__':
    from PYME.util import mProfile
    mProfile.profileOn(['tq_block_dec.py', 'dec.py'])
    
    from PYME.DSView import View3D, ViewIm3D
    from PYME.IO.image import ImageStack
    
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
