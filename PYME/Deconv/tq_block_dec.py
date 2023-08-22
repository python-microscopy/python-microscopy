#!/usr/bin/python

##################
# block_dec.py
#
# Copyright David Baddeley, 2009
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
##################

#f = f3
#import tcluster
from . import dec
from scipy import *
import decTask #import decTask



class blocking_deconv:
	def __init__(self, tq, data, psf, name, blocksize = {'x':64, 'y':64, 'z':256}, blockoverlap = {'x':10, 'y':10, 'z':50}):
		#self.tc = tcluster.ThreadedCluster(nodes)

		self.blocksize = blocksize
		self.blockoverlap = blockoverlap

		self.data = data
		#self.alpha = alpha
		self.psf = psf
		self.tq = tq
		self.name = name

		self.killed = False

		#print shape(data)

		(self.height, self.width, self.depth) = shape(self.data)

		if (self.width > self.blocksize['y']):
			#self.sp_y = [0,]
			#print self.sp_y
			self.sp_y = range(0,(self.width-self.blocksize['y']), (self.blocksize['y'] -2*self.blockoverlap['y']))
			#self.sp_y.extend(range((self.blocksize['y'] -2*self.blockoverlap['y']),(self.width-self.blocksize['y']), self.blocksize['y']))
			#print self.sp_y
			self.sp_y.append(self.width-self.blocksize['y'])
		else:
			self.sp_y = [0,]
			self.blocksize['y'] = self.width

		if (self.height > self.blocksize['x']):
			#self.sp_x = [0,]
			self.sp_x = range(0,(self.height-self.blocksize['x']), (self.blocksize['x'] -2*self.blockoverlap['x']))
			#self.sp_x.extend(range((self.blocksize['x'] -2*self.blockoverlap['x']),(self.width-self.blocksize['x']), self.blocksize['x']))
			self.sp_x.append(self.height-self.blocksize['x'])
		else:
			self.sp_x = [0,]
			self.blocksize['x'] = self.height

		if (self.depth > self.blocksize['z']):
			#self.sp_z = [0,]
			self.sp_z = range(0,(self.depth-self.blocksize['z']), (self.blocksize['z'] -2*self.blockoverlap['z']))
			#self.sp_z.extend(range((self.blocksize['z'] -2*self.blockoverlap['z']),(self.width-self.blocksize['z']), self.blocksize['z']))
			self.sp_z.append(self.depth-self.blocksize['z'])
		else:
			self.sp_z = [0,]
			self.blocksize['z'] = self.depth

		#print self.sp_y
		#print self.sp_x
		#print self.sp_z

		self.d4 = dec.dec_conv()


	def blockify(self):
		self.blocks = []
		#self.alphas = []
		#print self.blocksize
		k = 0
		for sl_start_b in self.sp_y:
			for sl_start_a in self.sp_x:
				for sl_start in self.sp_z:
					#alpha = ph_param(1)*(x+ sl_start_a) + ph_param(2)*(y + sl_start_b) + ph_param(3)*(z + sl_start) + ph_param(4);
					#print '(%d,%d,%d)' %(sl_start_a, sl_start_b, sl_start)

					#alpha_b = self.alpha[sl_start_a:(sl_start_a + self.blocksize['x']),sl_start_b:(sl_start_b + self.blocksize['y']), sl_start:(sl_start + self.blocksize['z'])]

					f = self.data[sl_start_a:(sl_start_a + self.blocksize['x']),sl_start_b:(sl_start_b + self.blocksize['y']), sl_start:(sl_start + self.blocksize['z'])]

					#f = self.data(sl_start_a + (1:length(x1)) - 1,sl_start_b + (1:length(y1)) - 1, sl_start + (1:length(z1)) - 1)

					self.blocks.append((f, k))
					k = k +1
					#self.alphas.append(alpha_b)


	def init_psf(self, kz):
		self.d4.psf_calc(self.psf, (self.blocksize['x'], self.blocksize['y'], self.blocksize['z']))

	def init_cluster(self):
		self.tq.createQueue('TaskQueueWithData', self.name)
		self.tq.setQueueData(self.name, 'dec', self.d4)

	def cleanup_cluster(self):
		#Free up memory on server
		self.tq.removeQueue(self.name)

	def push_deconv_tasks(self, lamb = 2e-2, num_iters = 10):
		for block, blocknum in zip(self.blocks, range(len(self.blocks))):
			task = decTask.decTask(self.name, block[0], blocknum, lamb=lamb, num_iters=num_iters)
			self.tq.postTask(task, self.name)
		#self.results = self.tc.loop_code('f = d4.deconv(ravel(a[0]), lamb, alpha=a[1], num_iters=num_it)', 'a', {'a':self.blocks, 'lamb':lamb, 'num_it':num_iters}, ('f',))
		#self.results = []
		#for r in self.blocks:
		#    self.results.append((r[0],r[2]))

	def fake_push_deconv(self, lamb = 2e-2, num_iters = 10):
		self.results = [None for b in self.blocks]

		#decTask.decObj = self.d4
		#decTask.queueID = self.name

		for block, blocknum in zip(self.blocks, range(len(self.blocks))):
			task = decTask.decTask(self.name, block[0], blocknum, lamb=lamb, num_iters=num_iters)
			t = task(taskQueue=self.tq)
			self.results[t.blocknum] = t.results
			#self.tq.postTask(task, self.name)

#    def do_sim(self):
#        self.results = self.tc.loop_code('f = d4.sim_pic(ravel(a[0]), alpha=a[1])', 'a', {'a':self.blocks}, ('f',))
#        #self.results = []
#        #for r in self.blocks:
#        #    self.results.append((r[0],r[2]))

	def pull_and_deblock(self):
		self.results = [None for b in self.blocks]

		while self.tq.getNumberTasksCompleted(self.name) > 0:
			t = self.tq.getCompletedTask(self.name)

			self.results[t.blocknum] = t.results

		self.deblock()

	def deblock(self):
		i = 0
		self.end_res = zeros((self.height, self.width, self.depth), 'f');
		self.blockno = zeros((self.height, self.width, self.depth), 'f');
		#print self.blocksize
		for sl_start_b in self.sp_y:
			for sl_start_a in self.sp_x:
				for sl_start in self.sp_z:
					res  =  self.results[i]
					i += 1
					#print(res[1])
					#print '(%d,%d,%d)' %(sl_start_a, sl_start_b, sl_start)

					fe = reshape(res, (self.blocksize['x'], self.blocksize['y'], self.blocksize['z']))

					block_n = i*ones(shape(fe))

					sel_x1 = self.blockoverlap['x']
					sel_x2 = self.blocksize['x']-self.blockoverlap['x']

					sel_z1 = self.blockoverlap['z']
					sel_z2 = self.blocksize['z']-self.blockoverlap['z']

					sel_y1 = self.blockoverlap['y']
					sel_y2 = self.blocksize['y']-self.blockoverlap['y']


					if (sl_start_a == 0):
						sel_x1 = 0

					if (sl_start == 0):
						sel_z1 = 0

					if (sl_start_b == 0):
						sel_y1 = 0

					if (sl_start_a >= (self.height - self.blocksize['x'])):
						sel_x2 = self.blocksize['x']

					if (sl_start >= (self.depth - self.blocksize['z'])):
						sel_z2 = self.blocksize['z']

					if (sl_start_b >= (self.width - self.blocksize['y'])):
						sel_y2 = self.blocksize['y']

					#print '(%d:%d,%d:%d, %d:%d)' % ((sl_start_a + sel_x1),(sl_start_a + sel_x2),(sl_start_b + sel_y1),(sl_start_b + sel_y2), (sl_start + sel_z1),(sl_start + sel_z2))

					self.end_res[(sl_start_a + sel_x1):(sl_start_a + sel_x2),(sl_start_b + sel_y1):(sl_start_b + sel_y2), (sl_start + sel_z1):(sl_start + sel_z2)] = cast['f'](real(fe[sel_x1:sel_x2,sel_y1:sel_y2, sel_z1:sel_z2]))

			self.blockno[(sl_start_a + sel_x1):(sl_start_a + sel_x2),(sl_start_b + sel_y1):(sl_start_b + sel_y2), (sl_start + sel_z1):(sl_start + sel_z2)] = cast['f'](real(block_n[sel_x1:sel_x2,sel_y1:sel_y2, sel_z1:sel_z2]))

	def go(self, kz = 1, lamb = 2e-2, num_iters = 10):
		print('Dividing into blocks ...')
		self.blockify()
		print('Doing some PSF related precomutations ...')
		self.init_psf(kz)
		print('Sending precomputed info to cluster ...')
		self.init_cluster()
		print('Starting the deconvolution ....')
		#self.push_deconv_tasks(lamb, num_iters)

	def retrieve(self):
		self.pull_and_deblock()
		self.cleanup_cluster()

		self.res = self.end_res


	def isAlive(self):
		if self.killed:
			return False
		else:
			ia =  self.tq.getNumberTasksCompleted(self.name) < len(self.blocks)
			if not ia:
				self.retrieve()
			return ia

	def kill(self):
		self.killed = True
		self.cleanup_cluster()
		#self.do_deconv(lamb, num_iters)
		#self.cleanup_cluster()
		#print 'Finished deconvolution, putting blocks back together ...'
		#self.deblock()

		#self.cleanup_cluster()



#    def sim(self, kz = 1):
#        print 'Dividing into blocks ...'
#        self.blockify()
#        print 'Doing some PSF related precomutations ...'
#        self.init_psf(kz)
#        print 'Sending precomputed info to cluster ...'
#        self.init_cluster()
#        print 'Starting the simulation ....'
#        self.do_sim()
#        self.cleanup_cluster()
#        print 'Finished simulation, putting blocks back together ...'
#        self.deblock()

#if __name__ == '__main__':
#    import sys
#    import os
#    from optparse import OptionParser
#    import read_kdf
#    import write_kdf
#    import time
#
#    parser = OptionParser()
#    parser.add_option("-d", "--data", dest="data", help="Read data from FILE", metavar="FILE")
#    parser.add_option("-p", "--psf", dest="psf", help="Read psf from FILE", metavar="FILE")
#    parser.add_option("-a", "--alpha", dest="alpha", help="Read phase info from FILE", metavar="FILE")
#    parser.add_option("-k", "--kz", dest="kz", type="float", help="kz - relates wavelength & voxelsize", metavar="FLOAT")
#    parser.add_option("-l", "--lambda", dest="lamb", type="float", help="Regularisation parameter", metavar="FLOAT", default=2e-2)
#    parser.add_option("-n", "--num_iters", dest="num_iters", type="int", help="Number of iterations", metavar="INT", default=10)
#    parser.add_option("-s", "--servers", dest="servers", type="string", help="Servers (in format host0:port0,host1:port1 etc)", metavar="STRING")
#    parser.add_option("-o", "--out", dest="out", help="Write result to FILE", metavar="FILE")
#    parser.add_option("-f", "--forward", action="store_true", dest="forward", help="Perform forward mapping instead of deconvolution", default=False)
#
#    (options, args) = parser.parse_args()
#
#    if options.data == None:
#        print 'Must give a file containing the data'
#        sys.exit(1)
#    if options.psf == None:
#        print 'Must give a file containing the psf'
#        sys.exit(1)
#    if options.out == None:
#        print 'Must give an output file'
#        sys.exit(1)
#    if options.kz == None:
#        print 'Must give the wavelength parameter'
#        sys.exit(1)
#
#    data = squeeze(read_kdf.ReadKdfData(options.data))
#    psf = squeeze(read_kdf.ReadKdfData(options.psf))
#
#    if not options.alpha == None:
#        alpha = squeeze(read_kdf.ReadKdfData(options.alpha))
#    else:
#        alpha = zeros(shape(data), 'f')
#
#    print shape(data)
#    print shape(alpha)
#    print(shape(psf))
#
#    servers = []
#    if not options.servers == None:
#        svrs = options.servers.split(',')
#        for s in svrs:
#            (host,port)=s.split(':')
#            servers.append((host, int(port)))
#    else:
#        print 'Creating a server on local machine'
#        #fid
#        os.spawnl(os.P_NOWAIT, '/usr/bin/python', 'python', '/usr/lib/python/site-packages/scipy/cow/sync_cluster.py', 'server', '10001')
#	#os.spawnl(os.P_NOWAIT, '/usr/bin/python', 'python', '/usr/lib/python/site-packages/scipy/cow/sync_cluster.py', 'server', '10002')
#        import socket
#        servers.append((socket.gethostname(), 10001))
#	#servers.append((socket.gethostname(), 10002))
#        #time.sleep(10)
#
#    #print options.lamb
#    d4 = blocking_deconv(servers, data, psf, alpha)
#    if not options.forward:
#        d4.go(options.kz, options.lamb, options.num_iters)
#    else:
#        d4.sim(options.kz)
#
#
#    write_kdf.WriteKhorosData(options.out, real(d4.end_res))
#    write_kdf.WriteKhorosData('blocks_out.kdf', real(d4.blockno))
#    write_kdf.WriteKhorosData('psf_out.kdf', real(d4.d4.g))