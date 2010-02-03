from dec import *


class dec_sofi(dec):
    def psf_calc(self, psf, data_size, orders, weights):
        pw = (numpy.array(data_size[:3]) - psf.shape)/2.
        pw1 = numpy.floor(pw)
        pw2 = numpy.ceil(pw)

        g = psf#/psf.sum();

        if pw1[0] < 0:
            if pw2[0] < 0:
                g = g[-pw1[0]:pw2[0]]
            else:
                g = g[-pw1[0]:]

            pw1[0] = 0
            pw2[0] = 0

        if pw1[1] < 0:
            if pw2[1] < 0:
                g = g[-pw1[1]:pw2[1]]
            else:
                g = g[-pw1[1]:]

            pw1[1] = 0
            pw2[1] = 0

        if pw1[2] < 0:
            if pw2[2] < 0:
                g = g[-pw1[2]:pw2[2]]
            else:
                g = g[-pw1[2]:]

            pw1[2] = 0
            pw2[2] = 0


        g = pad.with_constant(g, ((pw2[0], pw1[0]), (pw2[1], pw1[1]),(pw2[2], pw1[2])), (0,))

        self.height = data_size[0]
        self.width  = data_size[1]
        self.depth  = data_size[2]

        self.shape = data_size[:2]

        self.nOrders = data_size[3]
        self.orders = orders
        self.weights = weights
#
#        (x,y,z) = mgrid[-floor(self.height/2.0):(ceil(self.height/2.0)), -floor(self.width/2.0):(ceil(self.width/2.0)), -floor(self.depth/2.0):(ceil(self.depth/2.0))]
#
#        gs = shape(g);
#
#        g = g[int(floor((gs[0] - self.height)/2)):int(self.height + floor((gs[0] - self.height)/2)), int(floor((gs[1] - self.width)/2)):int(self.width + floor((gs[1] - self.width)/2)), int(floor((gs[2] - self.depth)/2)):int(self.depth + floor((gs[2] - self.depth)/2))]
#
#        #g = abs(ifftshift(ifftn(abs(fftn(g)))));
#        g = (g/sum(sum(sum(g))));

        self.g = g.astype('float32');

        #%g = circshift(g, [0, -1]);
        self.H = []
        self.Ht = []
        for i in range(self.nOrders):
            gp = self.g**(self.orders[i])
            gp = gp/sum(gp)
            print gp.dtype
            self.H.append((fftn(gp)))
            self.Ht.append((gp.size*ifftn(gp)))

    def startGuess(self, data):
        guess = zeros(data.shape[:3])

        for i in range(self.nOrders):
            guess += data[:,:,:,i]**(1./self.orders[i])
        
        return guess/self.nOrders

    def subsearch(self, f0, res, fdef, Afunc, Lfunc, lam, S):
        nsrch = size(S,1)
        pref = Lfunc(f0-fdef)
        w0 = dot(pref, pref)
        c0 = dot(res,res)

        AS = zeros((size(res), nsrch), 'f')
        LS = zeros((size(pref), nsrch), 'f')

        for k in range(nsrch):
            AS[:,k] = cast['f'](Afunc(f0 + S[:,k]) - Afunc(f0))
            LS[:,k] = cast['f'](Lfunc(S[:,k]))

        Hc = dot(transpose(AS), AS)
        Hw = dot(transpose(LS), LS)
        Gc = dot(transpose(AS), res)
        Gw = dot(transpose(-LS), pref)

        c = solve(Hc + pow(lam, 2)*Hw, Gc + pow(lam, 2)*Gw)

        cpred = c0 + dot(dot(transpose(c), Hc), c) - dot(transpose(c), Gc)
        wpred = w0 + dot(dot(transpose(c), Hw), c) - dot(transpose(c), Gw)

        fnew = f0 + dot(S, c)

        return (fnew, cpred, wpred)

    def deconv(self, data, lamb, num_iters=10, alpha = None):#, Afunc=self.Afunc, Ahfunc=self.Ahfunc, Lfunc=self.Lfunc, Lhfunc=self.Lhfunc):
        self.dataShape = data.shape

        #lamb = 2e-2
        if (not alpha == None):
            self.alpha = alpha
            self.e1 = fftshift(exp(1j*self.alpha))
            self.e2 = fftshift(exp(2j*self.alpha))

        self.f = self.startGuess(data)

        self.f = self.f.ravel()
        data = data.ravel()

        fdef = zeros(self.f.shape, 'f')

        S = zeros((size(self.f), 3), 'f')

        #print type(S)
        #print shape(S)

        #type(Afunc)

        nsrch = 2

        for loopcount in range(num_iters):
            pref = self.Lfunc(self.f - fdef);
            self.res = data - self.Afunc(self.f);

            #print type(Afunc(res))
            #print shape(Afunc(res))

            #print pref.typecode()

            S[:,0] = cast['f'](self.Ahfunc(self.res))
            S[:,1] = cast['f'](-self.Lhfunc(pref))

            #print S

            test = 1 - abs(dot(S[:,0], S[:,1])/(norm(S[:,0])*norm(S[:,1])))

            print 'Test Statistic %f\n' % (test,)
            self.tests.append(test)
            self.ress.append(norm(self.res))
            self.prefs.append(norm(pref))

            (fnew, cpred, spred) = self.subsearch(self.f, self.res, fdef, self.Afunc, self.Lfunc, lamb, S[:, 0:nsrch])

            fnew = cast['f'](fnew*(fnew > 0))

            S[:,2] = cast['f'](fnew - self.f)
            nsrch = 3

            self.f = fnew

        return real(self.f)

    def Lfunc(self, f):
        fs = reshape(f, (self.height, self.width, self.depth))
        a = -6*fs

        a[:,:,0:-1] += fs[:,:,1:]
        a[:,:,1:] += fs[:,:,0:-1]

        a[:,0:-1,:] += fs[:,1:,:]
        a[:,1:,:] += fs[:,0:-1,:]

        a[0:-1,:,:] += fs[1:,:,:]
        a[1:,:,:] += fs[0:-1,:,:]

        return ravel(cast['f'](a))

    Lhfunc=Lfunc

    def Afunc(self, f):
        fs = reshape(f, (self.height, self.width, self.depth))
        #print fs.shape

        ds = []

        for i in range(self.nOrders):
            F = fftn(fs**(self.orders[i]))
            d = ifftshift(ifftn(F*self.H[i])).real
            ds.append(d[:,:,:,None])

        #print ravel(array(ds)).shape
        return ravel(concatenate(ds, 3))

    def Ahfunc(self, f):
        #print f.shape
        fs = reshape(f, (self.height, self.width, self.depth, self.nOrders))

        ds = []

        for i in range(self.nOrders):
            F = fftn(fs[:,:,:,i])
            d = ifftshift(ifftn(F*self.Ht[i])).real
            #d = sign(d)*abs(d)**(1./self.orders[i])
            d = d/abs(d).max()
            #print d.shape
            ds.append(d*self.weights[i])

        return ravel(sum(ds, axis=0))