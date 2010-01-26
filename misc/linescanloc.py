import Image
import scipy.ndimage
from pylab import *

im = Image.open('/home/david/Desktop/linescan1.tif')
a = array(im.getdata()).reshape(im.size[::-1])
b = scipy.ndimage.gaussian_filter(a, [0,8])
c = (b-b.mean(0)[None, :])

pts = []

for t in range(c.shape[0]):
    l, nl = scipy.ndimage.label(c[t, :] > 20)
    if nl > 0:
        m = scipy.ndimage.measurements.center_of_mass(c[t, :] - 20, l, arange(nl) + 1)
        pts += [(atleast_1d(x)[0], t) for x in m]

pts = array(pts)

figure()
subplot(131)
imshow(a, interpolation='nearest', clim=(a.min(), a.max()-.25*(a.max()-a.min())))
xlabel('x')
ylabel('Time')
title('Raw')

subplot(132)
imshow(c, interpolation='nearest', clim = (0,100))
xlabel('x')
#ylabel('Time')
title('Filtered')

subplot(133)
imshow(c, interpolation='nearest', clim = (0,100))
plot(pts[:,0], pts[:,1], 'xw')
xlabel('x')
#ylabel('Time')
axis('tight')
axis('image')
title('Positions')



x1 = pts[:,0]
x1.sort()
x2 = x1[1:-1]

dx1 = diff(x1)
dx1 = sqrt(dx1[:-1]**2 + dx1[1:]**2)
dx1 = 3*scipy.ndimage.gaussian_filter(dx1, 5)

x_ = arange(0,512, .2)


figure()
subplot(311)
#plot(arange(512)/141.7, a.mean(0))
plot(arange(512)/141.7,(c*(c>0)).mean(0))
#xlabel('x [um]')
ylabel('Mean Intensity')

subplot(312)
hist(x1/141.7, arange(0, 512, 3)/141.7)
#xlabel('x [um]')
ylabel('Frequency')

subplot(313)
plot(x_/141.7,((1./sqrt(2*pi*dx1[None, :]**2))*exp(-(x_[:,None] - x2[None, :])**2/(2*dx1[None, :]**2))).sum(1))
xlabel('x [um]')
ylabel('Estimated Density')