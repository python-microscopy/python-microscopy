import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

class offline_plotting(object):
    def __enter__(self):
        plt.ioff()
        self.old_backend = plt.get_backend()
        plt.switch_backend('SVG')
    
    def __exit__(self, *args):
        plt.switch_backend(self.old_backend)
        plt.ion()

def plot(data, xvals='bins', yvals=['counts', ], type='line', xlabel=None, ylabel=None, title=None, figsize=(7,5), **kwargs):
    import matplotlib.pyplot as plt
    import mpld3
    import warnings
    if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
        #mpld3 has messed with warnings - undo
        warnings.filters.pop(0)
    
    #print type
    
    if xlabel is None:
        xlabel = xvals
        
    if ylabel is None:
        ylabel = yvals[0]
    
    with offline_plotting():
        f = plt.figure(figsize=figsize)
    
        xv = data[xvals]
        for yv_name in yvals:
            yv = data[yv_name]
        
            if type == 'bar':
                plt.bar(xv, yv, align='center', width=(xv[1] - xv[0]))
            elif type=='scatter':
                plt.scatter(xv, yv, **kwargs)
            elif type=='errorbar':
                plt.errorbar(xv, yv, **kwargs)
            else:
                plt.plot(xv, yv, **kwargs)
                
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if title:
            plt.title(title)
    
        plt.tight_layout()
    
        ret = mpld3.fig_to_html(f, template_type='simple')
        plt.close(f)
        
        return ret


def hist(data, bins=20, type='line', xlabel=None, ylabel=None, title=None, figsize=(7, 5)):
    import matplotlib.pyplot as plt
    import mpld3
    import warnings
    if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
        #mpld3 has messed with warnings - undo
        warnings.filters.pop(0)
        
    import numpy as np
    
    #print type
    
    if xlabel is None:
        xlabel = 'Bins'
    
    if ylabel is None:
        ylabel = 'Frequency'
    
    with offline_plotting():
        f = plt.figure(figsize=figsize)
        
        counts, edges = np.histogram(data, bins)
        
        centres = 0.5*(edges[1:] + edges[:-1])
            
        if type == 'bar':
            plt.bar(centres, counts, align='center', width=(edges[1] - edges[0]))
        else:
            plt.plot(centres, counts)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if title:
            plt.title(title)
        
        plt.tight_layout()
        
        ret = mpld3.fig_to_html(f, template_type='simple')
        plt.close(f)
        
        return ret

def movieplot(clump, image):
    #if image.size == 0:
    #    #stop us from blowing up if we get an empty image
    #    return ''
    
    import matplotlib.pyplot as plt
    import mpld3
    import warnings
    if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
        #mpld3 has messed with warnings - undo
        warnings.filters.pop(0)
    
    with offline_plotting():
        nRows = int(np.ceil(clump.nEvents / 10.))
        f = plt.figure(figsize=(12, 1.2 * nRows))
        
        #msdi = self.msdinfo
        #t = msdi['t']
        #plt.plot(t[1:], msdi['msd'][1:])
        #plt.plot(t, powerMod2D([msdi['D'], msdi['alpha']], t))
        
        if 'centroid' in clump.keys():
            xp, yp = clump['centroid'][0]
        elif 'x_pixels' in clump.keys():
            xp = clump['x_pixels'][0]
            yp = clump['y_pixels'][0]
        
        xp = int(np.round(xp))
        yp = int(np.round(yp))
        
        try:
            contours = clump['contour']
        except (KeyError, RuntimeError):
            contours = None
        
        for i in range(clump.nEvents):
            plt.subplot(nRows, min(clump.nEvents, 10), i + 1)
            if image.data.shape[3] == 1:
                #single color
                print((xp - 20), (xp + 20), (yp - 20), (yp + 20), int(clump['t'][i]))
                img = image.data[(xp - 20):(xp + 20), (yp - 20):(yp + 20), int(clump['t'][i])].squeeze()
                
                if 'mean_intensity' in clump.keys():
                    scMax = clump.featuremean['mean_intensity'] * 1.5
                else:
                    scMax = img.max()
                
                plt.imshow(img.T, interpolation='nearest', cmap=plt.cm.gray, clim=[0, scMax])
            
            else:
                img = np.zeros([40, 40, 3], 'uint8')
                
                for j in range(min(image.data.shape[3], 3)):
                    im_j = image.data[(xp - 20):(xp + 20), (yp - 20):(yp + 20), int(clump['t'][i]),
                           j].squeeze().T.astype('f')
                    
                    if 'mean_intensity' in clump.keys():
                        scMax = clump.featuremean['mean_intensity'] * 1.5
                    else:
                        scMax = im_j.max()
                    
                    #plt.imshow(img.T, interpolation='nearest', cmap=plt.cm.gray, clim=[0, scMax])
                    img[:, :, j] = np.clip(255. * im_j / (scMax), 0, 255).astype('uint8')
                
                plt.imshow(img.T, interpolation='nearest')
            
            if not contours is None:
                xc, yc = contours[i].T
                plt.plot(xc - xp + 20, yc - yp + 20, c='b')#plt.cm.hsv(clump.clumpID/16.))
            
            xsb = 5
            ysb = 5
            plt.plot([xsb, xsb + 200. / image.pixelSize], [ysb, ysb], 'y', lw=4)
            
            plt.xticks([])
            plt.yticks([])
            plt.axis('image')
            plt.axis('off')
        
        plt.tight_layout(pad=1)
        
        #plt.ion()
        
        ret = mpld3.fig_to_html(f)
        plt.close(f)
    return ret


#env.filters['movieplot'] = movieplot


def movieplot2(clump, image):
    #if image.size == 0:
    #    #stop us from blowing up if we get an empty image
    #    return ''
    
    import matplotlib.pyplot as plt
    #from PIL import Image
    import mpld3
    import warnings
    if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
        #mpld3 has messed with warnings - undo
        warnings.filters.pop(0)
    
    #plt.ioff()
    with offline_plotting():
        nRows = int(np.ceil(clump.nEvents / 10.))
        f = plt.figure(figsize=(12, 1.2 * nRows))
        
        #msdi = self.msdinfo
        #t = msdi['t']
        #plt.plot(t[1:], msdi['msd'][1:])
        #plt.plot(t, powerMod2D([msdi['D'], msdi['alpha']], t))
        
        roiHalfSize = 20
        roiSize = roiHalfSize * 2 + 1
        
        img_out = 255 * np.ones([nRows * (roiSize + 2), 10 * (roiSize + 2), image.data.shape[3]], 'uint8')
        
        if 'centroid' in clump.keys():
            xp, yp = clump['centroid'][0]
        elif 'x_pixels' in clump.keys():
            xp = clump['x_pixels'][0]
            yp = clump['y_pixels'][0]
        
        xp = int(np.round(xp))
        yp = int(np.round(yp))
        
        try:
            contours = clump['contour']
        except (KeyError, RuntimeError):
            contours = None
        
        for i in range(clump.nEvents):
            k = int(np.floor(i / 10))
            l = i % 10
            
            y_0 = l * (roiSize + 2)
            x_0 = img_out.shape[0] - (1 + k) * (roiSize + 2)
            #plt.subplot(nRows, min(clump.nEvents, 10), i + 1)
            if image.data.shape[3] == 1:
                #single color
                #print (xp - 20), (xp + 20), (yp - 20), (yp + 20), int(clump['t'][i])
                img = image.data[(xp - roiHalfSize):(xp + roiHalfSize + 1), (yp - roiHalfSize):(yp + roiHalfSize + 1),
                      int(clump['t'][i])].squeeze()
                
                if img.size == 0:
                    return ''
                
                if 'mean_intensity' in clump.keys():
                    scMax = clump.featuremean['mean_intensity'] * 1.5
                else:
                    scMax = img.max()
                
                # print(x_0, y_0, img.shape, img_out.shape, i)
                
                img_out[x_0:(x_0 + roiSize), y_0:(y_0 + roiSize), 0] = np.clip(255. * img.T / scMax, 0, 255).astype(
                    'uint8')
                
                #plt.imshow(img.T, interpolation='nearest', cmap=plt.cm.gray, clim=[0, scMax])
            
            else:
                img = np.zeros([40, 40, 3], 'uint8')
                
                for j in range(min(image.data.shape[3], 3)):
                    im_j = image.data[(xp - roiHalfSize):(xp + roiHalfSize + 1),
                           (yp - roiHalfSize):(yp + roiHalfSize + 1), int(clump['t'][i]), j].squeeze().T.astype(
                        'f')
                    
                    if 'mean_intensity' in clump.keys():
                        scMax = clump.featuremean['mean_intensity'] * 1.5
                    else:
                        scMax = im_j.max()
                    
                    #plt.imshow(img.T, interpolation='nearest', cmap=plt.cm.gray, clim=[0, scMax])
                    #img[:, :, j] = np.clip(255. * im_j / (scMax), 0, 255).astype('uint8')
                    img_out[x_0:(x_0 + roiSize), y_0:(y_0 + roiSize), :] = np.clip(255. * img.T / scMax, 0, 255).astype(
                        'uint8')
            
            if not contours is None:
                xc, yc = contours[i].T
                plt.plot(xc - xp + roiHalfSize + y_0, yc - yp + roiHalfSize + x_0, c='b')#plt.cm.hsv(clump.clumpID/16.))
        
        if img_out.shape[2] <= 1:
            plt.imshow(img_out.squeeze(), interpolation='nearest', cmap=plt.cm.gray, clim=[0, 255])
        else:
            plt.imshow(img_out, interpolation='nearest')
        
        xsb = 5
        ysb = 5
        plt.plot([xsb, xsb + 1000. / image.pixelSize], [ysb, ysb], 'y', lw=4)
        
        plt.xticks([])
        plt.yticks([])
        plt.axis('image')
        plt.axis('off')
        #plt.ylim(plt.ylim()[::-1])
        
        plt.tight_layout(pad=1)
        
        #plt.ion()
        
        ret = mpld3.fig_to_html(f)
        plt.close(f)
    return ret


#env.filters['movieplot'] = movieplot2