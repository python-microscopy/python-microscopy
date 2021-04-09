import numpy as np
from scipy import ndimage

def image_to_rgb(image, zoom=1.0, scaling='min-max', scaling_factor=0.99):
    data = np.zeros([int(image.data.shape[0] / zoom), int(image.data.shape[1] / zoom), 3], dtype='uint8')
    
    for i in range(min(3, image.data.shape[3])):
        chan_i = ndimage.zoom(image.data[:, :, :, i].mean(2).squeeze(), 1. / zoom)
        
        if scaling == 'min-max':
            chan_i = chan_i - chan_i.min()
            chan_i = (255 * chan_i / chan_i.max()).astype('uint8')
        elif scaling == 'percentile':
            lb = np.percentile(chan_i, 100 * (1 - scaling_factor))
            ub = np.percentile(chan_i, 100 * scaling_factor)
            chan_i = (255 * np.minimum(np.maximum(chan_i - lb, 0) / float(ub - lb), 1)).astype('uint8')
        
        data[:, :, i] = chan_i
    
    if image.data.shape[3] == 1:
        #make greyscale if a single colour channel
        data[:, :, 1] = data[:, :, 0]
        data[:, :, 2] = data[:, :, 0]
    
    return data

def image_to_cmy(image, *args, **kwargs):
    data = image_to_rgb(image, *args, **kwargs)
    if image.data.shape[3] == 1:
        return data  # grayscale, can't convert to CMY (rather than CMYK)
    
    output = np.zeros_like(data)
    
    k = 255 - data.max(2)
    
    output[:, :, 0] = 255 - (k + data[:, :, 0])
    output[:, :, 1] = 255 - (k + data[:, :, 1])
    output[:, :, 2] = 255 - (k + data[:, :, 2])
    
    return output
    
# jinja filters
# these output an image as a base64 encoded string suitable for incorporating into an html page

def base64_image(image, zoom=1.0, scaling='min-max', scaling_factor=0.99, colorblind_friendly=False, type='png'):
    """
    Jinga2 filter which converts an image stack (PYME.IO.image.ImageStack) into a base64 encoded image suitable for
    inline incorporation into an html page.
    
    Use with the following (or similar) html template code:
     
    .. code-block:: html
        
        <img src="data:image/png;base64,{{ image|base64_image(scaling='percentile') }}">
        
    can also output as jpg (or anything supported by both html and PIL) if the type parameter is set AND the html code
    is altered to give the correct mime type.
     
    Parameters
    ----------
    image: a PYME.io.image.ImageStack instance
    zoom: float, how large to zoom the image
    scaling: string, how to scale the intensity - one of 'min-max' or 'percentile'
    scaling_factor: float, Percentile only - which percentile to use
    colorblind_friendly: bool, Use cyan, magenta, and yellow rather than RGB to better accommodate colourblind users
    type: string, image type. See PIL / Pillow documentation for details.

    Returns
    -------
    
    a b64 encoded string

    """
    from io import BytesIO
    import base64
    
    try:
        from PIL import Image
    except ImportError:
        import Image
    
    if colorblind_friendly:
        im = image_to_cmy(image, zoom=zoom, scaling=scaling, scaling_factor = scaling_factor)
    else:
        im = image_to_rgb(image, zoom=zoom, scaling=scaling, scaling_factor=scaling_factor)

    
    outf = BytesIO()

    Image.fromarray(im, mode='RGB').save(outf,type.upper())
    s = outf.getvalue()
    outf.close()

    return base64.b64decode(s)