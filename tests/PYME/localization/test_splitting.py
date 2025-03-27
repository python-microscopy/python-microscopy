def test_split_image():
    from PYME.IO import MetaDataHandler
    from PYME.localization import splitting
    import numpy as np

    # create a fake image with pixel values equal to the pixel coordinates
    datax, datay = np.mgrid[0:1000, 0:1000]

    # create a fake metadata handler
    mdh = MetaDataHandler.DictMDHandler()
    mdh['Camera.ROIWidth'] = 1000
    mdh['Camera.ROIHeight'] = 1000
    mdh['Camera.ROIOriginX'] = 0
    mdh['Camera.ROIOriginY'] = 0
    mdh['Splitter.Channel0ROI'] = [0, 0, 1000, 500]
    mdh['Splitter.Channel1ROI'] = [0, 500, 1000, 500]
    mdh['Splitter.Flip'] = False

    # split the image
    spx = splitting.split_image(mdh, datax)
    spx0 = spx[:,:,0].squeeze()
    spx1 = spx[:,:,1].squeeze()
    spy = splitting.split_image(mdh, datay)
    spy0 = spy[:,:,0].squeeze()
    spy1 = spy[:,:,1].squeeze()

    # check that the split image is correct
    assert(spx0.shape == (1000, 500))
    assert(spx1.shape == (1000, 500))
    assert(spx0[0, 0] == 0)
    assert(spx1[0, 0] == 0)
    assert(spy0[0, 0] == 0)
    assert(spy1[0, 0] == 500)
    assert(spx0[-1, -1] == 999)
    assert(spx1[-1, -1] == 999)
    assert(spy0[-1, -1] == 499)
    assert(spy1[-1, -1] == 999)


def test_split_image_roi():
    '''
    If a ROI is specified, the split image should correspond to the maximum size shared region
    between the two channels given the ROI.
    '''
    from PYME.IO import MetaDataHandler
    from PYME.localization import splitting
    import numpy as np

    # create a fake image with pixel values equal to the pixel coordinates
    datax, datay = np.mgrid[0:1000, 0:1000]
    datax = datax[100:950, 100:950]
    datay = datay[100:950, 100:950]

    # create a fake metadata handler
    mdh = MetaDataHandler.DictMDHandler()
    mdh['Camera.ROIWidth'] = 850
    mdh['Camera.ROIHeight'] = 850
    mdh['Camera.ROIOriginX'] = 100
    mdh['Camera.ROIOriginY'] = 100
    mdh['Splitter.Channel0ROI'] = [0, 0, 1000, 500]
    mdh['Splitter.Channel1ROI'] = [0, 500, 1000, 500]
    mdh['Splitter.Flip'] = False

    # split the image
    spx = splitting.split_image(mdh, datax)
    spx0 = spx[:,:,0].squeeze()
    spx1 = spx[:,:,1].squeeze()
    spy = splitting.split_image(mdh, datay)
    spy0 = spy[:,:,0].squeeze()
    spy1 = spy[:,:,1].squeeze()

    print(spx0.shape, spx1.shape)

    # check that the split image is correct
    assert(spx0.shape == (850, 350))
    assert(spx1.shape == (850, 350))
    assert(spx0[0, 0] == 100)
    assert(spx1[0, 0] == 100)
    assert(spy0[0, 0] == 100)
    assert(spy1[0, 0] == 600)
    assert(spx0[-1, -1] == 949)
    assert(spx1[-1, -1] == 949)
    assert(spy0[-1, -1] == 449)
    assert(spy1[-1, -1] == 949)

def test_split_image_roi_lr():
    '''
    If a ROI is specified, the split image should correspond to the maximum size shared region
    between the two channels given the ROI.
    '''
    from PYME.IO import MetaDataHandler
    from PYME.localization import splitting
    import numpy as np

    # create a fake image with pixel values equal to the pixel coordinates
    datax, datay = np.mgrid[0:1000, 0:1000]
    datax = datax[100:950, 100:950]
    datay = datay[100:950, 100:950]

    # create a fake metadata handler
    mdh = MetaDataHandler.DictMDHandler()
    mdh['Camera.ROIWidth'] = 850
    mdh['Camera.ROIHeight'] = 850
    mdh['Camera.ROIOriginX'] = 100
    mdh['Camera.ROIOriginY'] = 100
    mdh['Splitter.Channel0ROI'] = [0, 0, 500, 1000]
    mdh['Splitter.Channel1ROI'] = [500, 0, 500, 1000]
    mdh['Splitter.Flip'] = False

    # split the image
    spx = splitting.split_image(mdh, datax)
    spx0 = spx[:,:,0].squeeze()
    spx1 = spx[:,:,1].squeeze()
    spy = splitting.split_image(mdh, datay)
    spy0 = spy[:,:,0].squeeze()
    spy1 = spy[:,:,1].squeeze()

    print(spx0.shape, spx1.shape)

    # check that the split image is correct
    assert(spx0.shape == (350, 850))
    assert(spx1.shape == (350, 850))
    assert(spx0[0, 0] == 100)
    assert(spx1[0, 0] == 600)
    assert(spy0[0, 0] == 100)
    assert(spy1[0, 0] == 100)
    assert(spx0[-1, -1] == 449)
    assert(spx1[-1, -1] == 949)
    assert(spy0[-1, -1] == 949)
    assert(spy1[-1, -1] == 949)

def test_split_image_roi_flip():
    '''
    If a ROI is specified, the split image should correspond to the maximum size shared region
    between the two channels given the ROI.
    '''
    from PYME.IO import MetaDataHandler
    from PYME.localization import splitting
    import numpy as np

    # create a fake image with pixel values equal to the pixel coordinates
    datax, datay = np.mgrid[0:1000, 0:1000]
    datax = datax[100:950, 100:950]
    datay = datay[100:950, 100:950]

    # create a fake metadata handler
    mdh = MetaDataHandler.DictMDHandler()
    mdh['Camera.ROIWidth'] = 850
    mdh['Camera.ROIHeight'] = 850
    mdh['Camera.ROIOriginX'] = 100
    mdh['Camera.ROIOriginY'] = 100
    mdh['Splitter.Channel0ROI'] = [0, 0, 1000, 500]
    mdh['Splitter.Channel1ROI'] = [0, 500, 1000, 500]
    mdh['Splitter.Flip'] = True

    # split the image
    spx = splitting.split_image(mdh, datax)
    spx0 = spx[:,:,0].squeeze()
    spx1 = spx[:,:,1].squeeze()
    spy = splitting.split_image(mdh, datay)
    spy0 = spy[:,:,0].squeeze()
    spy1 = spy[:,:,1].squeeze()

    print(spx0.shape, spx1.shape)

    # check that the split image is correct
    assert(spx0.shape == (850, 400))
    assert(spx1.shape == (850, 400))
    assert(spx0[0, 0] == 100)
    assert(spx1[0, 0] == 100)
    assert(spy0[0, 0] == 100)
    assert(spy1[0, 0] == 899)
    assert(spx0[-1, -1] == 949)
    assert(spx1[-1, -1] == 949)
    assert(spy0[-1, -1] == 499)
    assert(spy1[-1, -1] == 500)