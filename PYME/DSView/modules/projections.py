

def project(dsviewer, axis, type, crop=False):
    from PYME.recipes import processing
    from PYME.DSView import ViewIm3D
    
    if crop:
        from .cropping import crop_2D
        x0, x1, y0, y1, z0, z1 = dsviewer.do.sorted_selection
        roi = [[x0, x1 + 1],[y0, y1 +1], [0, dsviewer.image.data.shape[2]]]
        im = crop_2D(dsviewer.image, roi)
    else:
        im = dsviewer.image
    
    ViewIm3D(processing.Projection(kind=type, axis=axis).apply_simple(im), parent=dsviewer)


def Plug(dsviewer):
    # type: (PYME.DSView.dsviewer.DSViewFrame) -> object
    
    dsviewer.AddMenuItem('Processing>Project>Mean', 'x', lambda e : project(dsviewer, axis=0, type='Mean'))
    dsviewer.AddMenuItem('Processing>Project>Mean', 'y', lambda e: project(dsviewer, axis=1, type='Mean'))
    dsviewer.AddMenuItem('Processing>Project>Mean', 'z', lambda e: project(dsviewer, axis=2, type='Mean'))

    dsviewer.AddMenuItem('Processing>Project>Median', 'x', lambda e: project(dsviewer, axis=0, type='Median'))
    dsviewer.AddMenuItem('Processing>Project>Median', 'y', lambda e: project(dsviewer, axis=1, type='Median'))
    dsviewer.AddMenuItem('Processing>Project>Median', 'z', lambda e: project(dsviewer, axis=2, type='Median'))

    dsviewer.AddMenuItem('Processing>Project>Max', 'x', lambda e: project(dsviewer, axis=0, type='Max'))
    dsviewer.AddMenuItem('Processing>Project>Max', 'y', lambda e: project(dsviewer, axis=1, type='Max'))
    dsviewer.AddMenuItem('Processing>Project>Max', 'z', lambda e: project(dsviewer, axis=2, type='Max'))

    dsviewer.AddMenuItem('Processing>Project (Cropped)>Mean', 'x', lambda e: project(dsviewer, axis=0, type='Mean', crop=True))
    dsviewer.AddMenuItem('Processing>Project (Cropped)>Mean', 'y', lambda e: project(dsviewer, axis=1, type='Mean', crop=True))
    #dsviewer.AddMenuItem('Processing>Project (Cropped)>Mean', 'z', lambda e: project(dsviewer, axis=2, type='Mean', crop=True))

    dsviewer.AddMenuItem('Processing>Project (Cropped)>Median', 'x', lambda e: project(dsviewer, axis=0, type='Median', crop=True))
    dsviewer.AddMenuItem('Processing>Project (Cropped)>Median', 'y', lambda e: project(dsviewer, axis=1, type='Median', crop=True))
    #dsviewer.AddMenuItem('Processing>Project (Cropped)>Median', 'z', lambda e: project(dsviewer, axis=2, type='Median', crop=True))

    dsviewer.AddMenuItem('Processing>Project (Cropped)>Max', 'x', lambda e: project(dsviewer, axis=0, type='Max', crop=True))
    dsviewer.AddMenuItem('Processing>Project (Cropped)>Max', 'y', lambda e: project(dsviewer, axis=1, type='Max', crop=True))
    #dsviewer.AddMenuItem('Processing>Project (Cropped)>Max', 'z', lambda e: project(dsviewer, axis=2, type='Max', crop=True))
    
    