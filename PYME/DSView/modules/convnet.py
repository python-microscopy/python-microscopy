
def cnn_filter(dsviewer):
    from PYME.recipes.machine_learning import CNNFilter
    from PYME.DSView import ViewIm3D

    namespace = {'input': dsviewer.image}
    
    filter = CNNFilter()
    if filter.configure_traits(kind='modal'):
        
        filter.inputName='input'
        filter.execute(namespace)
        
        print('Finished running CNN filter')
        
        ViewIm3D(namespace[filter.outputName], parent=dsviewer, glCanvas=dsviewer.glCanvas)
        
    else:
        print('configure_traits failed')
        
        
        
def Plug(dsviewer):
    dsviewer.AddMenuItem('Processing', "Neural net filter", lambda e: cnn_filter(dsviewer))