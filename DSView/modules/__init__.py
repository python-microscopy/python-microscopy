__author__="david"
__date__ ="$3/02/2011 9:42:00 PM$"

def load(mode, dsviewer):
    '''install the relevant modules for a particular mode'''

    if mode == 'lite': #don't load any modules
        return
    
    #always load the playback, deconvolution & tiling modules
    import playback, deconvolution, tiling
    dsviewer.player = playback.player(dsviewer)
    dsviewer.deconvolver = deconvolution.deconvolver(dsviewer)
    dsviewer.tiler = tiling.tiler(dsviewer)

    if mode == 'LM':
        #load the localisation module
        import LMAnalysis
        dsviewer.LMAnalyser = LMAnalysis.LMAnalyser(dsviewer)
    else:
        if mode == 'blob':
            import blobFinding
            dsviewer.blobFinder = blobFinding.blobFinder(dsviewer)

        import psfExtraction
        dsviewer.psfExtractor = psfExtraction.psfExtractor(dsviewer)