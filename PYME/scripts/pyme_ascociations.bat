assoc .h5=PYME.RawData
assoc .h5r=PYME.AnalysedData
assoc .psf=PYME.PSF
assoc .sf=PYME.ShiftField
assoc .md=PYME.Metadata

#most people probably don't want to ascociate .tif with PYME
#assoc .tiff=PYME.Tiff
#assoc .tif=PYME.Tiff

ftype PYME.AnalysedData=VisGUI %*
ftype PYME.RawData=dh5view %*
ftype PYME.PSF=dh5view %*
#ftype PYME.Tiff=dh5view %*
ftype PYME.Metadata=dh5view %*