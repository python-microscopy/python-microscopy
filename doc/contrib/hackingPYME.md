## Code fragments to access your data in PYME GUI Apps

The code fragments below are a random selection of tips to access your data from the console/shell/command line within `dh5view` and `Visgui` etc.

### Accessing data in dh5view

`do` is the top level object. It has a member `do.ds` which is the data source. Use that to access your data:

	ims = do.ds[:,:,100:105]

### Getting at h5 stacks from the command line in python etc

The following code fragments show how to access an h5 from within a script or directly from a command line (e.g. `ipython`):

	import PYME.DSView.image as im
	stack = im.ImageStack(filename='/Full/Path/14_6_series_B.h5')`

	ims = stack.data[:,:,1000:1040]

### Note when plotting histograms

Flatten the data before histograming, e.g.

	res = hist(imm.ravel(),bins=2000)

Otherwise we are apparently plotting many histograms as we thread over all 1D sub-vectors. Flattening first, for example using the numpy `ravel()` command, should speed things up considerably.

### Reading and writing tiffs

	import PYME.FileUtils.saveTiffStack as st
	from PYME.gohlke import tifffile
	
	# a simple definition but using st.saveTiffMultipage
	# directly works just as well
	def tiffsave(data,fname):
    	st.saveTiffMultipage(data,fname)

	# open a tiff file and force conversion to float type (32 bit)
	rdata = tifffile.TIFFfile(dname).asarray().astype('f')
	
### MetaData

Metadata can be accessed in the `dh5view` console using the global variable `mdv` (MetaDataPanel) which has a member `mdh`:

	mdv.mdh['voxelsize']

