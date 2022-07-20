#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

if __name__ == '__main__':
   #load numpy
   import numpy as np

   #load the pytables library - this produces HDF5 formatted data (which should be
   #able to be read using other HDF5 readers such as those distributed with Matlab)
   #note that it also encodes numpy data types (and some other stuff) in file metadata
   #so getting e.g. Matlab to write PYME compatible HDF might be a bit of a challenge
   import tables

   #this will let us write microscope and acquisition related data into the file
   #I'm not sure how much of this VisGUI actually assumes is present when dealing 
   #with h5r files, so you might not need to use this
   from PYME.IO import MetaDataHandler


   #create a new file
   h5ResultsFile = tables.open_file(resultsFilename, 'w')

   #create a metadata handler for the results file to allow us to add entries
   resultsMDH = MetaDataHandler.HDFMDHandler(h5ResultsFile)

   #example metadata entry - this one is almost certainly not needed, but you might
   #run across some errors where it's looking for entries which aren't present
   #the metadata is accessed in VisGUI by looking at pipeline.mdh 
   resultsMDH['Camera.ADOffset'] = 107.

   #create events table - this table is for recording things which happen during the 
   #acquisition such as changes to laser power or focus stepping. I have a feeeling
   #that VisGUI complains if this isn't there, but you can safely leave the table
   #empty.
   from PYME.IO.events import SpoolEvent
      
   resultsEvents = h5ResultsFile.create_table(h5ResultsFile.root, 'Events', SpoolEvent,filters=tables.Filters(complevel=5, shuffle=True))




   #This is the business end of things - the 'FitResults' table is where the actual
   #molecule positions get saved. The results should be in a numpy record array (ie with a 
   #dtype specifying column names) 

   #this is the data type I use for 2D fits (see PYME/localization.FitFactories/LatGaussFitFR.py). 
   #Some of this is superfluous (notably 
   #resultCode and slicesUsed) and you can add whatever additional information you like
   #and it will be available in VisGUI. That said, the format is somewhat more prescriptive
   #than for text files. The x positions must be in results['FitResults']['x0'], the y 
   #positions in results['FitResults']['y0'], and the z positions (if present) in 
   #results['FitResults']['z0']. The amplitude and width should be in 'A' and 'sigma'
   #in order to be recognized (although they don't strictly need to be there - 3D 
   #fitting, for example, doesn't define sigma). Errors, if available, should go in
   #results['fitError']['...']. The frame number should go in 'tIndex'
   #NB - all positions should be in nm
   fresultdtype=[('tIndex', '<i4'),
               ('fitResults', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]),
               ('fitError', [('A', '<f4'),('x0', '<f4'),('y0', '<f4'),('sigma', '<f4'), ('background', '<f4'),('bx', '<f4'),('by', '<f4')]), 
               ('resultCode', '<i4'), 
               ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])])]


   #an example of how you might fill the results - this is a little laborious - have 
   #a look at e.g. PYME/localization.FitFactories/LatGaussFitFR.py to see how I do it
   result1 = np.zeros(1, fresultdtype)
   result1['fitResults']['x0'] = 5.34
   result1['fitResults']['y0'] = 343.87
   #etc ....

   result2 = np.zeros(1, fresultdtype)
   #etc ....

   results = np.hstack([result1, result2])


   #create a new table which matches the datatype of the results and populate it
   h5ResultsFile.create_table(h5ResultsFile.root, 'FitResults', results, filters=tables.Filters(complevel=5, shuffle=True), expectedrows=500000)

   #add some more results (the results array can be any length, but adding in chuncks
   #as the data comes in avoids having to keep it all in memory)
   moreResults = np.zeros(20, fresultdtype)
   h5ResultsFile.root.FitResults.append(moreResults)

   #flush the buffers in the pytables library
   h5ResultsFile.flush()

   #close the file
   h5ResultsFile.close()

