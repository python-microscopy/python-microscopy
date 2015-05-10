#!/usr/bin/python

##################
# write_kdf.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

from scipy import *
#from scipy.io import *
import scipy

structTypes = {'b':'s',
               'uchar':'s',
               'int32':'i'}


if int(scipy.version.version.split('.')[1]) > 5: 
  from scipy.io.fopen import *
  from scipy.io.numpyio import *
else:
  from scipy.io import *
  import struct
  import numpy
  class fopen:
    def __init__(self, fname, mode='r'):
      self.fid = open(fname, mode)
    
    def fread(self, *args):
      return fread(self.fid, *args)

    def write(self, *args):
      if isinstance(args[0], numpy.ndarray): 
        return fwrite(self.fid, args[0].size, *args)
      else:
        f = structTypes[args[1]]
        if f == 's':
          f = '%ds' % len(args[0])
        str = struct.pack(f, args[0])
        #print str
        return self.fid.write(str)
    def close(self):
      self.fid.close()

def WriteKhorosHeader(fid, Date, TypeString, ValueDimX, ValueDimY, ValueDimZ, ValueDimE, ValueDimT):


  #if (ValueDimX <= 0 );  ValueDimX=1; end
  #if (ValueDimY <= 0 );  ValueDimY=1; end
  #if (ValueDimZ <= 0 );  ValueDimZ=1; end
  #if (ValueDimE <= 0 );  ValueDimE=1; end

  MagicNumVer = '\x01\x03\x13\x5E\x00\x02';
  
  MachineType = '\x41';   # This is PC format
  
  NumSets = 1
  NumBlocks = 2

  ObjAttrName =''
  ObjSegNr = 2

  SegAttrName = 'date'
  SegAttrNum = 1

  DateDim = 1
  DateType = 'String'
  EOA = '<>'            # end of attribute tag

  Seg2AttrName = 'locationGrid'
  Seg2AttrNum = 1

  LocationDim = 1
  LocationType = 'Integer'
  Location = 0
  EOA2 ='<>'            # end of attribute tag

  ValueAttrName ='value'
  ValueAttrNum = 0

  ValueDim = 5
  # const int  ValueDimT = 1;
  ValueOrder = [2,3,4,5,6]
  FixedDim = -1 
  FixedIndex = -1

  fid.write(MagicNumVer, 'b')
  fid.write(MachineType,'uchar')
  fid.write(NumSets, 'int32')
  fid.write(NumBlocks, 'int32')

  fid.write(ObjAttrName + '\0', 'uchar')
  fid.write(ObjSegNr, 'int32');

  fid.write(SegAttrName + '\0', 'uchar');
  fid.write(SegAttrNum, 'int32');
  fid.write(DateDim, 'int32');
  fid.write(DateType + '\0', 'uchar');
  
  #to->write((char *) Date,strlen(Date)+1);
  fid.write(Date + '\0', 'uchar');
  
  fid.write(EOA + '\0', 'uchar');

  fid.write(Seg2AttrName + '\0', 'uchar');
  fid.write(Seg2AttrNum, 'int32');
  fid.write(LocationDim, 'int32');
  fid.write(LocationType + '\0', 'uchar');
  fid.write(Location, 'int32');
  fid.write(EOA2 + '\0', 'uchar');

  fid.write(ValueAttrName + '\0', 'uchar');
  fid.write(ValueAttrNum, 'int32');
  fid.write(ValueDim, 'int32');
  
  #to->write((char *) TypeString,strlen(TypeString)+1);
  fid.write(TypeString + '\0', 'uchar');
  
  fid.write(ValueDimX, 'int32');
  fid.write(ValueDimY, 'int32');
  fid.write(ValueDimZ, 'int32');
  fid.write(ValueDimT, 'int32');
  fid.write(ValueDimE, 'int32');
  for i in range(5):
    fid.write(ValueOrder[i], 'int32');
  

  fid.write(FixedDim, 'int32');
  fid.write(FixedIndex, 'int32');

      
      
def WriteKhorosData(fname, d):
    
    fid = fopen(fname, 'wb');
    
    siz = ones(5, 'i')
    
    siz2 = shape(d)
    
    siz[0:len(siz2)] = siz2
    
    (DimX,DimY,DimZ,DimE, DimT) = siz
    
    if (d.dtype == 'b'):           
        TypeString = 'Unsigned Byte'
    elif (d.dtype == 'i'):       
        TypeString = 'Integer'
    elif (d.dtype == 'l'):       
        TypeString = 'Long'
    elif (d.dtype == 'f'):      
        TypeString = 'Float' 
    elif (d.dtype == 'd'):      
        TypeString = 'Double' 
    elif (d.dtype == 'H'):
        TypeString = 'Unsigned Short'
    else:
        fid.close()
        print(('Dont know about %ss ... fixme' % (d.dtype,)))
        error()
    
    
    WriteKhorosHeader(fid, 'hello', TypeString, DimX, DimY, DimZ, DimE, DimT)
    
    
    #fid.write(d)
    d1 = d.reshape(siz)

    for i in range(DimE):
      for j in range(DimZ):
        for k in range(DimY):
          a = fid.write(d1[:,k,j,i,0])#, d1.dtype)
    
    fid.close()
    
    return
