#!/usr/bin/python

##################
# read_kdf.py
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

from scipy.io import *
from scipy import *
#import pdb

def ReadKdfHeader(fid):

  count = 1
  #TypeString = ''
  #DimX = -1
  #DimY = -1
  #DimZ = -1
  #DimE = -1
  
  SegDims = -1*ones(5)
  SegOrders = -1*ones(5)
  
  
  MagicNumVer = fid.fread( 6, 'b');
  
  #print MagicNumVer
  
  if( not MagicNumVer[0] == 1) or (not MagicNumVer[1] == 3):
    DimX = -1
    error('Error: No Khoros file format\n');
  
  MachineType = fid.fread( 1, 'b')
  NumSets = fid.fread(1,'i')
  if not (NumSets == 1):
      error('Error: Can only read 1-Set Kdf data !\n')
  
  
  NumBlocks = fid.fread(1,'i')
  
  #// So called "header" is done
  #// Now read Object Attribute Block

  ObjAttrName = fid.fread( 1, 'b')
  if not (ObjAttrName == 0):
    error('Error : Unexpected char as ObjAttrName in Kdf !\n')
    DimX = -1
    return
  
  
  ObjSegNr = fid.fread( 1, 'i')
  #print ObjSegNr  
  #// Now Read Atrribute Block and Attributes for each Segment

  for i in range(ObjSegNr + 1):
    SegAttrName = ReadString(fid)
    #print SegAttrName
    
    SegAttrNum = fid.fread(1,'i')
    
    SegDim = fid.fread(1,'i')
    SegType = ReadString(fid)
    
    if(SegAttrNum == 0):
      count = SegAttrNum + 1
    else:
      count = SegAttrNum

    for k in range(count): #// What happens with more than one attribute ??
      if(SegDim > 1):  #// Read Sizes and Orders
        for j in range(SegDim):
          SegDims[j] = fid.fread(1,'i')
        
        for j in range(SegDim) :
          SegOrders[j] = fid.fread(1,'i')
          
          if not (SegOrders[j] == (j + 2)):
            print('Warning : Orders of segment nr. ? is anormal !\n')
          
        
        FixedDim = fid.fread( 1, 'i')
        FixedIndex = fid.fread( 1, 'i')

        if(SegAttrName == 'value'):
          DimX = SegDims[0]
          DimY = SegDims[1]
          DimZ = SegDims[2]
          DimE = SegDims[4]
          TypeString = SegType
          print(TypeString)
          
        
        #continue;
      else:
        if(SegDim == 1):
          #print SegType
          if(SegType == 'String'):
            AString = ReadString(fid)
           
          elif(SegType == 'Integer'):
            AnInt = fid.fread(1,'i');
          elif(SegType == 'Long'):
            ALong = fid.fread(1,'l');
          elif(SegType == 'Float'):
            AFloat = fid.fread(1,'f');
          elif(SegType == 'Double'):
            ADouble = fid.fread(1,'d');
          elif(SegType == 'Unsigned Long'):
            AULong = fid.fread(1,'u');
          else:
            DimX = -1;
            print('Unknown 1 Dim Segmenttype!\n')
            error()
            #return
          
    
    if not (SegAttrName == 'value'):
      EOA = ReadString(fid)
      if not (EOA == '<>'): 
          error('Error : EOA not found in KDF\n');
    else:
      if not(SegAttrNum == 0): 
            fid.fread( 9, 'f')
        #// Dirty
        #// Method for scipping something unknown, which causes trouble

  #print SegOrders
  return (AString,TypeString,DimX,DimY,DimZ,DimE) #;  %// from is now pointing to first segment data!
  
  

  
def ReadString(fid):

    count =0;
    s = '';
    c = '1';

    while (count < 100) and  not (c == '\0'):
        c = fid.fread( 1, 'c')
        s = s + c
    
        count += 1
        
    #s = char(s(1:(length(s)-1)));
    return s.strip('\0')

def ReadKdfData(fname):
    fid = fopen(fname)
    
    (AString,TypeString,DimX,DimY,DimZ,DimE) = ReadKdfHeader(fid)
    
    count = DimX*DimY*DimZ*DimE
    
#     if(TypeString == 'Integer'):
#         d = fid.fread(count,'i');
#     elif(TypeString == 'Long'):
#         d = fid.fread(count,'l');
#     elif(TypeString == 'Float'):
#         d = fid.fread(count,'f');
#     elif(TypeString == 'Double'):
#         d = fid.fread(count,'d');
#     elif(TypeString == 'Unsigned Long'):
#         d = fid.fread(count,'u');
#     elif(TypeString == 'Unsigned Short'):
#         d = fid.fread(count,'w');
#     elif(TypeString == 'Unsigned Byte'):
#         d = fid.fread(count,'b');
#     else:
#         DimX = -1;
#         print('Unknown Datatype\n')
#         error()
    
    if(TypeString == 'Integer'):
        rtype = 'i'
    elif(TypeString == 'Long'):
        rtype = 'l'
    elif(TypeString == 'Float'):
        rtype = 'f'
    elif(TypeString == 'Double'):
        rtype = 'd'
    elif(TypeString == 'Unsigned Long'):
        rtype = 'u'
    elif(TypeString == 'Unsigned Short'):
        rtype = 'w'
    elif(TypeString == 'Unsigned Byte'):
        rtype = 'b'
    else:
        DimX = -1;
        print('Unknown Datatype\n')
        error()
    
    d = zeros((DimX,DimY,DimZ,DimE), rtype)
    
    for i in range(DimE):
        for j in range(DimZ):
            for k in range(DimY):
                #for l in range(DimX):
                d[:,k,j,i] = fid.fread(DimX, rtype)
    return d
    