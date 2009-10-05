#!/usr/bin/python

##################
# logparser.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

class logparser:
    def __init__(self):
        pass
    
    def parse(self, s):
        s = s.split('\n')
        dic = {};
        
        curdic = dic;
        
        for entry in s:
            if entry == '':
                pass 
            elif entry[0] == '[':
                newdic = {}
                curdic = newdic
                dic[entry.strip()[1:-1]] = newdic
            elif entry[0] == '#':
                pass
            else:
                e = entry.split('=')
                val  = ''
                
                #Try to interpret the value as an int, then as a float. 
                #If that doesn't work then store as string
                try:
                    val = int(e[1].strip())
                except ValueError:
                    try:
                        val = float(e[1].strip())
                    except ValueError:
                        val = e[1].strip()
                
                curdic[e[0]] = val
                
        return dic
    
class confocal_parser(logparser):
    def __init__(self, log_s):
        self.dic = self.parse(log_s)
        
        self.Width = self.dic['GLOBAL']['ImageWidth']
        self.Height = self.dic['GLOBAL']['ImageLength']
        self.Depth = self.dic['GLOBAL']['NumOfFrames']
        self.NumChannels = self.dic['FILTERSETTING1']['NumOfVisualisations']
        
        self.VoxelSize = (self.dic['GLOBAL']['VoxelSizeX'],self.dic['GLOBAL']['VoxelSizeY'],self.dic['GLOBAL']['VoxelSizeZ'])
        self.VoxelX = self.VoxelSize[0]
        self.VoxelY = self.VoxelSize[1]
        self.VoxelZ = self.VoxelSize[2]
        
        self.Averaging = self.dic['GLOBAL']['Accu']
        
class logwriter:
    def __init__(self):
        pass
    
    def write(self, log):
        #s = s.split('\n')
        #dic = {};
        
        #curdic = dic;
        
        s = ''
        
        cats = log.keys()
        cats.sort()
        for category in cats:
            s = s + '[%s]\n' % category
            
            entries = log[category].keys()
            entries.sort()
            for entry in entries:
                s = s + '%s=%s\n' % (entry, log[category][entry])
                
        return s
            