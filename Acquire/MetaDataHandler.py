#!/usr/bin/python
'''
define meta data handlers - these should expose three methods,

setEntry(self, entryName, value) 
getEntry(self, entryName)

which set and get a meta data value. entryName is a name of the form:
"key" or "group.key" or "group.subgroup.key" etc...

and 

getEntryNames(self)

which returns a list of entry names to help with copying data between handlers
'''

#lists where bits of hardware can register the fact that they are capable of 
#providing metadata, by appending a function with the signature:
#genMetadata(MetaDataHandler)
provideStartMetadata = []
provideStopMetadata = []


class HDFMDHandler:
    def __init__(self, h5file, mdToCopy=None):
        self.h5file = h5file
        self.md = None

        if self.h5file.__contains__('/MetaData'):
            self.md = self.h5file.root.MetaData
        else:
            self.md = self.h5file.createGroup(self.h5file.root, 'MetaData')

        if not mdToCopy == None:
            self.copyEntriesFrom(mdToCopy)


    def setEntry(self,entryName, value):
        entPath = entryName.split('.')
        en = entPath[-1]
        ep = entPath[:-1]

        currGroup = self.h5file._getOrCreatePath('/'.join(['', 'MetaData']+ ep), True)
        currGroup._f_setAttr(en, value)
        self.h5file.flush()


    def getEntry(self,entryName):
        entPath = entryName.split('.')
        en = entPath[-1]
        ep = entPath[:-1]

        currGroup = self.h5file.getNode('/'.join(['', 'MetaData']+ ep))
        currGroup._f_getAttr(en)


    def getEntryNames(self):
        entryNames = []
        for a in [self.md] + list(self.md._f_walkNodes()):
            entryNames.extend(['.'.join(a._v_pathname.split('/')[2:] +[ i]) for i in a._v_attrs._f_list()])

        return entryNames

    def copyEntriesFrom(self, mdToCopy):
        for en in mdToCopy.getEntryNames:
            self.setEntry(en, mdToCopy.getEntry(en))
        
       
