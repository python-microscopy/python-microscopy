#!/usr/bin/python

##################
# MetaDataHandler.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

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

        return self.h5file.getNodeAttr('/'.join(['', 'MetaData']+ ep), en)
        


    def getEntryNames(self):
        entryNames = []
        for a in [self.md] + list(self.md._f_walkNodes()):
            entryNames.extend(['.'.join(a._v_pathname.split('/')[2:] +[ i]) for i in a._v_attrs._f_list()])

        return entryNames

    def copyEntriesFrom(self, mdToCopy):
        for en in mdToCopy.getEntryNames():
            self.setEntry(en, mdToCopy.getEntry(en))

    def mergeEntriesFrom(self, mdToCopy):
        #only copies values if not already defined
        for en in mdToCopy.getEntryNames():
            if not en in self.getEntryNames():
                self.setEntry(en, mdToCopy.getEntry(en))

    def __repr__(self):
        s = ['%s: %s' % (en, self.getEntry(en)) for en in self.getEntryNames()]
        return '<%s>:\n\n' % self.__class__.__name__ + '\n'.join(s)

class QueueMDHandler:
    def __init__(self, tq, queueName, mdToCopy=None):
        self.tq = tq
        self.queueName = queueName
        self.md = None

        if not mdToCopy == None:
            self.copyEntriesFrom(mdToCopy)


    def setEntry(self,entryName, value):
        self.tq.setQueueMetaData(self.queueName, entryName, value)


    def getEntry(self,entryName):
        #print entryName
        return self.tq.getQueueMetaData(self.queueName, entryName)


    def getEntryNames(self):
        return self.tq.getQueueMetaDataKeys(self.queueName)
        

    def copyEntriesFrom(self, mdToCopy):
        for en in mdToCopy.getEntryNames():
            self.setEntry(en, mdToCopy.getEntry(en))

    def mergeEntriesFrom(self, mdToCopy):
        #only copies values if not already defined
        for en in mdToCopy.getEntryNames():
            if not en in self.getEntryNames():
                self.setEntry(en, mdToCopy.getEntry(en))

    def __repr__(self):
        s = ['%s: %s' % (en, self.getEntry(en)) for en in self.getEntryNames()]
        return '<%s>:\n\n' % self.__class__.__name__ + '\n'.join(s)


class NestedClassMDHandler:
    def __init__(self, mdToCopy=None):
        if not mdToCopy == None:
            self.copyEntriesFrom(mdToCopy)


    def setEntry(self,entryName, value):
        entPath = entryName.split('.')
        if len(entPath) == 1: #direct child of this node
            self.__dict__[entPath[0]] = value
        else:
            if not entPath[0] in dir(self):
                self.__dict__[entPath[0]] = NestedClassMDHandler()
            self.__dict__[entPath[0]].setEntry('.'.join(entPath[1:]), value)


    def getEntry(self,entryName):
        return eval('self.'+entryName)


    def getEntryNames(self):
        en = []
        for k in self.__dict__.keys():
            if self.__dict__[k].__class__ == NestedClassMDHandler:
                en += [k + '.' + kp for kp in self.__dict__[k].getEntryNames()]
            else:
                en.append(k)

        return en


    def copyEntriesFrom(self, mdToCopy):
        for en in mdToCopy.getEntryNames():
            self.setEntry(en, mdToCopy.getEntry(en))

    def mergeEntriesFrom(self, mdToCopy):
        #only copies values if not already defined
        for en in mdToCopy.getEntryNames():
            if not en in self.getEntryNames():
                self.setEntry(en, mdToCopy.getEntry(en))

    def __repr__(self):
        s = ['%s: %s' % (en, self.getEntry(en)) for en in self.getEntryNames()]
        return '<%s>:\n\n' % self.__class__.__name__ + '\n'.join(s)

from xml.dom.minidom import getDOMImplementation, parse

class XMLMDHandler:
    def __init__(self, filename = None, mdToCopy=None):
        if not filename == None:
            #loading an existing file
            self.doc = parse(filename)
            self.md = self.doc.documentElement.getElementsByTagName('MetaData')[0]
        else:
            #creating a new document
            self.doc = getDOMImplementation().createDocument(None, 'PYMEImageData', None)
            self.md = self.doc.createElement('MetaData')
            self.doc.documentElement.appendChild(self.md)

        if not mdToCopy == None:
            self.copyEntriesFrom(mdToCopy)

    def writeXML(self, filename):
        f = open(filename, 'w')
        f.write(self.doc.toprettyxml())
        f.close()


    def setEntry(self,entryName, value):
        entPath = entryName.split('.')

        node = self.md
        while len(entPath) >= 1:
            el = [e for e in node.childNodes if e.tagName == entPath[0]]
            if len(el) == 0:
                #need to create node
                newNode = self.doc.createElement(entPath[0])
                node.appendChild(newNode)
                node = newNode
            else:
                node = el[0]

            entPath.pop(0)

        node.setAttribute('class', type(value).__name__)
        node.setAttribute('value', repr(value))


#    def getEntry(self,entryName):
#        entPath = entryName.split('.')
#        en = entPath[-1]
#        ep = entPath[:-1]
#
#        return eval('self.'+entryName)


#    def getEntryNames(self):
#        en = []
#        for k in self.__dict__.keys():
#            if self.__dict__[k].__class__ == NestedClassMDHandler:
#                en += [k + '.' + kp for kp in self.__dict__[k].getEntryNames()]
#            else:
#                en.append(k)
#
#        return en


    def copyEntriesFrom(self, mdToCopy):
        for en in mdToCopy.getEntryNames():
            self.setEntry(en, mdToCopy.getEntry(en))

#    def mergeEntriesFrom(self, mdToCopy):
#        #only copies values if not already defined
#        for en in mdToCopy.getEntryNames():
#            if not en in self.getEntryNames():
#                self.setEntry(en, mdToCopy.getEntry(en))
#
#    def __repr__(self):
#        s = ['%s: %s' % (en, self.getEntry(en)) for en in self.getEntryNames()]
#        return '<%s>:\n\n' % self.__class__.__name__ + '\n'.join(s)
