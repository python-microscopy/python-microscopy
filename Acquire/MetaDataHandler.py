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
from UserDict import DictMixin

#lists where bits of hardware can register the fact that they are capable of 
#providing metadata, by appending a function with the signature:
#genMetadata(MetaDataHandler)
provideStartMetadata = []
provideStopMetadata = []

def instanceinlist(cls, list):
    for c in list:
        if isinstance(cls, c):
            return True

    return False
    

class MDHandlerBase(DictMixin):
    #base class to make metadata behave like a dictionary
    def __setitem__(self, name, value):
        self.setEntry(name, value)

    def __getitem__(self, name):
        return self.getEntry(name)

    def keys(self):
        return self.getEntryNames()

    def copyEntriesFrom(self, mdToCopy):
        for en in mdToCopy.getEntryNames():
            #print en
            self.setEntry(en, mdToCopy.getEntry(en))
        #self.update(mdToCopy)

    def mergeEntriesFrom(self, mdToCopy):
        #only copies values if not already defined
        for en in mdToCopy.getEntryNames():
            if not en in self.getEntryNames():
                self.setEntry(en, mdToCopy.getEntry(en))

    def __repr__(self):
        s = ['%s: %s' % (en, self.getEntry(en)) for en in self.getEntryNames()]
        return '<%s>:\n\n' % self.__class__.__name__ + '\n'.join(s)

    def WriteSimple(self, filename):
        '''Writes out metadata in simplfied format'''
        import cPickle
        import numpy as np
        s = ['#PYME Simple Metadata v1\n']

        for en in self.getEntryNames():
            val = self.getEntry(en)

            if val.__class__ in [str, unicode] or np.isscalar(val): #quote string
                val = repr(val)
            elif not val.__class__ in [int, float, list, dict]: #not easily recovered from representation
                val = "cPickle.loads('''%s''')" % cPickle.dumps(val)

            s.append("md['%s'] = %s\n" % (en, val))
        
        f = open(filename, 'w')
        f.writelines(s)
        f.close()


class HDFMDHandler(MDHandlerBase):
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

        res =  self.h5file.getNodeAttr('/'.join(['', 'MetaData']+ ep), en)
        
        #dodgy hack to get around a problem with zero length strings not
        #being picklable if they are numpy (rather than pure python) types
        #this code should convert a numpy empty string into a python empty string
        if res == '':
            return ''
        
        return res
        


    def getEntryNames(self):
        entryNames = []
        for a in [self.md] + list(self.md._f_walkNodes()):
            entryNames.extend(['.'.join(a._v_pathname.split('/')[2:] +[ i]) for i in a._v_attrs._f_list()])

        return entryNames

class QueueMDHandler(MDHandlerBase):
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
        


class NestedClassMDHandler(MDHandlerBase):
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
            if hasattr(self.__dict__[k], 'getEntryNames') and not self.__dict__[k].__module__ == 'Pyro.core':
                en += [k + '.' + kp for kp in self.__dict__[k].getEntryNames()]
            else:
                en.append(k)

        return en


from xml.dom.minidom import getDOMImplementation, parse
#from xml.sax.saxutils import escape, unescape
import base64

class SimpleMDHandler(NestedClassMDHandler):
    '''simple metadata format - consists of a python script with a .md extension
    which adds entrys using the dictionary syntax to a metadata handler called md'''

    def __init__(self, filename = None, mdToCopy=None):
        if not filename == None:
            import cPickle
            #loading an existing file
            md = self
            fn = __file__
            globals()['__file__'] = filename
            execfile(filename)
            globals()['__file__'] = fn

        if not mdToCopy == None:
            self.copyEntriesFrom(mdToCopy)

    def write(self, filename):
        s = ''
        for en in self.getEntryNames():
            s += "md['%s'] = %s\n" % (en, self.getEntry(en))

        fid = open(filename, 'w')
        fid.write(s)
        fid.close()

    

class XMLMDHandler(MDHandlerBase):
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
        import cPickle
        import numpy as np
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

        #typ = type(value) #.__name__
        
        if isinstance(value, float):
            node.setAttribute('class', 'float')
            node.setAttribute('value', str(value))
        elif isinstance(value, int):
            node.setAttribute('class', 'int')
            node.setAttribute('value', str(value))
        elif isinstance(value, str):
            node.setAttribute('class', 'str')
            node.setAttribute('value', value)
        elif isinstance(value, unicode):
            node.setAttribute('class', 'unicode')
            node.setAttribute('value', value)
        elif np.isscalar(value):
            node.setAttribute('class', 'float')
            node.setAttribute('value', str(value)) 
        else: #pickle more complicated structures
            node.setAttribute('class', 'pickle')
            print value, cPickle.dumps(value)
            node.setAttribute('value', base64.b64encode((cPickle.dumps(value))))


    def getEntry(self,entryName):
        import cPickle
        entPath = entryName.split('.')

        node = self.md
        while len(entPath) >= 1:
            el = [e for e in node.childNodes if e.nodeName == entPath[0]]
            if len(el) == 0:
                #node not there
                raise RuntimeError('Requested node not found')
            else:
                node = el[0]

            entPath.pop(0)

        cls = node.getAttribute('class')
        val = node.getAttribute('value')

        if cls == 'int':
            val = int(val)
        if cls == 'float':
            val = float(val)
        if cls == 'pickle':
            #return None
            val = cPickle.loads(base64.b64decode(val))

        return val


    def getEntryNames(self):
        elements = self.md.getElementsByTagName('*')

        en = []

        for e in elements:
            if not e.hasChildNodes(): #we are at the end of the tree
                n = e.nodeName #starting name
                while not e.parentNode == self.md:
                    e = e.parentNode
                    n = '.'.join((e.nodeName, n))

                en.append(n)        

        return en


#    def copyEntriesFrom(self, mdToCopy):
#        for en in mdToCopy.getEntryNames():
#            self.setEntry(en, mdToCopy.getEntry(en))

#    def mergeEntriesFrom(self, mdToCopy):
#        #only copies values if not already defined
#        for en in mdToCopy.getEntryNames():
#            if not en in self.getEntryNames():
#                self.setEntry(en, mdToCopy.getEntry(en))
#
#    def __repr__(self):
#        s = ['%s: %s' % (en, self.getEntry(en)) for en in self.getEntryNames()]
#        return '<%s>:\n\n' % self.__class__.__name__ + '\n'.join(s)
