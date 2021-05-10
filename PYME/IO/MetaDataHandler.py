#!/usr/bin/python

##################
# MetaDataHandler.py
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

#!/usr/bin/python
"""
Defines metadata handlers for the saving of acquisiton metadata to a variety 
of file formats, as well as keeping track of metadata sources. 

Metadata sources
----------------

Metadata sources are simply functions, which when called, write information into
a provided handler. e.g.::
    def metadataGenerator(mdhandler):
        mdhandler['a.key'] = value

These generator functions are registered by adding them to one of two lists exposed 
by this module: **provideStartMetadata** or **provideStopMetadata**. depending on
whether it makes more sense to record the metadata at the start or stop of an 
acquisition.

A good example can be found in PYME.Acquire.Hardware.Camera.AndorIXon.AndorIXon.

MetaData Handlers
-----------------

**NestedClassMDHandler**
    An in-memory metadatahandler used to buffer metadata or to store values prior
    to the file format being known.
**HDFMDHandler**
    For local pytables/hdf5 datasets
**QueueMDHandler**
    For use with data hosted in a taskqueue
**XMLMDHandler**
    For use with PYMEs XML metadata format - typically used with .tiff files or
    other data for which it is difficult to embed metadata.
**SimpleMDHandler**
    Saves and reads metadata as a python script (a series of md[key]=value statements).
    Used where you might want to construct or modify metadata by hand - e.g. with
    foreign source data.
    

The format of a metadata handler is defined by the `MDHandlerBase` class. 


"""
try:
    # noinspection PyCompatibility
    from UserDict import DictMixin
except ImportError:
    #py3
    from collections import MutableMapping as DictMixin
    
import six
from collections import namedtuple

import logging
logger = logging.getLogger(__name__)

#lists where bits of hardware can register the fact that they are capable of 
#providing metadata, by appending a function with the signature:
#genMetadata(MetaDataHandler)
provideStartMetadata = []
provideStopMetadata = []

# Define a voxelsize class
# NB the default units across PYME (everywhere but the metadata) are nm, with metadata being the exception
# in that it defaults to um
# this class exists to make things a bit simpler when accessing the metadata
VoxelSize = namedtuple('VoxelSize', 'x,y,z')
VoxelSize.units='nm'

def instanceinlist(cls, list):
    for c in list:
        if isinstance(cls, c):
            return True

    return False
    
    
def get_camera_roi_origin(mdh):
    """
    helper function to allow us to transition to 0 based ROIs.
    
    Returns the first of these it finds:
    - [Camera.ROIOriginX, Camera.ROIOriginY]
    - [Camera.ROIPosX -1, Camera.ROIPosY-1]
    - [0,0]
    
    NOTE: this is not yet widely supported in calling code (ie you should still write ROIPosX, ROIPosY, although it is
    safe to write both the old and new versions.
    
    Parameters
    ----------
    mdh : metadata handler

    Returns
    -------
    
    ROIOriginX, ROIOriginY

    """
    if 'Camera.ROIOriginX' in mdh.getEntryNames():
        return mdh['Camera.ROIOriginX'], mdh['Camera.ROIOriginY']
    elif 'Camera.ROIPosX' in mdh.getEntryNames():
        return mdh['Camera.ROIPosX']-1, mdh['Camera.ROIPosY']-1
    else:
        return 0,0
    
def get_camera_physical_roi_origin(mdh):
    """
    Get the camera roi offset to use for positioning. For a standard camera this is the same as the ROI pixel origin,
    but for cameras with an image splitting device this may not be the case. For these cameras, return a position
    relative to the left-most ROI.
    
    TODO: This currently works for the HTSMS multiview setup, expand to also work with the dual-view splitters used on
    other setups.
    
    Parameters
    ----------
    mdh

    Returns
    -------

    """
    if len(mdh.get('Multiview.ActiveViews', [])) > 0:
        # multiview cameras are on, assume the origin of the 0th channel is the stage-origin of all views
        return mdh['Multiview.ROI0Origin']
    else:
        return get_camera_roi_origin(mdh)

def origin_nm(mdh, default_pixel_size=1.):
    """ origin, in nm of the **image** ROI from the camera upper left hand pixel - used for 
    overlaying with different ROIs.

    Parameters
    ----------
    mdh : PYME.IO.MetaDataHandler
    default_pixel_size : float, optional
        Safe to ignore. Parameter only exists for a niche case in ImageStack, when no
        pixel size is defined. Should probably change to emit a warning if we actually use this fallback.

    Returns
    -------
    tuple
        x, y, z origin in nanometers
    
    Notes
    -----
    
    **Use with localisation / point data:** When used with localization data,
    `origin_nm()` returns the origin of the pixel data in the **raw** image series used 
    to derive the localisations. Whilst x and y localisations are referenced to the ROI
    (and hence share an origin with the pixel data) z localisations are absolute
    (technically referenced to the 0 position of the z-piezo). As a result, the z-component
    of `origin_nm()` should be ignored when used with localisation data, which does
    not require z origin correction.

    transferred from image.ImageStack.origin so that it can be used for tabular data too.
    """

    if 'Origin.x' in mdh.getEntryNames():
        # Used in composite images, cropped images, and renderings. Takes precendence if defined.
        # Example in PYME.LMVis.renderers.ColourRenderer
        return mdh['Origin.x'], mdh['Origin.y'], mdh['Origin.z']

    elif ('Camera.ROIPosX' in mdh.getEntryNames()) or ('Camera.ROIOriginX' in mdh.getEntryNames()):
        #has ROI information
        try:
            voxx, voxy, _ = mdh.voxelsize_nm
        except AttributeError:
            voxx = default_pixel_size
            voxy = voxx
    
        roi_x0, roi_y0 = get_camera_roi_origin(mdh)
    
        ox = (roi_x0) * voxx
        oy = (roi_y0) * voxy
    
        oz = 0
    
        if 'AcquisitionType' in mdh.getEntryNames() and mdh['AcquisitionType'] == 'Stack':
            oz = mdh['StackSettings.StartPos'] * 1e3
        elif 'Positioning.z' in mdh.getEntryNames():
            oz = mdh['Positioning.z'] * 1e3
        elif 'Positioning.PIFoc' in mdh.getEntryNames():
            oz = mdh['Positioning.PIFoc'] * 1e3
    
        return ox, oy, oz

    elif 'Source.Camera.ROIPosX' in mdh.getEntryNames():
        # TODO - can we somehow defer this and next case to get_camera_roi_corigin()
        #a rendered image with information about the source ROI
        voxx, voxy = 1e3 * mdh['Source.voxelsize.x'], 1e3 * mdh['Source.voxelsize.y']
    
        ox = (mdh['Source.Camera.ROIPosX'] - 1) * voxx
        oy = (mdh['Source.Camera.ROIPosY'] - 1) * voxy
    
        return ox, oy, 0
    elif 'Source.Camera.ROIOriginX' in mdh.getEntryNames():
        #a rendered image with information about the source ROI
        voxx, voxy = 1e3 * mdh['Source.voxelsize.x'], 1e3 * mdh['Source.voxelsize.y']
    
        ox = (mdh['Source.Camera.ROIOriginX']) * voxx
        oy = (mdh['Source.Camera.ROIOriginY']) * voxy
    
        return ox, oy, 0
    else:
        return 0, 0, 0
    
def localisation_origin_nm(mdh):
    ''' Get the origin of localisation data. 
    
    Effectively a shortcut for `origin_nm()`, but discarding the z-component as whilst
    x and y co-ordinates in localisation data are referenced to the ROI, the z component
    is absolute.
    '''
    ox, oy, _ = origin_nm(mdh)
    return ox, oy, 0
    
def get_voxelsize_nm(mdh):
    '''
    Helper function to obtain the voxel size, in nm, from the metadata (to replace the many 1e3*mdh['voxelsize.x'] calls)
    
    NOTE: supplies a default z voxelsize of 0 if none in metadata.
    
    Parameters
    ----------
    mdh

    Returns
    -------

    '''
    
    return VoxelSize(1e3*mdh['voxelsize.x'], 1e3*mdh['voxelsize.y'], 1e3*mdh.get('voxelsize.z', 0))


# compatibility stubs to permit attribute based access while we transition away from NestedClassMDHandler
class _AttrProxy(object):
    def __init__(self, mdh, parent=None):
        self._mdh = mdh
        self._parent = parent
    
    def __getattr__(self, item):
        if self._parent is not None:
            item = '.'.join([self._parent, item])
        
        return _attr_access(self._mdh, item)


def _attr_access(mdh, key):
    try:
        return mdh[key]
    except KeyError:
        if any([k.startswith(key) for k in mdh.keys()]):
            return _AttrProxy(mdh, parent=key)
        else:
            raise AttributeError('Attribute %s not found' % key)


class MDHandlerBase(DictMixin):
    """Base class from which all metadata handlers are derived.

    Metadata attributes can be read and set using either a dictionary like
    interface, or by calling the `getEntry` and `setEntry` methods. 
    
    .. note:: Derived classes **MUST** override `getEntry`, `setEntry`, and `getEntryNames`.
    """
    #base class to make metadata behave like a dictionary
    def getEntry(self, name):
        """Returns the entry for a given name.
        
        Parameters
        ----------
        name : string
            The entry name. This name should be heirachical, and deliminated
            with dots e.g. 'Camera.EMCCDGain'
            
        Returns
        -------
        value : object
            The value stored for the given key. This can, in principle, be 
            anything that can be pickled. strings, ints, bools and floats are
            all stored in a human readable form in the textual metadata 
            representations, wheras more complex objects are base64 encoded.
        """
        raise NotImplementedError('getEntry must be overridden in derived classes')
        
    def setEntry(self, name, value):
        """Sets the entry for a given name.
        
        Parameters
        ----------
        name : string
            The entry name. This name should be heirachical, and deliminated
            with dots e.g. 'Camera.EMCCDGain'
            
        value : object
            The value stored for the given key. This can, in principle, be 
            anything that can be pickled. strings, ints, bools and floats are
            all stored in a human readable form in the textual metadata 
            representations, wheras more complex objects are base64 encoded.
        """
        raise NotImplementedError('setEntry must be overridden in derived classes')
        
    def getEntryNames(self):
        """Returns a list of defined entries.
            
        Returns
        -------
        names : list of string
            The keys which are defined in the metadata.
        """
        raise NotImplementedError('getEntryNames must be overridden in derived classes')
        
    def __setitem__(self, name, value):
        self.setEntry(name, value)

    def __getitem__(self, name):
        try:
            return self.getEntry(name)
        except AttributeError:
            raise KeyError('Key %s not defined' % name)

    if six.PY3:
        def __len__(self):
            return len(self.getEntryNames())
    
        def __iter__(self):
            for k in self.getEntryNames():
                yield k
                
        def __delitem__(self, key):
            raise RuntimeError('Cannot delete metadata item')
        
    def getOrDefault(self, name, default):
        """Returns the entry for a given name, of a default value if the key
        is not present.
        
        Parameters
        ----------
        name : string
            The entry name. This name should be heirachical, and deliminated
            with dots e.g. 'Camera.EMCCDGain'
        default : object
            What to return if the name is not defined
            
        Returns
        -------
        value : object
            The value stored for the given key. This can, in principle, be 
            anything that can be pickled. strings, ints, bools and floats are
            all stored in a human readable form in the textual metadata 
            representations, wheras more complex objects are base64 encoded.
        """
        try: 
            return self.getEntry(name)
        except (KeyError, AttributeError):
            return default

    def keys(self):
        """Alias for getEntryNames to make us look like a dictionary"""
        return self.getEntryNames()

    def copyEntriesFrom(self, mdToCopy):
        """Copies entries from another metadata object into this one. Duplicate
        keys will be overwritten.
        
        Parameters
        ----------
        mdToCopy : an instance of a metadata handler
            The metadata handler from which to copy entries.
        """
        for en in mdToCopy.keys():
            # 9/10/20 DB - change to dictionary access notation so we can also pass a dictionary
            self.setEntry(en, mdToCopy.get(en))
        #self.update(mdToCopy)

    def mergeEntriesFrom(self, mdToCopy):
        """Copies entries from another metadata object into this one. Values
        are only copied if they are not already defined locally.
        
        Parameters
        ----------
        mdToCopy : an instance of a metadata handler
            The metadata handler from which to copy entries.
        """
        #only copies values if not already defined
        for en in mdToCopy.getEntryNames():
            if not en in self.getEntryNames():
                self.setEntry(en, mdToCopy.getEntry(en))

    def __repr__(self):
        import re
        s = ['%s: %s' % (en, self.getEntry(en) if not re.search(r'Time$',en) else self.tformat(self.getEntry(en))) for en in self.getEntryNames()]
        return '<%s>:\n\n' % self.__class__.__name__ + '\n'.join(s)

    @staticmethod
    def tformat(timeval):
        import time
        
        print(timeval)
        
        if isinstance(timeval, six.string_types):
            if timeval == '':
                return timeval
                
            timeval = float(timeval)
        
        if timeval < 946684800: # timestamp for year 2000 as heuristic
            return timeval
        else:
            return "%s (%s)" % (timeval,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timeval)))

    def GetSimpleString(self):
        """Writes out metadata in simplfied format.
        
        Returns
        -------
            mdstring : string
                The metadata in a simple, human readable format.
                
        See Also
        --------
        SimpleMDHandler
        """
        
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
            
        import numpy as np
        import six
        
        s = ['#PYME Simple Metadata v1\n']

        for en in self.getEntryNames():
            val = self.getEntry(en)

            if isinstance(val, six.string_types) or np.isscalar(val): #quote string
                val = repr(val)
            elif not isinstance(val, (int, float, list, dict, tuple)): #not easily recovered from representation
                val = "pickle.loads('''%s''')" % pickle.dumps(val).replace('\n', '\\n')

            s.append("md['%s'] = %s\n" % (en, val))
        
        return s
    
    @property
    def voxelsize_nm(self):
        return get_voxelsize_nm(self)
        
    def WriteSimple(self, filename):
        """Dumps metadata to file in simplfied format.
        
        Parameters
        ----------
            filename : string
                The the filename to write to. Should end in .md.
                
        See Also
        --------
        SimpleMDHandler
        """
        s = self.GetSimpleString()
        f = open(filename, 'w')
        f.writelines(s)
        f.close()
        
    def to_JSON(self):
        import json
        import numpy as np
        
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.generic):
                    return obj.tolist()
                try:
                    return obj.to_JSON()
                except AttributeError:
                    return json.JSONEncoder.default(self, obj)

        d = {k: self.getEntry(k) for k in self.getEntryNames()}
        
        return json.dumps(d, indent=2, sort_keys=True, cls=CustomEncoder)
    
    def __getattr__(self, key):
        """ Compatibility stub to support transition away from NestedClassMDHandler"""
        import warnings
        warnings.warn('Metadata access should use dictionary based or getEntry() syntax, not attribute based access', DeprecationWarning)
        _attr_access(self, key)

class HDFMDHandler(MDHandlerBase):
    def __init__(self, h5file, mdToCopy=None):
        self.h5file = h5file
        self.md = None

        if self.h5file.__contains__('/MetaData'):
            self.md = self.h5file.root.MetaData
        else:
            self.md = self.h5file.create_group(self.h5file.root, 'MetaData')

        if not mdToCopy is None:
            self.copyEntriesFrom(mdToCopy)


    def setEntry(self,entryName, value):
        entPath = entryName.split('.')
        en = entPath[-1]
        ep = entPath[:-1]

        currGroup = self.h5file._get_or_create_path('/'.join(['', 'MetaData']+ ep), True)
        currGroup._f_setattr(en, value)
        self.h5file.flush()


    def getEntry(self,entryName):
        entPath = entryName.split('.')
        en = entPath[-1]
        ep = entPath[:-1]

        res =  self.h5file.get_node_attr('/'.join(['', 'MetaData']+ ep), en)
        
        if isinstance(res, bytes):
            res = res.decode('ascii')
        
        #dodgy hack to get around a problem with zero length strings not
        #being picklable if they are numpy (rather than pure python) types
        #this code should convert a numpy empty string into a python empty string
        if res == '':
            return ''
        
        return res
        


    def getEntryNames(self):
        entryNames = []
        for a in [self.md] + list(self.md._f_walknodes()):
            entryNames.extend(['.'.join(a._v_pathname.split('/')[2:] +[ i]) for i in a._v_attrs._f_list()])

        return entryNames

class QueueMDHandler(MDHandlerBase):
    def __init__(self, tq, queueName, mdToCopy=None):
        self.tq = tq
        self.queueName = queueName
        self.md = None

        if not mdToCopy is None:
            self.copyEntriesFrom(mdToCopy)
            
    def copyEntriesFrom(self, mdToCopy):
        self.tq.setQueueMetaDataEntries(self.queueName, mdToCopy)

    def setEntry(self,entryName, value):
        self.tq.setQueueMetaData(self.queueName, entryName, value)


    def getEntry(self,entryName):
        #print entryName
        return self.tq.getQueueMetaData(self.queueName, entryName)


    def getEntryNames(self):
        return self.tq.getQueueMetaDataKeys(self.queueName)
        


class NestedClassMDHandler(MDHandlerBase):
    def __init__(self, mdToCopy=None):
        if not mdToCopy is None:
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
        #print(entryName)
        return eval('self.'+entryName)
#        try:
#            return eval('self.'+entryName)
#        except AttributeError:
#            raise KeyError('No entry found for %s' % entryName)


    def getEntryNames(self):
        en = []
        for k in self.__dict__.keys():
            if hasattr(self.__dict__[k], 'getEntryNames') and not self.__dict__[k].__module__ == 'Pyro.core':
                en += [k + '.' + kp for kp in self.__dict__[k].getEntryNames()]
            else:
                en.append(k)

        return en
    
    def __getattr__(self, item):
        '''Replace __getattr__ of base class which spoofs nested class like attribute access with a move vanilla implementation'''
        try:
            return self.__dict__[item]
        except KeyError:
            raise AttributeError('NestedClassMDHandler has no attribute "%s"' % item)

class DictMDHandler(MDHandlerBase):
    """Simple implementation using a dict.
    
    Should eventually replace most instances of NestedClassMDHandler
    
    Adds a writable flag to enable read-only metadata (currently unused, but gives us the option of enforcing good
    behaviour in, e.g., recipe modules, in the future). TODO - propagate writable flag up to base?
    """
    def __init__(self, mdToCopy=None, writable=True):
        self._storage = {}
        self._writable = True
        if not mdToCopy is None:
            self.copyEntriesFrom(mdToCopy)
        
        # set our writable after we've done the initial copy
        self._writable = writable
        
        import warnings
        warnings.warn('DictMDHandler is not yet fully supported, and will likely cause failures for anything related to localisation fitting')
    
    def setEntry(self, entryName, value):
        if not self._writable:
            raise RuntimeError('metadata handler is read-only')
        
        self._storage[entryName] = value
    
    def getEntry(self, entryName):
        return self._storage[entryName]
    
    def getEntryNames(self):
        return list(self._storage.keys())
    
    def set_readonly(self):
        self._writable = False
        
class CachingMDHandler(MDHandlerBase):
    def __init__(self, mdToCache):
        self.mdToCache = mdToCache
        
        if not mdToCache is None:
            self.cache = dict(mdToCache.items())
            
    @classmethod
    def recreate(cls, cache):
        c = cls(None)
        c.cache = cache
        
    def __reduce__(self):
        return (CachingMDHandler.recreate, (self.cache,))
        
    def getEntry(self, entryName):
        return self.cache[entryName]
        
    def setEntry(self, entryName, value):
        self.cache[entryName] = value
        if not self.mdToCache is None:
            self.mdToCache.setEntry(entryName, value)
        
    def getEntryNames(self):
        return self.cache.keys()
    
    
class CopyOnWriteMDHandler(MDHandlerBase):
    def __init__(self, orig_md):
        self._orig_md = orig_md
        
        self._cache = dict()
    
    def getEntry(self, entryName):
        try:
            return self._cache[entryName]
        except KeyError:
            return self._orig_md.getEntry(entryName)
    
    def setEntry(self, entryName, value):
        self._cache[entryName] = value
    
    def getEntryNames(self):
        return sorted(list(set(self._cache.keys()).union(self._orig_md.keys())))
    

from xml.dom.minidom import getDOMImplementation, parse, parseString
#from xml.sax.saxutils import escape, unescape
import base64

class SimpleMDHandler(NestedClassMDHandler):
    """simple metadata format - consists of a python script with a .md extension
    which adds entrys using the dictionary syntax to a metadata handler called md"""

    def __init__(self, filename = None, mdToCopy=None):
        if not filename is None:
            from PYME.util.execfile import _execfile
            try:
                import cPickle as pickle
            except ImportError:
                import pickle
                
            #loading an existing file
            md = self
            fn = __file__
            globals()['__file__'] = filename
            _execfile(filename, locals(), globals())
            globals()['__file__'] = fn

        if not mdToCopy is None:
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
        if not filename is None:
            #loading an existing file
            self.doc = parse(filename)
            self.md = self.doc.documentElement.getElementsByTagName('MetaData')[0]
        else:
            #creating a new document
            self.doc = getDOMImplementation().createDocument(None, 'PYMEImageData', None)
            self.md = self.doc.createElement('MetaData')
            self.doc.documentElement.appendChild(self.md)

        if not mdToCopy is None:
            self.copyEntriesFrom(mdToCopy)

    def writeXML(self, filename):
        f = open(filename, 'w')
        f.write(self.doc.toprettyxml())
        f.close()


    def setEntry(self,entryName, value):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        
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
            node.setAttribute('value', str(value))#.encode('utf-8'))
        elif isinstance(value, int):
            node.setAttribute('class', 'int')
            node.setAttribute('value', str(value))#.encode('utf-8'))
        elif isinstance(value, six.binary_type):
            node.setAttribute('class', 'str')
            node.setAttribute('value', value)
        elif isinstance(value, six.text_type):
            node.setAttribute('class', 'unicode')
            node.setAttribute('value', str(value))#.encode('utf-8'))
        elif np.isscalar(value):
            node.setAttribute('class', 'float')
            node.setAttribute('value', str(value))#.encode('utf-8'))
        else: #pickle more complicated structures
            node.setAttribute('class', 'pickle')
            #print((value, pickle.dumps(value)))
            node.setAttribute('value', base64.b64encode((pickle.dumps(value))).decode('ascii'))


    def getEntry(self,entryName):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
            
        entPath = entryName.split('.')

        node = self.md
        while len(entPath) >= 1:
            el = [e for e in node.childNodes if e.nodeName == entPath[0]]
            if len(el) == 0:
                #node not there
                raise AttributeError(u'Requested node not found')
            else:
                node = el[0]

            entPath.pop(0)

        cls = node.getAttribute('class')
        val = node.getAttribute('value')
        
        if val == 'True': #booleans get cls 'int'
                val = True
        elif val == 'False':
                val = False
        elif cls == 'int':
                val = int(val)
        elif cls == 'float':
            val = float(val)
        elif cls == 'unicode':
            if six.PY2:
                val = val.decode('utf8')
        elif cls == 'pickle':
            #return None
            try:
                val = pickle.loads(base64.b64decode(val.encode('ascii')))
            except:
                logger.exception(u'Error loading metadata from pickle')

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


class OMEXMLMDHandler(XMLMDHandler):
    _OME_UNITS_TO_UM = {'m': 1e6, 'mm': 1e3, 'um': 1.0, u'\u00B5m': 1.0, 'nm': 1e-3}
    
    @classmethod
    def _get_pixel_size_um(cls, pix, axis, default=0.1):
        axis = axis.upper()
        try:
            ps = float(pix.getAttribute('PhysicalSize%s' % axis))
        except:
            logger.error('No %s pixel size defined, using default' % axis)
            return default
        try:
            ps = ps * cls._OME_UNITS_TO_UM[pix.getAttribute('PhysicalSize%sUnit' % axis)]
        except:
            logger.error('No units defined for axis %s, defaulting to um' % axis)
            return ps
            
        return ps
        
    def __init__(self, XMLData = None, mdToCopy=None):
        if not XMLData is None:
            #loading an existing file
            self.doc = parseString(XMLData)
            #try:
            try:
                self.md = self.doc.documentElement.getElementsByTagName('MetaData')[0]
            except IndexError:
                self.md = self.doc.createElement('MetaData')
                self.doc.documentElement.appendChild(self.md)
                
                #try to load pixel size etc fro OME metadata
                pix = self.doc.getElementsByTagName('Pixels')[0]

                #using -ve defaults will trigger a voxelsize prompt in the GUI if pixel size metadata is not present
                self['voxelsize.x'] = self._get_pixel_size_um(pix, 'X', -.1)
                self['voxelsize.y'] = self._get_pixel_size_um(pix, 'Y', -.1)
                self['voxelsize.z'] = self._get_pixel_size_um(pix, 'Z', 0.0)
                self['voxelsize.units'] = 'um' #this is a courtesy - to define anything else is an error.
                    
                try:
                    self['Camera.CycleTime'] = float(pix.getAttribute('TimeIncrement'))
                except:
                    pass
                
                self['OME.SizeX'] = int(pix.getAttribute('SizeX'))
                self['OME.SizeY'] = int(pix.getAttribute('SizeY'))
                self['OME.SizeZ'] = int(pix.getAttribute('SizeZ'))
                self['OME.SizeT'] = int(pix.getAttribute('SizeT'))
                self['OME.SizeC'] = int(pix.getAttribute('SizeC'))
                
                self['OME.DimensionOrder'] = pix.getAttribute('DimensionOrder')
                    
                #except:
                #    pass
            
            
                
            
        else:
            #creating a new document
            self.doc = getDOMImplementation().createDocument(None, 'OME', None)
            self.doc.documentElement.setAttribute('xmlns', "http://www.openmicroscopy.org/Schemas/OME/2015-01")
            #self.doc.documentElement.setAttribute('xmlns:ROI', "http://www.openmicroscopy.org/Schemas/ROI/2015-01")
            #self.doc.documentElement.setAttribute('xmlns:BIN', "http://www.openmicroscopy.org/Schemas/BinaryFile/2015-01")
            self.doc.documentElement.setAttribute('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
            self.doc.documentElement.setAttribute('xsi:schemaLocation','http://www.openmicroscopy.org/Schemas/OME/2015-01 http://www.openmicroscopy.org/Schemas/OME/2015-01/ome.xsd')
            
            
            self.img = self.doc.createElement('Image')
            self.img.setAttribute('ID', 'Image:0')
            self.img.setAttribute('Name', 'Image:0')
            self.doc.documentElement.appendChild(self.img)
            
            self.pixels = self.doc.createElement('Pixels')
            self.img.appendChild(self.pixels)
            self.pixels.setAttribute('ID', 'Pixels:0')
            self.pixels.setAttribute('DimensionOrder', 'XYZTC')
            self.pixels.setAttribute('BigEndian', 'false')
            self.pixels.setAttribute('Interleaved', 'false')
            
            tf = self.doc.createElement('TiffData')
            self.pixels.appendChild(tf)
            
            sa = self.doc.createElement('StructuredAnnotations')
            self.doc.documentElement.appendChild(sa)
            
            xa = self.doc.createElement('XMLAnnotation')
            sa.appendChild(xa)
            xa.setAttribute('ID', 'PYME')
            #self.doc = getDOMImplementation().createDocument(None, 'PYMEImageData', None)
            v = self.doc.createElement('Value')
            xa.appendChild(v)
            
            self.md = self.doc.createElement('MetaData')
            v.appendChild(self.md)

        if not mdToCopy is None:
            self.copyEntriesFrom(mdToCopy)
            
    def getXML(self, data = None):
        #sync the OME data from the ordinary metadata
        if not data is None:
            ims = data.shape
            if (data.ndim == 5):
                SizeY, SizeX, SizeZ, SizeT, SizeC = ims[:data.ndim]
            elif(data.ndim == 4):
                SizeY, SizeX, SizeT, SizeC = ims[:data.ndim]
                SizeZ = 1
            else:
                SizeY, SizeX, SizeT = ims[:3]
                SizeZ = 1
                SizeC = 1
                
            
            if str(data[0,0,0,0,0].dtype) in ('float32', 'float64'):
                self.pixels.setAttribute('Type', 'float')
            else:
                self.pixels.setAttribute('Type', str(data.dtype))
            self.pixels.setAttribute('SizeX', str(SizeX))
            self.pixels.setAttribute('SizeY', str(SizeY))
            self.pixels.setAttribute('SizeZ', str(SizeZ))
            self.pixels.setAttribute('SizeT', str(SizeT))
            self.pixels.setAttribute('SizeC', str(SizeC))
            
            for i in range(SizeC):
                c = self.doc.createElement('Channel')
                c.setAttribute('ID', 'Channel:0:%d' %i)
                c.setAttribute('SamplesPerPixel', '1')
                l = self.doc.createElement('LightPath')
                c.appendChild(l)
                self.pixels.appendChild(c)
            
            if 'voxelsize.x' in self.getEntryNames():
                self.pixels.setAttribute('PhysicalSizeX', '%3.4f' % self.getEntry('voxelsize.x'))
                self.pixels.setAttribute('PhysicalSizeY', '%3.4f' % self.getEntry('voxelsize.y'))
    
        return self.doc.toprettyxml()
    
    def writeXML(self, filename):
        f = open(filename, 'w')
        f.write(self.getXML())
        f.close()

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


def from_json(json_string):
    import json
    mdh = NestedClassMDHandler()
    mdh.update(json.loads(json_string))
    
    return mdh
    
def load_json(filename):
    import json
    from PYME.IO import unifiedIO
    mdh = NestedClassMDHandler()
    mdh.update(json.loads(unifiedIO.read(filename)))
    
    return mdh
