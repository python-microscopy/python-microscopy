from BaseHTTPServer import BaseHTTPRequestHandler
import urlparse
import os
from PYME.FileUtils import nameUtils
from PYME.Acquire import MetaDataHandler
from PYME.ParallelTasks import HDFTaskQueue
import time
import cPickle as pickle

from PYME.io import PZFFormat

import tables
import json

fileCache = {}
accessTimes = {}
#will contain entries of from 

class H5File(object):
    def __init__(self, pth, mode='r'):
        if mode in ['w', 'a', 'r+'] and os.path.exists(pth):
            raise RuntimeError('Cannot open existing file in write mode')
        self.h5f = tables.openFile(pth, mode)
        self.mode = mode

        self.complevel = 6
        self.complib = 'zlib'
        
        self.usePZFFormat = True
        self.PZFCompression = 'huffman_chunks'

        self.mdh = MetaDataHandler.CachingMDHandler(MetaDataHandler.HDFMDHandler(self.h5f))

        if 'ImageData' in dir(self.h5f.root):
            self.dshape = [self.h5f.root.ImageData.shape[1], self.h5f.root.ImageData.shape[2], self.h5f.root.ImageData.shape[0]]
            self.usePZFFormat = False
        elif 'PZFImageData' in dir(self.h5f.root):
            self.dshape = [0,0,self.hf5.root.PZFImageData.shape[0]]
            self.dshape[:2] = self.hf5.root.PZFImageData.framesize
            self.usePZFFormat = True
        else:
            self.dshape = [0,0,0]
        
        if 'Events' in dir(self.h5f.root):
            self.nEvents = self.h5f.root.Events.shape[0]
        else:
            self.nEvents = 0
            
            if mode == 'w':
                self._checkCreateEventsTable()

    def getFrame(self, frameNo):
        if frameNo >= self.dshape[2]:
            raise IndexError('Index out of bounds')
        if not self.usePZFFormat:
            return self.h5f.root.ImageData[frameNo, :,:].dumps()
        else:
            f, h = PZFFormat.loads(self.h5f.root.PZFImageData[frameNo])
            return f.dumps()# f.reshape((1,) + f.shape[:2]).dumps()
            
    def getPZFFrame(self, frameNo):
        if frameNo >= self.dshape[2]:
            raise IndexError('Index out of bounds')
        if not self.usePZFFormat:
            return PZFFormat.dumps(self.h5f.root.ImageData[frameNo, :,:].squeeze(), compression = self.PZFCompression)
        else:
            return self.h5f.root.PZFImageData[frameNo]

    def getEvent(self, eventNo):
    	if eventNo >= self.nEvents:
    		raise IndexError('Index out of bounds')
    	return self.h5f.root.Events[eventNo].dumps()

    def getEvents(self):
        if self.nEvents > 0:
            return self.h5f.root.Events[:].dumps()
        else:
            return pickle.dumps([], 2)

    def getMetadata(self):
    	return pickle.dumps(MetaDataHandler.NestedClassMDHandler(self.mdh), 2)

    def _checkCreateDataTable(self, f):
        if (not 'ImageData' in dir(self.h5f.root)) and (not 'PZFImageData' in dir(self.h5f.root)):
            if isinstance(f, str):
                #is a PZF file
                f = PZFFormat.loads(f)[0]
                f.reshape((1,) + f.shape[:2])

            framesize = f.shape[1:3]
            self.dshape[:2] = framesize            
            
            if not self.usePZFFormat:                
                filt = tables.Filters(self.complevel, self.complib, shuffle=True)
                self.imageData = self.h5f.createEArray(self.h5f.root, 'ImageData', tables.UInt16Atom(), (0,)+tuple(framesize), filters=filt, chunkshape=(1,)+tuple(framesize))                
            else:
                self.compImageData = self.h5f.createVLArray(self.h5f.root, 'PZFImageData', tables.VLStringAtom())
                self.compImageData.attrs.framesize = framesize            


    def _checkCreateEventsTable(self):
    	if not 'Events' in dir(self.h5f.root):
         filt = tables.Filters(self.complevel, self.complib, shuffle=True)
         self.events = self.h5f.createTable(self.h5f.root, 'Events', HDFTaskQueue.SpoolEvent,filters=filt)
            
    
    def putFrame(self, frame):
        f = pickle.loads(frame)
        self._checkCreateDataTable(f)
        
        if self.usePZFFormat:
            self.compImageData.append(PZFFormat.dumps(f.squeeze(), compression = self.PZFCompression))
            self.compImageData.flush()
        else:
            self.imageData.append(f)
            self.imageData.flush()
        self.dshape[2] += 1
        
    def putPZFFrame(self, frame):
        self._checkCreateDataTable(frame)
        
        if self.usePZFFormat:
            self.compImageData.append(frame)
            self.compImageData.flush()
        else:
            f, h = PZFFormat.loads(frame)
            self.imageData.append(f.reshape((1,) + f.shape[:2]))
            self.imageData.flush()
        self.dshape[2] += 1

    def putFrames(self, frames):
        t1 = time.time()
        fs = pickle.loads(frames)
        t2 = time.time()
        self._checkCreateDataTable(fs[0])
        
        if self.usePZFFormat:        
            for f in fs:
                self.compImageData.append(PZFFormat.dumps(f.squeeze(), compression = self.PZFCompression))
                self.dshape[2] += 1
            
            self.compImageData.flush()
        else:
            for f in fs:
                self.imageData.append(f)
                self.dshape[2] += 1
        
            self.imageData.flush()
        
        print (time.time() - t1)/float(len(fs)), (t2-t1)/float(len(fs))
        
    def putPZFFrames(self, frames):
        t1 = time.time()
        fs = pickle.loads(frames)
        t2 = time.time()
        self._checkCreateDataTable(fs[0])
        
        if self.usePZFFormat:        
            for f in fs:
                self.compImageData.append(f)
                self.dshape[2] += 1
            
            self.compImageData.flush()
        else:
            for f in fs:
                f, h = PZFFormat.loads(f)
                self.imageData.append(f.reshape([0,] + f.shape[:2]))
                self.dshape[2] += 1
        
            self.imageData.flush()
        
        print (time.time() - t1)/float(len(fs)), (t2-t1)/float(len(fs))
        
    def putEvent(self, event):
        #self._checkCreateEventsTable()
        eventName, eventDescr, evtTime = pickle.loads(event)
        ev = self.events.row
        
        ev['EventName'] = eventName
        ev['EventDescr'] = eventDescr
        ev['Time'] = evtTime
        
        ev.append()
        self.events.flush()
        self.nEvents += 1


    def putMetadata(self, metadata):
    	md = pickle.loads(metadata)
    	self.mdh.copyEntriesFrom(md)

    def putMetadataEntry(self, msg):
        key, value = pickle.loads(msg)
        self.mdh[key] = value

class GetHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"   
    #protocol_version = "HTTP/1.1"   
    def _getH5File(self, pth, mode='r'):
        try:
            h5f = fileCache[pth]
            accessTimes[pth] = time.time()
            return h5f
        except KeyError:
            #we don't yet have the file open:
            h5f = H5File(pth, mode)

            fileCache[pth] = h5f
            accessTimes[pth] = time.time()
            return h5f

    def do_GET(self):
        parsed_path = urlparse.urlparse(self.path)
        fullpath = os.path.join(nameUtils.datadir, parsed_path.path[1:])
        
        pth = fullpath
        ptail = []
        
        while not os.path.exists(pth):
            pth, pt = os.path.split(pth)
            ptail.insert(0, pt)
        

        if pth.endswith('.h5'): #PYME HDF
            h5f = self._getH5File(pth)
            
            if ptail == []:
                self.send_response(200)
                self.end_headers()
                
                message = '''PYME h5 data:
                Data size: [%d,%d, %d]
                NumEvents: %d
                ''' % tuple(h5f.dshape + [h5f.nEvents])
                
                self.wfile.write(message)
                return
            
            elif len(ptail) == 2 and ptail[0] == "DATA":
                frameNo = int(ptail[1])
                try:
                    message = h5f.getFrame(frameNo)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(message)
                    return
                except IndexError:
                    self.send_error(404, 'Index out of bounds')
                    return
            
            elif ptail[0] == "SHAPE":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(pickle.dumps(h5f.dshape, 2))
                return
            elif ptail[0] == "EVENTS":
                if len(ptail) == 2:
                    eventNo = int(ptail[1])
                    try:
                        message = h5f.getEvent(eventNo)
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(message)
                        return
                    except IndexError:
                        self.send_error(404, 'Index out of bounds')
                        return
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(h5f.getEvents())
                    return 
                    
            elif ptail[0] == "METADATA":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(h5f.getMetadata())
                return 
        
        
        else:
            message_parts = [
                'CLIENT VALUES:',
                'client_address=%s (%s)' % (self.client_address,
                    self.address_string()),
                'command=%s' % self.command,
                'path=%s' % self.path,
                'real path=%s' % parsed_path.path,
                'query=%s' % parsed_path.query,
                'fragment=%s' % parsed_path.fragment,
                'request_version=%s' % self.request_version,
                '',
                'SERVER VALUES:',
                'server_version=%s' % self.server_version,
                'sys_version=%s' % self.sys_version,
                'protocol_version=%s' % self.protocol_version,
                '',
                'PYMEDATADIR=%s' % nameUtils.datadir,
                'full_path=%s' %fullpath,
                #'File_type=%s' % fileType,
                'short_path=%s' % pth,
                'path_tail=%s' % ptail,
                '',
                'HEADERS RECEIVED:',
                ]
            for name, value in sorted(self.headers.items()):
                message_parts.append('%s=%s' % (name, value.rstrip()))
            message_parts.append('')
            message = '\r\n'.join(message_parts)
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(message)
            return

    def do_POST(self):
        parsed_path = urlparse.urlparse(self.path)
        fullpath = os.path.join(nameUtils.datadir, parsed_path.path[1:])
        # Begin the response
        filepth, entry = os.path.split(fullpath)
        
        #print filepth, entry
        
        #print self.headers.items()
        #print self.headers['content-length']
        
        message = self.rfile.read(int(self.headers['content-length']))
        #print len(message)
        
        if filepth.endswith('.h5'):
            h5f = self._getH5File(filepth, 'w')
            
            if entry == 'NEWFRAME':
                h5f.putFrame(message)
            elif entry == 'NEWPZFFRAME':
                h5f.putPZFFrame(message)
            elif entry == 'NEWFRAMES':
                h5f.putFrames(message)
            elif entry == 'NEWPZFFRAMES':
                h5f.putPZFFrames(message)
            elif entry == 'NEWEVENT':
                h5f.putEvent(message)
            elif entry == 'METADATA':
                h5f.putMetadata(message)
            elif entry == 'METADATAENTRY':
                h5f.putMetadataEntry(message)
            else:
                self.send_error(404, 'Operation Not Supported')
                return
        
            #print 'About to return'        
            self.send_response(200, '')
            #print 'sr'
            #self.send_header("Connection", "keep-alive")
            self.end_headers()
            #self.wfile.write('')
            self.wfile.close()
            
            #print 'here we go'
            return

        self.send_error(404, 'Operation Not Supported')
        return

if __name__ == '__main__':
    from BaseHTTPServer import HTTPServer
    server = HTTPServer(('localhost', 8080), GetHandler)
    print 'Starting server, use <Ctrl-C> to stop'
    server.serve_forever()