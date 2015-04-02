# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:58:22 2013

@author: David Baddeley
"""

import numpy as np
import socket
from PIL import Image

try:
    import StringIO
except ImportError:
    import io as StringIO

class ENList(list):
    def __getattr__(self, key):
        return self.index(key)

#Packet types
LC_PACKET_TYPE = ENList(['SYSTEM_BUSY', 'ERROR', 'HOST_WRITE', 'WRITE_RESPONSE', 'HOST_READ', 'READ_RESPONSE'])
#LC_PACKET_TYPE = {v:i for i,v in enumerate(LC_PACKET_TYPE_BY_VALUE)}

#error codes
LC_ERROR = ENList([
    'SUCCESS',
    'FAIL',
    'ERR_OUT_OF_RESOURCE',
    'ERR_INVALID_PARAM',
    'ERR_NULL_PTR',
    'ERR_NOT_INITIALIZED',
    'ERR_DEVICE_FAIL',
    'ERR_DEVICE_BUSY',
    'ERR_FORMAT_ERROR',
    'ERR_TIMEOUT',
    'ERR_NOT_SUPPORTED',
    'ERR_NOT_FOUND'])


 

CMD_VERSION_STRING = 0x0100
CMD_DISPLAY_MODE = 0x0101
CMD_TEST_PATTERN = 0x0103
CMD_LED_CURRENT = 0x0104
CMD_STATIC_IMAGE = 0x0105
CMD_STATIC_COLOR = 0x0106
CMD_DISPLAY_SETTING = 0x0107
CMD_VIDEO_SETTING = 0x0200
CMD_VIDEO_MODE = 0x0201
CMD_PATTERN_SETTING = 0x0400
CMD_PATTERN_DEFINITION = 0x0401
CMD_PATTERN_START = 0x0402
CMD_PATTERN_ADVANCE = 0x0403
CMD_TRIGGER_OUT = 0x0404
CMD_DISPLAY_PATTERN = 0x0405
CMD_PATTERN_EXTENDED_SETTING = 0x0480
CMD_PATTERN_EXTENDED_DEFINITION = 0x0481
CMD_CAMERA_CAPTURE = 0x0500
CMD_LOADSAVE_SOLUTION = 0x0600
CMD_MANAGE_SOLUTION = 0x0601
CMD_INSTALL_FIRMWARE = 0x0700
CMD_SET_IP_ADDRESS = 0x0800
CMD_SET_REGISTER = 0xFF00

PACKET_FLAGS = ENList(['COMPLETE', 'BEGIN', 'MIDDLE', 'END'])

PAYLOAD_MAX_SIZE = 65535
DATA_MAX_SIZE = PAYLOAD_MAX_SIZE - 7

HEADER_DTYPE = np.dtype([('pktType', 'uint8'), ('command', '>u2'), ('flag', 'uint8'), ('datalength', 'uint16')])
IMAGE_DTYPE = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])

PATTERN_INFO_DTYPE = np.dtype([('depth', 'u1'), ('nPatterns', 'u1'), ('invert', 'u1'), ('trigger', 'u1'), ('triggerDelay', 'u4'), ('triggerPeriod', 'u4'), ('exposureTime', 'u4'), ('LEDSelect', 'u1')])
PATTERN_TRIGGER = ENList(['COMMAND', 'AUTO', 'EXT_POS', 'EXT_NEG', 'CAM_POS','CAM_NEG','EXT_EXP'])


def MakePacket(pktType, command, flag, data):
    pktType = np.uint8(pktType)
    command = np.uint16(command)
    flag = np.uint8(flag)
    pkt = np.zeros(data.nbytes + 7, 'uint8')
    pktv = pkt.view('|S1')
    pkt[0] = pktType
    pktv[1] = command.data[1]
    pktv[2] = command.data[0]
    pkt[3] = flag
    pkt[4:6] = np.uint16(data.nbytes).view('2uint8')
    pkt[6:-1] = data.view('uint8')
    pkt[-1] = np.mod(pkt.sum(), 0x100)
    
    return pkt.data
    
def Packetize(pktType, command, data, endFlag=PACKET_FLAGS.COMPLETE, startFlag=PACKET_FLAGS.BEGIN):
    if data.nbytes <= DATA_MAX_SIZE:
        return [MakePacket(pktType, command, endFlag, data)]
    else:
        return [MakePacket(pktType, command, startFlag, data.view('uint8')[:DATA_MAX_SIZE])] + Packetize(pktType, command, data.view('uint8')[DATA_MAX_SIZE:], PACKET_FLAGS.END, PACKET_FLAGS.MIDDLE)
    
def DecodePacket(pkt):
    pkt = np.fromstring(pkt, 'uint8')
    header = pkt[:6].view(HEADER_DTYPE)
    data = pkt[6:-1]
    
    return header, data

class LightCrafter(object):
    X, Y = np.mgrid[0:608, 0:684]
    
    DISPLAY_MODE= ENList([
        'DISP_MODE_IMAGE',		#/* Static Image */
        'DISP_MODE_TEST_PTN',		#/* Internal Test pattern */
        'DISP_MODE_VIDEO',		#/* HDMI Video */
        'DISP_MODE_VIDEO_INT_PTN',	#/* Interleaved pattern */
        'DISP_MODE_PTN_SEQ',		#/* Pattern Sequence */
        'DISP_NUM_MODES'])
 
    TEST_PATTERN = ENList([
        'CHECKERBOARD',
        'SOLID_BLACK',
        'SOLID_WHITE',
        'SOLID_GREEN',
        'SOLID_BLUE',
        'SOLID_RED',
        'VERT_LINES',    
        'HORIZ_LINES',
        'VERT_FINE_LINES',
        'HORIZ_FINE_LINES',
        'DIAG_LINES',
        'VERT_RAMP',
        'HORIZ_RAMP',
        'ANSI_CHECKERBOARD'])
        
    def __init__(self, IPAddress='192.168.1.100'):
        self.IPAddress = IPAddress
        self.sock = None
        
        self.StoredMasks = {}
        
    def Connect(self):
        self.sock = socket.socket()
        self.sock.settimeout(1)
        self.sock.connect((self.IPAddress, 0x5555))
        
    
    def Close(self):
        self.sock.close()
        
    def _send(self, msg):
        totalsent = 0
        while totalsent < len(msg):
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent

    def _receive(self, MSGLEN):
        msg = ''
        while len(msg) < MSGLEN:
            chunk = self.sock.recv(MSGLEN-len(msg))
            if chunk == '':
                raise RuntimeError("socket connection broken")
            msg = msg + chunk
        return msg
        
    def _ExecCommand(self, pktType, command, data):
        #send data
        packets = Packetize(pktType, command, data)
        for pkt in packets:
            self._send(pkt)
            
        #read reply
        header = np.fromstring(self._receive(6), 'uint8').view(HEADER_DTYPE)
        payload = np.fromstring(self._receive(header['datalength'] +1), 'uint8')
        data = payload[:-1]
        checksum = payload[-1]
        
        while not header['flag'] in [PACKET_FLAGS.COMPLETE, PACKET_FLAGS.END]:
            header = np.fromstring(self._receive(6), 'uint8').view(HEADER_DTYPE)
            payload = np.fromstring(self._receive(header['datalength'] +1), 'uint8')
            data = np.hstack([data, payload[:-1]])
            
        return header, data
        
    def SetDisplayMode(self, mode):
        self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_DISPLAY_MODE, np.uint8(mode))
        
    def SetImage(self, data):
        im = Image.fromarray(data)
        output = StringIO.StringIO()
        im.save(output, format='BMP')
        contents = output.getvalue()
        output.close()
        #print contents[:50], np.fromstring(contents, 'u1')[:50]
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_STATIC_IMAGE, np.fromstring(contents, 'u1'))
        return h, d
        
    def SetPatternDefs(self, dataFrames):
        patternSettings = np.zeros(1,PATTERN_INFO_DTYPE)
        patternSettings['depth'] = 1
        patternSettings['nPatterns'] = len(dataFrames)
        patternSettings['invert'] = 0
        patternSettings['trigger'] = PATTERN_TRIGGER.COMMAND
        #patternSettings[']
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_PATTERN_SETTING, patternSettings)
        print(( h, d))
        
        for index, data in enumerate(dataFrames):
            im = Image.fromarray(data)#.convert('1')
            output = StringIO.StringIO()
            im.save(output, format='BMP')
            contents = output.getvalue()
            output.close()
            #print contents[:50], np.fromstring(contents, 'u1')[:50]
            h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_PATTERN_DEFINITION, np.hstack([np.ubyte(index),np.fromstring(contents, 'u1')]))
            print(( h, d))
        return h, d
        
    def SetPattern(self, index):
        self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_DISPLAY_PATTERN, np.uint8(index))
        
    def SetTestPattern(self, pattern):
        #print contents[:50], np.fromstring(contents, 'u1')[:50]
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_TEST_PATTERN, np.uint8(pattern))
        return h, d
        
    def SetMask(self, data, intensity = 255):
        return self.SetImage(((data > 0)*intensity).astype('uint8'))
        
    def SetStoredMask(self, key, intensity = 255):
        self.SetMask(self.StoredMasks[key], intensity)
        
    def SetStatic(self, value):
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_STATIC_COLOR, np.uint8(value*np.array([0,1,1,1])))
        return h, d
        
    def StartPatternSeq(self, start=True):
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_PATTERN_START, np.uint8(start))
        return h, d
    
    def AdvancePatternSeq(self, start=True):
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_PATTERN_ADVANCE, np.empty(0))
        return h, d
        
    def SetSpot(self, x, y, radius=10, intensity=255):
        self.SetMask(((self.X - x)**2 + (self.Y-y)**2) < radius**2, intensity)
        
    def SetLineIllumination(self, period, phase=0, angle=0, boolean=True, intensity=255):
        kx = np.cos(angle)*2*np.pi/period
        ky = np.sin(angle)*2*np.pi/period
        
        ll = .5 + .5*np.cos(self.X*kx+ self.Y*ky + phase)
        
        if  boolean:
            self.SetImage(((ll > .5)*intensity).astype('uint8'))
        else:
            self.SetImage((ll*intensity).astype('uint8'))
            
    def SetVDCLines(self, period, phase=0, angle=0, dutyCycle=.5, intensity=255.):
        kx = np.cos(angle)/period
        ky = np.sin(angle)/period
        
        d = kx*self.X + ky*self.Y + phase
        
        ll = (d%1) < dutyCycle
        
        self.SetImage((ll*intensity).astype('uint8'))