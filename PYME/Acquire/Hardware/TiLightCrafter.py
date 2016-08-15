# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:58:22 2013

@author: David Baddeley
"""

import numpy as np
import socket
from PIL import Image
from PYME.IO import MetaDataHandler

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
CMD_PATTERN_SETTING = 0x0480
CMD_PATTERN_DEFINITION = 0x0481
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

PATTERN_INFO_DTYPE = np.dtype([('depth', 'u1'), ('nPatterns', 'u2'), ('invert', 'u1'), ('trigger', 'u1'), ('triggerDelay', 'u4'), ('triggerPeriod', 'u4'), ('exposureTime', 'u4'), ('LEDSelect', 'u1'), ('playMode', 'u1')])
PATTERN_TRIGGER = ENList(['COMMAND', 'AUTO', 'EXT_POS', 'EXT_NEG', 'CAM_POS','CAM_NEG','EXT_EXP'])

PATTERN_START_DTYPE = np.dtype([('patternNum', 'u2'), ('columnPos', 'u2'), ('rowPos', 'u2')]) 


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
    
#    DISPLAY_MODE= ENList([
#        'DISP_MODE_IMAGE',		#/* Static Image */
#        'DISP_MODE_TEST_PTN',		#/* Internal Test pattern */
#        'DISP_MODE_VIDEO',		#/* HDMI Video */
#        'DISP_MODE_VIDEO_INT_PTN',	#/* Interleaved pattern */
#        'DISP_MODE_PTN_SEQ',		#/* Pattern Sequence */
#        'DISP_NUM_MODES'])
        
    DISPLAY_MODE= ENList([
        'DISP_MODE_IMAGE',		#/* Static Image */
        'DISP_MODE_TEST_PTN',		#/* Internal Test pattern */
        'DISP_MODE_VIDEO',		#/* HDMI Video */
        'DISP_MODE_PTN_SEQ',		#/* Pattern Sequence */
        ])
 
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
        self.PatternVars = {'Period':'', 'DutyCycle':'', 'Angle':'', 'Phase':'', 'ExpTime':'', 'Steps':'', 'Type':'', 'MaskName':''}
        
        #register as a provider of metadata
        MetaDataHandler.provideStartMetadata.append(self.GenStartMetadata)
        
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
        
    def SetPatternDefs(self, dataFrames, triggerMode = PATTERN_TRIGGER.AUTO, exposureMs = 1000, playMode = 1):
        patternSettings = np.zeros(1,PATTERN_INFO_DTYPE)
        patternSettings['depth'] = 1
        patternSettings['nPatterns'] = len(dataFrames)
        patternSettings['invert'] = 0
        patternSettings['trigger'] = triggerMode
        
        patternSettings['exposureTime'] = exposureMs*1000
        patternSettings['LEDSelect'] = 1
        patternSettings['playMode'] = playMode
        
        #print patternSettings, patternSettings.view('uint8')
        #patternSettings[']
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_PATTERN_SETTING, patternSettings)
        #print(( h, d))
        
        for index, data in enumerate(dataFrames):
            im = Image.fromarray(data).convert('1')
            output = StringIO.StringIO()
            im.save(output, format='BMP')
            contents = output.getvalue()
            output.close()
            #print contents[:50], np.fromstring(contents, 'u1')[:50]
            h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_PATTERN_DEFINITION, np.fromstring(np.hstack([np.uint16(index),np.uint16(0),np.uint16(0)]).data + contents, 'u1'))
            #print(( h, d))
        return h, d
        
    def SetPattern(self, index):
        self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_DISPLAY_PATTERN, np.uint8(index))
        
    def SetTestPattern(self, pattern):
        #print contents[:50], np.fromstring(contents, 'u1')[:50]
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_TEST_PATTERN, np.uint8(pattern))
        return h, d
        
    def SetMask(self, data, intensity = 255, Name = ''):
        self.PatternVars['Type'] = 'Mask'
        self.PatternVars['MaskName'] = Name
        return self.SetImage(((data > 0)*intensity).astype('uint8'))
        
    def SetStoredMask(self, key, intensity = 255):
        self.PatternVars['Type'] = 'Mask'
        self.SetMask(self.StoredMasks[key], intensity)
        
    def SetStatic(self, value):
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_STATIC_COLOR, np.uint8(value*np.array([0,1,1,1])))
        return h, d
        
    def StartPatternSeq(self, start=True):
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_PATTERN_START, np.uint8(start))
        return h, d
    
    def AdvancePatternSeq(self, start=True, dummyX=0, dummyY=0):
        h, d = self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_PATTERN_ADVANCE, np.empty(0))
        return h, d
        
    def SetLeds(self, red=274, green=274, blue=274):
        self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_LED_CURRENT, np.array([red, green, blue]).astype('uint16').view('u1'))
        
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
            
    def SetScanningVDC(self, period, exposureMs=100, angle=0, dutyCycle=.5, nSteps = 20, double = True):
        width = int(period*dutyCycle)
        dd = 1
        if double:
          if width%2 == 1:
            width += 1
          dd = 2
        nSteps = int(1/dutyCycle)
        period = nSteps * width
        nSteps = dd * nSteps
        pats = [self.GenVDCLines(period, phase=p, angle = angle, width = width).T for p in np.linspace(0, period, nSteps, False)]
        pats1, pats2 = pats[:(len(pats)//2)], pats[(len(pats)//2):]
        pats3 = [val for pair in zip(pats1, pats2) for val in pair]
        if len(pats3) < len(pats):
            pats3 += [max(pats1, pats2, key = len)[-1]]

        from PYME.DSView import View3D
        View3D(np.array(pats3).transpose(2,1,0), 'stack')
        View3D(np.array(pats3).transpose(2,1,0).mean(2), 'mean')
        
        self.SetPatternDefs(pats3, exposureMs=exposureMs)
        self.StartPatternSeq()
        
        self.PatternVars['Type'] = 'Lines'
        self.PatternVars['Period'] = period
        self.PatternVars['DutyCycle'] = 1.0*width/period
        self.PatternVars['Phase'] = ''
        self.PatternVars['Angle'] = angle
        self.PatternVars['ExpTime'] = exposureMs
        self.PatternVars['Steps'] = nSteps
        
    def SetCalibrationPattern(self):
        pats = []
        pats.append( (((((self.X-200)**2 + (self.Y-200)**2) < 100) > 0)*255).astype('uint8').T )
        pats.append( (((((self.X-200)**2 + (self.Y-300)**2) < 100) > 0)*255).astype('uint8').T )
        pats.append( (((((self.X-300)**2 + (self.Y-200)**2) < 100) > 0)*255).astype('uint8').T )
        pats.append( (((((self.X-300)**2 + (self.Y-300)**2) < 100) > 0)*255).astype('uint8').T )
        pats.append( (((((self.X-300)**2 + (self.Y-400)**2) < 100) > 0)*255).astype('uint8').T )
        pats.append( (((((self.X-400)**2 + (self.Y-300)**2) < 100) > 0)*255).astype('uint8').T )
        
        self.PatternVars['Type'] = 'Calibration'
        
        self.SetPatternDefs(pats, triggerMode = PATTERN_TRIGGER.COMMAND, exposureMs=500, playMode = 0)
        self.StartPatternSeq()
        
    def SetScanningHex(self, period, exposureMs=100, angle=0, dutyCycle=.5, nSteps = 20):
        pats = []
        periodX, periodY, dc = period, int(period/np.tan(np.pi/6)), np.sqrt(dutyCycle)
        widthX, widthY = int(periodX*dc), int(periodY*dc)
        if widthX%2 == 1:
          widthX += 1
        while widthY%4 > 0:
          widthY += 1
        nSteps = int(1/dc)
        periodX, periodY = nSteps * widthX, nSteps * widthY
        for py in np.linspace(0, periodY, nSteps, False):
            for px in np.linspace(0, periodX, nSteps, False):
                pats.append(self.GenHex(periodX, periodY, widthX = widthX, widthY = widthY, phasex=px, phasey = py).T)
        from PYME.DSView import View3D
        pats1, pats2 = pats[:len(pats)//2], pats[len(pats)//2:]
        pats3 = [val for pair in zip(pats1, pats2) for val in pair]
        
        if len(pats3) < len(pats):
          pats3 += [max(pats1, pats2, key = len)[-1]]
        View3D(np.array(pats3).transpose(2,1,0), 'stack')
        View3D(np.array(pats3).transpose(2,1,0).mean(2), 'mean')
        
        self.SetPatternDefs(pats3, exposureMs=exposureMs)
        self.StartPatternSeq()
        
        self.PatternVars['Type'] = 'Hex'
        self.PatternVars['Period'] = periodX
        self.PatternVars['DutyCycle'] = 1.0*widthX/periodX
        self.PatternVars['Phase'] = ''
        self.PatternVars['Angle'] = angle
        self.PatternVars['ExpTime'] = exposureMs
        self.PatternVars['Steps'] = nSteps
    
    def SetVDCLines(self, period, phase=0, angle=0, dutyCycle=.5, intensity=255.):
        self.SetImage(self.GenVDCLines(period, phase, angle, dutyCycle, intensity))
        
    def GenVDCLines(self, period, phase=0, angle=0, dutyCycle=.5, width=15, intensity=255.):
        if angle == 0:
            d = self.X + phase
            ll = (d%period) < width
        else:
            kx = np.cos(angle)/period
            ky = np.sin(angle)/period
            
            d = kx*self.X + ky*self.Y + phase
            
            ll = (d%1) < dutyCycle
        
        return (ll*intensity).astype('uint8')
        
    def GenHex(self, periodX = 100, periodY = 170, widthX = 15, widthY = 25, phasex = 0, phasey = 0, intensity = 255.):
        widthY, offsetX, offsetY = widthY/2, periodX/2, periodY/2
        return (((((self.X + phasex)%periodX) < widthX)*(((self.Y + phasey) %periodY) < widthY) + (((self.X + phasex + offsetX)%periodX) < widthX)*(((self.Y + phasey + offsetY) %periodY) < widthY))*intensity).astype('uint8')
        
    def GenStartMetadata(self, mdh):   
        mdh['DMD.Name'] = 'TI DLP DM365'
        for key in self.PatternVars.keys():
            mdh['DMD.' + key] = self.PatternVars[key]
