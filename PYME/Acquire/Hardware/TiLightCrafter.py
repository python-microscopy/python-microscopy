# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:58:22 2013

@author: David Baddeley
"""

import numpy as np
import socket

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

LC_DISPLAY_MODE= ENList([
	'DISP_MODE_IMAGE',		#/* Static Image */
	'DISP_MODE_TEST_PTN',		#/* Internal Test pattern */
	'DISP_MODE_VIDEO',		#/* HDMI Video */
	'DISP_MODE_VIDEO_INT_PTN',	#/* Interleaved pattern */
	'DISP_MODE_PTN_SEQ',		#/* Pattern Sequence */
	'DISP_NUM_MODES'])
 

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
    def __init__(self, IPAddress='192.168.1.100'):
        self.IPAddress = IPAddress
        self.sock = None
        
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
        self._ExecCommand(LC_PACKET_TYPE.HOST_WRITE, CMD_STATIC_IMAGE, data)