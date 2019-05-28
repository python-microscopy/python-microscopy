import ctypes
from . import GCS_DLL as _gcs


#class MercuryWrap:

class fcnWrap:
    def __init__(self, fcn):
        self.fcn = fcn
        self.___doc__ = fcn.__doc__
    
    def HandleError(self, ID):
        errno = _gcs.GetError(ID)
        raise RuntimeError('GCS error: %d - %s' % (errno, TranslateError(errno)))


class AxesSetter(fcnWrap):
    def __call__(self, ID, szAxes, *args):
        in_args = []
        
        #build input arrays
        for argtype, argname, arg in zip(self.fcn.argtypes[2:], self.fcn.argnames[2:], args):
            if argtype.__name__.startswith('LP_'): #array input
                basetype = ctypes.__dict__[argtype.__name__[3:]]
                
                in_args.append((basetype * len(arg))(*arg))
            else:
                in_args.append(argtype(arg))
        
        #print "%s(%s, '%s', '%s')" % (self.fcn.__name__, ID, szAxes, ','.join([repr(ia)for ia in in_args]))
        succ = self.fcn(ID, szAxes, *in_args)
        if not succ:
            self.HandleError(ID)


class ArraySetter(fcnWrap):
    def __call__(self, ID, *args):
        in_args = []
        
        #build input arrays
        for argtype, argname, arg in zip(self.fcn.argtypes[1:-1], self.fcn.argnames[1:-1], args):
            if argtype.__name__.startswith('LP_'): #array input
                basetype = ctypes.__dict__[argtype.__name__[3:]]
                
                in_args.append((basetype * len(arg))(*arg))
            else:
                in_args.append(argtype(arg))
        
        in_args.append(len(arg))
        
        #print "%s(%s, '%s', '%s')" % (self.fcn.__name__, ID, szAxes, ','.join([repr(ia)for ia in in_args]))
        succ = self.fcn(ID, *in_args)
        if not succ:
            self.HandleError(ID)


class ArrayGetter(fcnWrap):
    def __call__(self, ID, szAxes):
        out_args = []
        
        AxIDs = [1 for id in szAxes]  # have to be all 1 for mercury
        #build input arrays
        for argtype in self.fcn.argtypes[3:-1]:
            if argtype.__name__.startswith('LP_'): #array input
                basetype = ctypes.__dict__[argtype.__name__[3:]]
                
                out_args.append((basetype * len(szAxes))())
            else:
                out_args.append(argtype())
        
        out_args.append(len(szAxes))
        
        #print "%s(%s, '%s', '%s')" % (self.fcn.__name__, ID, szAxes, ','.join([repr(oa)for oa in out_args]))
        succ = self.fcn(ID, szAxes, AxIDs, *out_args)
        if not succ:
            self.HandleError(ID)
        
        if len(out_args) == 1:
            out_args = out_args[0]
        
        return out_args


class AxesGetter(fcnWrap):
    def __call__(self, ID, szAxes):
        out_args = []
        #ret_args = []
        
        #build output arrays
        for argtype in self.fcn.argtypes[2:]:
            if argtype.__name__.startswith('LP_'): #array output
                basetype = ctypes.__dict__[argtype.__name__[3:]]
                
                out_args.append((basetype * len(szAxes))())
                #ret_args.append(out_args[-1])
            else:
                out_args.append(argtype())
        
        #print "%s(%s, '%s', '%s')" % (self.fcn.__name__, ID, szAxes, ','.join([repr(ia)for ia in out_args]))
        succ = self.fcn(ID, szAxes, *out_args)
        if not succ:
            self.HandleError(ID)
        
        if len(out_args) == 1:
            out_args = out_args[0]
        
        return out_args


class StringGetter(fcnWrap):
    def __call__(self, ID):
        buff = ctypes.create_string_buffer(500)
        succ = self.fcn(ID, buff, len(buff))
        if not succ:
            self.HandleError(ID)
        
        return buff.value


class AxesStringGetter(fcnWrap):
    def __call__(self, ID, szAxes):
        buff = ctypes.create_string_buffer(500)
        succ = self.fcn(ID, szAxes, buff, len(buff))
        if not succ:
            self.HandleError(ID)
        
        return buff.value


class ValGetter(fcnWrap):
    def __call__(self, ID):
        argtype = self.fcn.argtypes[-1]
        if argtype.__name__.startswith('LP_'): #array output
            basetype = ctypes.__dict__[argtype.__name__[3:]]
            buff = basetype()
            succ = self.fcn(ID, ctypes.byref(buff))
            if not succ:
                self.HandleError(ID)
            
            return buff.value


class NotImplemented(fcnWrap):
    def __call__(self, ID, *args):
        raise RuntimeError('Function not wrapped')


ConnectRS232  = _gcs.ConnectRS232
IsConnected = _gcs.IsConnected
CloseConnection = _gcs.CloseConnection

ConnectUSB = _gcs.ConnectUSB

def EnumerateUSB(filter=''):
    buff = ctypes.create_string_buffer(500)
    num_devices = _gcs.EnumerateUSB(buff, len(buff), filter.encode())
    
    return int(num_devices), buff.value
    

IsRunningMacro = ValGetter(_gcs.IsRunningMacro)

#SetErrorCheck = _gcs.SetErrorCheck
GetError = _gcs.GetError
TranslateError = StringGetter(_gcs.TranslateError)

GcsGetAnswerSize = NotImplemented(_gcs.GcsGetAnswerSize)
GcsGetAnswer = NotImplemented(_gcs.GcsGetAnswer)
GcsCommandset = NotImplemented(_gcs.GcsCommandset)

InterfaceSetupDlg = NotImplemented(_gcs.InterfaceSetupDlg)





#DEL = _gcs.DEL

MAC_BEG = _gcs.MAC_BEG
MAC_DEL = _gcs.MAC_DEL
#MAC_END = _gcs.MAC_END
MAC_NSTART = _gcs.MAC_NSTART
MAC_START = _gcs.MAC_START

MOV = AxesSetter(_gcs.MOV)

MVR = AxesSetter(_gcs.MVR)
#POS = AxesSetter(_gcs.POS)


#SAI = _gcs.SAI
SPA = AxesSetter(_gcs.SPA)
SVO = AxesSetter(_gcs.SVO)


qERR = ValGetter(_gcs.qERR)
qHLP = StringGetter(_gcs.qHLP)

qIDN = StringGetter(_gcs.qIDN)

#qMAC = NotImplemented(_gcs.qMAC)
qMOV = AxesGetter(_gcs.qMOV)
qONT = AxesGetter(_gcs.qONT)
qPOS = AxesGetter(_gcs.qPOS)

#qSAI = StringGetter(_gcs.qSAI)

qSPA = NotImplemented(_gcs.qSPA)

qSVO = AxesGetter(_gcs.qSVO)
