import wx
import logging

logging.basicConfig(filename='/tmp/vspy.log',level=logging.DEBUG)

#class MyEventHandler(wx.EvtHandler):
#    def __init__(self, *args, **kwargs):
#        wx.EvtHandler.__init__(self, *args, **kwargs)

#    def ProcessEvent(self, evt):
#        logging.info('Event')
#        print 'Event: ' #+ evt
#        return wx.EvtHandler.ProcessEvent(self, evt)

eventIds = {wx.EVT_ENTER_WINDOW.typeId : 'EVT_ENTER_WINDOW',
wx.EVT_LEAVE_WINDOW.typeId : 'EVT_LEAVE_WINDOW',
wx.EVT_LEFT_DOWN.typeId : 'EVT_LEFT_DOWN',
wx.EVT_LEFT_UP.typeId : 'EVT_LEFT_UP',
wx.EVT_LEFT_DCLICK.typeId : 'EVT_LEFT_DCLICK',
wx.EVT_MIDDLE_DOWN.typeId : 'EVT_MIDDLE_DOWN',
wx.EVT_MIDDLE_UP.typeId : 'EVT_MIDDLE_UP',
wx.EVT_MIDDLE_DCLICK.typeId : 'EVT_MIDDLE_DCLICK',
wx.EVT_RIGHT_DOWN.typeId : 'EVT_RIGHT_DOWN',
wx.EVT_RIGHT_UP.typeId : 'EVT_RIGHT_UP',
wx.EVT_RIGHT_DCLICK.typeId : 'EVT_RIGHT_DCLICK',
wx.EVT_MOTION.typeId : 'EVT_MOTION',
wx.EVT_MOUSEWHEEL.typeId : 'EVT_MOUSEWHEEL'}

eventAttrs = ['m_altDown',
'm_controlDown',
'm_leftDown',
'm_middleDown',
'm_rightDown',
'm_metaDown',
'm_shiftDown',
'm_x',
'm_y',
'm_wheelRotation',
'm_wheelDelta',
'm_linesPerAction']


lastEvents = []

class eventInfo:
    def __init__(self,event):
        self.type = event.GetEventType()
        self.attrs = {}

        for attrName in eventAttrs:
            self.attrs[attrName] = event.__getattribute__(attrName)

    def __repr__(self):
        buttons = ''

        if self.attrs['m_leftDown']:
            buttons += 'l'

        if self.attrs['m_rightDown']:
            buttons += 'r'

        if self.attrs['m_middleDown']:
            buttons += 'm'

        return eventIds[self.type] + ' (%d, %d) %s' % (self.attrs['m_x'], self.attrs['m_y'], buttons)

    def recreate(self):
        e = wx.MouseEvent(self.type)
        for attrName in eventAttrs:
            e.__setattr__(attrName, self.attrs[attrName])

        return e
        

def MouseEvent(event):
    #global lastEvent
    #print event.GetEventType()
    lastEvents.append(eventInfo(event))
    print lastEvents[-1]
    event.Skip()

def InstallSpy(window):
    #get the current event handler
    eh = window.GetEventHandler()

    #install our custom event handler at the top of the stack
    meh = wx.EvtHandler()
    #window.SetEventHandler(meh)

    meh.Bind(wx.EVT_MOUSE_EVENTS, MouseEvent)

    #push the original handler back onto the stack so everything works as expected
    window.PushEventHandler(meh)

def PlayEvents(events, window):
    for evt in events:
        window.AddPendingEvent(evt.recreate())

    #return meh

