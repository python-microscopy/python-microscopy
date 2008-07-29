WantEventNotification = []

def logEvent(eventName, eventDescr = ''):
    for evl in WantEventNotification:
            evl.logEvent(eventName, eventDescr)