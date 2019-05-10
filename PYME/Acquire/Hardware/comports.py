class ComPort:
    def __init__(self, port, doc=None):
        import re
        try:
            m = re.match('^COM(\d+)',port)
        except TypeError:
            m = None
        if m:
            self.pnum = int(m.group(1))
        else:
            try:
                self.pnum = int(port)
            except TypeError:
                print("could not parse port name: %s" % port)

        self.pname = 'COM%d' % self.pnum
        self.doc = doc

    def portname(self):
        return self.pname

    def portnumber(self):
        return self.pnum

    def getdoc(self):
        return self.doc
