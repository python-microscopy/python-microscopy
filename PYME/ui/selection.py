"""
Common selection settings interface for PYMEVis/PYMEImage

"""
import logging
logger = logging.getLogger(__name__)

class Point(list):
    def __init__(self, x, y, z=None):
        list.__init__(self, [x,y,z])

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

SELECTION_RECTANGLE, SELECTION_LINE, SELECTION_SQUIGGLE = range(3)
UNITS_NM, UNITS_PIXELS = range(2)

class Selection(object):
    def __init__(self, units=UNITS_NM):
        self._start = Point(0, 0, 0)
        self._finish = Point(0, 0, 0)

        self.width = 1
        self.trace = [] # path taken whilst making the selection

        self.mode = SELECTION_RECTANGLE

        if units == UNITS_PIXELS:
            logger.warning('Using units=UNITS_PIXELS')
            
        self.units = units

        #self.colour = [1, 1, 0]
        #self.show = False

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        self._start = Point(*value) 

    @property
    def finish(self):
        return self._finish

    @finish.setter
    def finish(self, value):
        self._finish = Point(*value) 