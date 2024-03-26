"""A mixin for wxPython windows that allows for cascading layout of parent windows.
This allows easier hiding and showing of nested panels.

This is a cleaner and more easily readable replacement for the previous `fold1()` method in the fold panels.
"""


class CascadingLayoutMixin(object):
    def cascading_layout(self):
        """Lays out the parent window in a cascading fashion.
        """
        self.Layout()

        try:
            self.GetParent().cascading_layout()
        except AttributeError:
            try:
                self.GetParent().Layout()
            except AttributeError:
                pass