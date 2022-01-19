#!/usr/bin/python

# snapshot.py
#
# Copyright Michael Graff
#   graff@hm.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import PIL

#from scipy.misc import toimage
import wx
from OpenGL.GL import GL_LUMINANCE, GL_RGB

#FIXME - does this shadow / duplicate existing functionality?

from PYME.recipes import traits

class SnapshotSettings(traits.HasTraits):
    """
    Computes the correct rendering pixelsize to generate the displayed area at a given size and DPI
    """
    ppi = traits.Float(72)
    width_inches = traits.Float(5) # width, reluctantly in inches, as that is what everyone seems to use
    _width_nm = traits.Float()
    _height_nm = traits.Float()
    _view_pixelsize = traits.Float()

    height_inches = traits.Property(traits.Float, depends_on='_height_nm, pixelsize, ppi')
    pixelsize = traits.Property(traits.Float, depends_on='_width_nm, width_inches ,ppi')

    width_pixels = traits.Property(traits.Float, depends_on='_width_nm, pixelsize')
    height_pixels = traits.Property(traits.Float, depends_on='_height_nm, pixelsize')

    filename = traits.File(filter=['*.png']) #"PNG files(*.png)|*.png")

    def _get_height_inches(self):
        return (self._height_nm/self.pixelsize)/self.ppi

    def _get_pixelsize(self):
        return (self._width_nm/(self.width_inches*self.ppi))

    def _get_width_pixels(self):
        return (self._width_nm/self.pixelsize)

    def _set_width_pixels(self, value):
        self.width_inches = value/self.ppi

    def _get_height_pixels(self):
        return (self._height_nm/self.pixelsize)

    def __init__(self, *args, **kwargs):
        if 'viewport_size_px' in kwargs:
            sx, sy = kwargs.pop('viewport_size_px')
            px = kwargs['_view_pixelsize']
            kwargs['_width_nm'] = px*sx
            kwargs['_height_nm'] = px*sy

        if ('width' not in kwargs):
            ppi = kwargs.get('ppi', 72)
            kwargs['width_inches'] = (kwargs['_width_nm']/kwargs['_view_pixelsize'])/ppi
        
        traits.HasTraits.__init__(self, *args, **kwargs)

    def default_traits_view(self):
        import wx
        if wx.GetApp() is None:
            return None
        
        from traitsui.api import View, Item, Group

        return View([Item('ppi'),
                     Item('width_inches'),
                     Item('height_inches', style='readonly'),
                     Item('width_pixels'),
                     Item('height_pixels', style='readonly'),
                     #Item('pixelsize', style='readonly'),
                     Item('_'),
                     Item('filename'),
                    ], buttons=['OK']) #TODO - should we have cancel? Traits update whilst being edited and cancel doesn't roll back


def save_snapshot(canvas):
    #FIXME - This is the WRONG way to do pixel sizes - we should be using a value in nm
    # we also shouldn't be calling a static method of an unrelated class to do this.
    #pixel_size = float(VideoPanel.ask(canvas, message='Please enter the pixel size (1 pixel on the screen = x pixel '
    #                                                  'in the snapshot', default_value='1'))
    
    
    #dlg = wx.TextEntryDialog(canvas, "Snapshot pixel size", "Please enter the desired pixel size in the snapshot", "5")
    #if dlg.ShowModal() == wx.ID_OK:
    if True:
        #pixel_size = float(dlg.GetValue())
        pixel_size=None
        #dlg.Destroy()

        settings = SnapshotSettings(_view_pixelsize=canvas.pixelsize, viewport_size_px=canvas.view_port_size)
        settings.configure_traits(kind='modal')
    
        #file_name = wx.FileSelector('Save current view as', wildcard="PNG files(*.png)|*.png",
        #                flags=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        
        if settings.filename:
            # snap = canvas.getIm(pixel_size, GL_LUMINANCE)
            snap = canvas.getIm(settings.pixelsize, GL_RGB)
            print(snap.dtype, snap.shape, snap.max())
            if snap.ndim == 3:
                img = PIL.Image.fromarray(snap.transpose(1, 0, 2))
                #img = toimage(snap.transpose(1, 0, 2))
            else:
                img = PIL.Image.fromarray(snap.transpose())
                #img = toimage(snap.transpose())
            
            img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            
            if not settings.filename.endswith('.png'):
                img.save('{}.png'.format(settings.filename))
            else:
                img.save('{}'.format(settings.filename))


def Plug(vis_fr):
    vis_fr.AddMenuItem('View', 'Save Snapshot', lambda e: save_snapshot(vis_fr.glCanvas))
