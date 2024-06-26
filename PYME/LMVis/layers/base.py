#!/usr/bin/python

# layers.py
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
"""
Test module docstring
"""

import abc
import warnings

from PYME.LMVis.shader_programs.ShaderProgramFactory import ShaderProgramFactory

from PYME.recipes.traits import HasTraits, Bool, Instance
import numpy as np

try:
    # Put this in a try-except clause as a) the quaternion module is not packaged yet and b) it has a dependency on a recent numpy version
    # so we might not want to make it a dependency yet.
    import quaternion
    HAVE_QUATERNION = True
except ImportError:
    print('quaternion module not found, disabling custom clip plane orientations')
    HAVE_QUATERNION = False

class BaseEngine(object):
    def __init__(self):
        self._shader_program_cls = None
    
    def set_shader_program(self, shader_program):
        #self._shader_program = ShaderProgramFactory.get_program(shader_program, self._context, self._window)
        self._shader_program_cls = shader_program

    @property
    def shader_program(self):
        warnings.warn(DeprecationWarning('Use get_shader_program(canvas) instead'))
        return ShaderProgramFactory.get_program(self._shader_program_cls)
    
    def get_shader_program(self, canvas):
        return ShaderProgramFactory.get_program(self._shader_program_cls, canvas.gl_context, canvas)

    @abc.abstractmethod
    def render(self, gl_canvas, layer):
        pass
    
    def _set_shader_clipping(self, gl_canvas):
        sp = self.get_shader_program(gl_canvas)
        sp.xmin, sp.xmax = gl_canvas.bounds['x'][0]
        sp.ymin, sp.ymax = gl_canvas.bounds['y'][0]
        sp.zmin, sp.zmax = gl_canvas.bounds['z'][0]
        sp.vmin, sp.vmax = gl_canvas.bounds['v'][0]
        if False:#HAVE_QUATERNION:
            sp.v_matrix[:3, :3] = quaternion.as_rotation_matrix(gl_canvas.view.clip_plane_orientation)
            sp.v_matrix[3, :3] = -gl_canvas.view.clip_plane_position
        else:
            #use current view
            sp.v_matrix[:,:] = gl_canvas.object_rotation_matrix
            sp.v_matrix[3,:3] = -np.linalg.lstsq(gl_canvas.object_rotation_matrix[:3,:3], gl_canvas.view.translation, rcond=None)[0]
        

class BaseLayer(HasTraits):
    """
    This class represents a layer that should be rendered. It should represent a fairly high level concept of a layer -
    e.g. a Point-cloud of data coming from XX, or a Surface representation of YY. If such a layer can be rendered multiple
    different but similar ways (e.g. points/pointsprites or shaded/wireframe etc) which otherwise share common settings
    e.g. point size, point colour, etc ... these representations should be coded as one layer with a selectable rendering
    backend or 'engine' responsible for managing shaders and actually executing the opengl code. In this case use the
    `EngineLayer` class as a base
.
    In simpler cases, such as rendering an overlay it is acceptable for a layer to do it's own rendering and manage it's
    own shader. In this case, use `SimpleLayer` as a base.

    """
    visible = Bool(True)
    
    def __init__(self, **kwargs):
        HasTraits.__init__(self, **kwargs)
        
    @property
    def bbox(self):
        """Bounding box in form [x0,y0,z0, x1,y1,z1] (or none if a bounding box does not make sense for this layer)
        over-ride in derived classes
        """
        return None

    

    @abc.abstractmethod
    def render(self, gl_canvas):
        """
        Abstract render method to be over-ridden in derived classes. Should check self.visible before drawing anything.
        
        Parameters
        ----------
        gl_canvas : the canvas to draw to - an instance of PYME.LMVis.gl_render3D_shaders.LMGLShaderCanvas


        """
        pass

    def settings_dict(self):
        d =  self.get([n for n in self.class_trait_names() if not n.startswith('_')])
        for k, v in d.items():
            if isinstance(v, dict) and not type(v) == dict:
                v = dict(v)
            elif isinstance(v, list) and not type(v) == list:
                v = list(v)
            elif isinstance(v, set) and not type(v) == set:
                v = set(v)

            d[k] = v
        return d
    
    def serialise(self):
        return {'type': self.__class__.__name__,
                'settings': self.settings_dict()}
    
class EngineLayer(BaseLayer):
    """
    Base class for layers who delegate their rendering to an engine.
    """
    engine = Instance(BaseEngine)
    show_lut = Bool(True)

    def render(self, gl_canvas):
        if self.visible:
            return self.engine.render(gl_canvas, self)
        
    def settings_dict(self):
        r = BaseLayer.settings_dict(self)
        r.pop('engine') # don't include the enginge in the dict represetation (this is set by `method`)
        return r
        
    
    @abc.abstractmethod
    def get_vertices(self):
        """
        Provides the engine with a way of obtaining vertex data. Should be over-ridden in derived class
        
        Returns
        -------
        a numpy array of vertices suitable for passing to glVertexPointerf()

        """
        raise(NotImplementedError())

    @abc.abstractmethod
    def get_normals(self):
        """
        Provides the engine with a way of obtaining vertex data. Should be over-ridden in derived class

        Returns
        -------
        a numpy array of normals suitable for passing to glNormalPointerf()

        """
        raise (NotImplementedError())

    @abc.abstractmethod
    def get_colors(self):
        """
        Provides the engine with a way of obtaining vertex data. Should be over-ridden in derived class

        Returns
        -------
        a numpy array of vertices suitable for passing to glColorPointerf()

        """
        raise (NotImplementedError())
    
    
class SimpleLayer(BaseLayer):
    """
    Layer base class for layers which do their own rendering and manage their own shaders
    """
    def __init__(self, **kwargs):
        BaseLayer.__init__(self, **kwargs)
        self._shader_program_cls = None
    
    def set_shader_program(self, shader_program):
        #self._shader_program = ShaderProgramFactory.get_program(shader_program, self._context, self._window)
        self._shader_program_cls = shader_program

    @property
    def shader_program(self):
        warnings.warn(DeprecationWarning('Use get_shader_program(canvas) instead'))
        return ShaderProgramFactory.get_program(self._shader_program_cls)
    
    def get_shader_program(self, canvas):
        return ShaderProgramFactory.get_program(self._shader_program_cls, canvas.gl_context, canvas)
    
    def _clear_shader_clipping(self, canvas):
        sp = self.get_shader_program(canvas)
        sp.xmin, sp.xmax = [-1e6, 1e6]
        sp.ymin, sp.ymax = [-1e6, 1e6]
        sp.zmin, sp.zmax = [-1e6, 1e6]
        sp.vmin, sp.vmax = [-1e6, 1e6]