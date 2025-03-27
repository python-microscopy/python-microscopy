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

import logging
logger = logging.getLogger(__name__)

try:
    # Put this in a try-except clause as a) the quaternion module is not packaged yet and b) it has a dependency on a recent numpy version
    # so we might not want to make it a dependency yet.
    import quaternion
    HAVE_QUATERNION = True
except ImportError:
    print('quaternion module not found, disabling custom clip plane orientations')
    HAVE_QUATERNION = False


class BindMixin(object):
    """ Contains the core logic for binding data to OpenGL buffers """

    def __init__(self) -> None:
        self._bound_data = {}

    def __del__(self):
        for vao, vbo, sig in self._bound_data.values():
            from OpenGL.GL import glDeleteVertexArrays, glDeleteBuffers
            glDeleteVertexArrays(1, [vao,])
            glDeleteBuffers(3, vbo)

    def _bind_data_core(self, name, vertices, normals, colors, sp):
        from OpenGL.GL import glGenVertexArrays, glGenBuffers, glDeleteBuffers, glDeleteVertexArrays, glBindVertexArray, glBindBuffer, glBufferData, glVertexAttribPointer, glEnableVertexAttribArray, GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE
        
        n_vertices = vertices.shape[0]
        if n_vertices == 0:
            return False
        
        sig = (id(vertices), id(normals), id(colors), id(sp)) # signature for this data - do not rebind if the same data is already bound

        # check to see if old vao and vbo if they exist, and if they point to the same data
        # remove them if they exist byt are differeng      
        old_vao, old_vbo, old_sig = self._bound_data.get(name, (None, None, None))
        if old_sig == sig:
            glBindVertexArray(old_vao) # rebind the old vao  - TODO should we defer this to client?
            #logger.debug('Reusing existing VAO')
            return n_vertices
        elif old_vao is not None:
            glBindVertexArray(0) # unbind the old vao
            glBindBuffer(GL_ARRAY_BUFFER, 0) # unbind the old vbo
            glDeleteVertexArrays(1, [old_vao,])
            glDeleteBuffers(3, old_vbo)
        
        vertices = np.ascontiguousarray(vertices, 'f')
        normals = np.ascontiguousarray(normals, 'f')
        colors = np.ascontiguousarray(colors, 'f')

        #print('vertices_dtype = ', vertices.dtype)
        
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(3)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._bound_data[name] = (vao, vbo, sig)

        return n_vertices
    
    def _bind_data_legacy(self, vertices, normals, colors):
        from OpenGL.GL import glVertexPointerf, glNormalPointerf, glColorPointerf
        vertices = np.ascontiguousarray(vertices, 'f')
        normals = np.ascontiguousarray(normals, 'f')
        colors = np.ascontiguousarray(colors, 'f')

        #glVertexPointer expects an Nx3 array
        #glNormalPointer expects an Nx3 array
        #glColorPointer expects an Nx4 array
        if vertices.ndim == 1:
            vertices = vertices.reshape([-1, 3])

        if colors.ndim == 1:
            colors = colors.reshape([-1,4])

        if normals.ndim == 1:
            normals = normals.reshape([-1,3])

        n_vertices = vertices.shape[0]

        glVertexPointerf(vertices)
        glNormalPointerf(normals)
        glColorPointerf(colors)

        return n_vertices
    
    def _bind_data(self, name, vertices, normals, colors, sp, core_profile=True):
        if core_profile:
            return self._bind_data_core(name, vertices, normals, colors, sp)
        else:
            return self._bind_data_legacy(vertices, normals, colors)
        
    @classmethod
    def _gen_rect_triangles(cls, x0, y0, w, h, z=0.0):
        # generate two triangles to make a rectangle (as a replacement for deprecated GL_QUADS)
        # TODO - move this somewhere more sensible

        return np.array([[x0, y0, z],  
                         [x0+w, y0, z], 
                         [x0+w, y0+h, z],
                         [x0+w, y0 + h, z], 
                         [x0, y0+h, z],
                         [x0, y0, z]], 'f')
    
    @classmethod
    def _gen_rect_texture_coords(cls):
        return np.array([[0., 0.], 
                         [1., 0.], 
                         [1., 1.], 
                         [1., 1.], 
                         [0., 1.], 
                         [0., 0.]], 'f')
    
class ShaderMixin(object):
    """ Contains the core logic for managing shaders """

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
    
    def get_specific_shader_program(self, canvas, shader_program_cls):
        '''Get a specific shader program instance for this engine. This is useful for engines that need to use multiple
        different shaders. Added to allow points layers to switch between using points and using quads as the desired
        point size exceeds the maximum point size supported by the hardware.'''
        return ShaderProgramFactory.get_program(shader_program_cls, canvas.gl_context, canvas)
      

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
        params = [n for n in self.class_trait_names() if not n.startswith('_')]
        d = self.trait_get(params)
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
    
class SimpleLayer(BaseLayer, ShaderMixin):
    """
    Layer base class for layers which do their own rendering and manage their own shaders

    Derived classes should implement the render method to draw to the canvas.
    """
    def __init__(self, **kwargs):
        BaseLayer.__init__(self, **kwargs)
        ShaderMixin.__init__(self)
    

class BaseEngine(BindMixin, ShaderMixin):
    def __init__(self):
        BindMixin.__init__(self)
        ShaderMixin.__init__(self)

    @abc.abstractmethod
    def render(self, gl_canvas, layer):
        pass    
    

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
    
    

    
