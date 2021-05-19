#!/usr/bin/python

# GouraudShaderProgram.py
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
from PYME.LMVis.shader_programs.GLProgram import GLProgram
from OpenGL.GL import *

class GouraudShaderProgram(GLProgram):
    INPUT_LIGHT = b'inputLight'
    test_value = [1.0, 0.0, 1.0, 1.0]
    lights_vector = (1.0, 1.0, -1.0, 0.0)
    view_vector = (0.0, 0.0, -1.0, 0.0)

    LIGHT_PROPS = {
            'light_ambient': (0.2, 0.2, 0.2, 1.0),
            'light_diffuse': (0.8, 0.8, 0.8, 1.0),
            'light_specular': (0.7, 0.7, 0.7, 0.7),
            'light_position': lights_vector
    }
    
    shininess = 8

    def __init__(self):
        GLProgram.__init__(self, "gouraud_vs.glsl", "gouraud_fs.glsl")
        

    def __enter__(self):
        self._old_prog = glGetInteger(GL_CURRENT_PROGRAM)
        self.get_shader_program().use()
        for name, value in GouraudShaderProgram.LIGHT_PROPS.items():
            location = self.get_uniform_location(name)
            glUniform4f(location, *value)
        location = self.get_uniform_location('shininess')
        glUniform1f(location, self.shininess)
        location = self.get_uniform_location('view_vector')
        glUniform4f(location, *self.view_vector)

        glUniform1f(self.get_uniform_location('x_min'), float(self.xmin))
        glUniform1f(self.get_uniform_location('x_max'), float(self.xmax))
        glUniform1f(self.get_uniform_location('y_min'), float(self.ymin))
        glUniform1f(self.get_uniform_location('y_max'), float(self.ymax))
        glUniform1f(self.get_uniform_location('z_min'), float(self.zmin))
        glUniform1f(self.get_uniform_location('z_max'), float(self.zmax))
        glUniform1f(self.get_uniform_location('v_min'), float(self.vmin))
        glUniform1f(self.get_uniform_location('v_max'), float(self.vmax))

        #glUniformMatrix4fv(self.get_uniform_location('clip_rotation_matrix'), 1, GL_FALSE, self.v_matrix)
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND) #turn off alpha - as we don't sort triangles we don't want to do anything fancy here
        #glEnable(GL_CULL_FACE)
        #glCullFace(GL_BACK)
        
        glDisable(GL_POINT_SMOOTH)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA)
        #glEnable(GL_BLEND)

    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(self._old_prog)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_POINT_SMOOTH)
        
        
        
class GouraudSphereShaderProgram(GouraudShaderProgram):
    def __init__(self):
        GLProgram.__init__(self, "pointsprites_vs.glsl", "spheres_fs.glsl")

    def __enter__(self):
        self.get_shader_program().use()
        GouraudShaderProgram.__enter__(self)
    
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_PROGRAM_POINT_SIZE)

    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(0)
        glDisable(GL_POINT_SPRITE)
        glDisable(GL_PROGRAM_POINT_SIZE)
        

class OITGouraudShaderProgram(GouraudShaderProgram):
    def __init__(self):
        GLProgram.__init__(self, "gouraud_vs.glsl", "gouraud_oit_fs.glsl")
    
    def __enter__(self):
        self.get_shader_program().use()
        GouraudShaderProgram.__enter__(self)
        
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA)
        
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(0)
        glDisable(GL_DEPTH_TEST)
        


