import os
from PYME.LMVis.shader_programs.GLProgram import GLProgram
from OpenGL.GL import *
from PYME.LMVis.shader_programs.shader_program import ShaderProgram

class OITCompositorProgram(GLProgram):
    def __init__(self):
        GLProgram.__init__(self)
        shader_path = os.path.join(os.path.dirname(__file__), "shaders")
        shader_program = ShaderProgram(shader_path)
        shader_program.add_shader("compose_vs.glsl", GL_VERTEX_SHADER)
        shader_program.add_shader("compose_fs.glsl", GL_FRAGMENT_SHADER)
        shader_program.link()
        self.set_shader_program(shader_program)
    
    def __enter__(self):
        self._old_prog = glGetInteger(GL_CURRENT_PROGRAM)
        self.get_shader_program().use()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        glUseProgram(self._old_prog)