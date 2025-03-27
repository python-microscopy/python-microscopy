"""Simple replacements for deprecated OpenGL projection and modelview functions
"""
import numpy as np

def ortho(left, right, bottom, top, near, far):
        """Create an orthographic projection matrix (replace glOrtho and remove glm dependency)"""

        m = np.eye(4)
        m[0, 0] = 2.0 / (right - left)
        m[0, 3] = -(right + left) / (right - left)
        m[1, 1] = 2.0 / (top - bottom)
        m[1, 3] = -(top + bottom) / (top - bottom)
        
        m[2, 2] = -2.0 / (far - near)

        m[2, 3] = -(far + near) / (far - near)

        return m

def frustrum(left, right, bottom, top, near, far):
    """Create a frustum projection matrix (replace glFrustum and remove glm dependency)"""

    m = np.zeros((4, 4))

    m[0, 0] = 2.0 * near / (right - left)
    m[1, 1] = 2.0 * near / (top - bottom)
    m[0, 2] = (right + left) / (right - left)
    m[1, 2] = (top + bottom) / (top - bottom)
    m[2, 2] = -(far + near) / (far - near)
    m[3, 2] = -1.0
    m[2, 3] = -2.0 * far * near / (far - near)

    return m


def translate_m(x, y, z):
    """Create a translation matrix (replace glTranslate and remove glm dependency)"""

    m = np.eye(4)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z

    return m

def translate(m, x, y, z):
    """Translate a matrix (replace glTranslate and remove glm dependency)"""

    return np.dot(m, translate_m(x, y, z))

def scale_m(x, y, z):
    """Create a scaling matrix (replace glScale and remove glm dependency)"""

    m = np.eye(4)
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z

    return m

def scale(m, x, y, z):
    """Scale a matrix (replace glScale and remove glm dependency)"""

    return np.dot(m, scale_m(x, y, z))

def mat3_to_mat4(m):
    """Convert a 3x3 matrix to a 4x4 matrix"""

    m4 = np.eye(4)
    m4[:3, :3] = m

    return m4


def vec3_to_vec4(v):
    """Convert a 3-element vector to a 4-element vector"""

    return np.array([v[0], v[1], v[2], 1.0])