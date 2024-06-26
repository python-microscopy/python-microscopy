#!/usr/bin/python

# __init__.py
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

from .AxesOverlayLayer import AxesOverlayLayer
from .LUTOverlayLayer import LUTOverlayLayer
#from .Point3DRenderLayer import Point3DRenderLayer
#from .PointSpriteRenderLayer import PointSpritesRenderLayer
#from .QuadTreeRenderLayer import QuadTreeRenderLayer
#from .VertexRenderLayer import VertexRenderLayer
from .ScaleBarOverlayLayer import ScaleBarOverlayLayer
from .SelectionOverlayLayer import SelectionOverlayLayer
#from .ShadedPointRenderLayer import ShadedPointRenderLayer
#from .TetrahedraRenderLayer import TetrahedraRenderLayer
from .base import BaseLayer

from . import pointcloud, mesh, tracks, image_layer

layer_types = {
    'PointCloudRenderLayer' : pointcloud.PointCloudRenderLayer,
    'TriangleRenderLayer' : mesh.TriangleRenderLayer,
    'TrackRenderLayer' : tracks.TrackRenderLayer,
    'ImageRenderLayer' : image_layer.ImageRenderLayer,
}

def layer_from_session_info(pipeline, layer_info):
    '''Construct a layer from a dictionary of layer information. 
    
    Used for de-serialising PYMEVis sessions.
    '''
    layer_type = layer_info['type']
    layer = layer_types[layer_type](pipeline, **layer_info['settings'])
    return layer