from ._base import Plugin
import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    import neuroglancer
    from neuroglancer import local_volume
except ImportError:
    # this is to keep tests which automatically import all modules happy.
    logger.exception('Could not import neuroglancer, install if you need PYMEImage neuroglancer connectivity')
    # NB - will cause a subsequent error when the code tries to access the modules


class PyramidLocalVolume(local_volume.LocalVolume):
    """
    Use the correct sub-volume if we have a pyramid, rather than down-sampling on the fly
    """
    def get_encoded_subvolume(self, data_format, start, end, scale_key):
        rank = self.rank
        if len(start) != rank or len(end) != rank:
            raise ValueError('Invalid request')
        
        downsample_factor = np.array(scale_key.split(','), dtype=np.int64)
        if (len(downsample_factor) != rank or np.any(downsample_factor < 1)
            or np.any(downsample_factor > self.max_downsampling)
            or np.prod(downsample_factor) > self.max_downsampling):
            raise ValueError('Invalid downsampling factor.')
        
        downsampled_shape = np.cast[np.int64](np.ceil(self.shape / downsample_factor))
        if np.any(end < start) or np.any(start < 0) or np.any(end > downsampled_shape):
            raise ValueError('Out of bounds data request.')
        
        print('downsample_factor:', downsample_factor)
        
        pyramid_level = min(int(np.log2(downsample_factor.min())), len(self.data.levels))
        downsample_factor = (downsample_factor / (2**pyramid_level)).astype(np.int64)

        indexing_expr = tuple(np.s_[start[i] * downsample_factor[i]:end[i] * downsample_factor[i]]
                              for i in range(rank))
        
        subvol = np.array(self.data.levels[pyramid_level][indexing_expr].squeeze(), copy=False)
        if subvol.dtype == 'float64':
            subvol = np.cast[np.float32](subvol)

        if np.any(downsample_factor != 1):
            if self.volume_type == 'image':
                subvol = local_volume.downsample.downsample_with_averaging(subvol, downsample_factor)
            else:
                subvol = local_volume.downsample.downsample_with_striding(subvol, downsample_factor)
        
        content_type = 'application/octet-stream'
        if data_format == 'jpeg':
            data = local_volume.encode_jpeg(subvol)
            content_type = 'image/jpeg'
        elif data_format == 'npz':
            data = local_volume.encode_npz(subvol)
        elif data_format == 'raw':
            data = local_volume.encode_raw(subvol)
        else:
            raise ValueError('Invalid data format requested.')
        return data, content_type

class Neuroglancer(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        self._ng_viewer = None

        dsviewer.AddMenuItem('&3D', 'View in Neuroglancer', self.OnNeuroglancer)
    
    @property
    def ng_viewer(self):
        #import neuroglancer
        import webbrowser
        
        if self._ng_viewer is None:
            self._ng_viewer = neuroglancer.Viewer()
            webbrowser.open(str(self._ng_viewer), new=1)
            
        return self._ng_viewer
    
    def ToZarr(self):
        import zarr
        import dask.array
        ds = zarr.array(self.image.data)
    
    def OnNeuroglancer(self, event):
        import neuroglancer
        from PYME.IO.DataSources import BaseDataSource
        
        with self.ng_viewer.txn() as s:
            dims = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=list(self.image.voxelsize_nm))
            s.dimensions = dims
            
            nChans = self.image.data_xyztc.shape[4]
            
            if nChans == 1:
                cols = ['white']
            else:
                cols = ['cyan', 'magenta', 'yellow', 'red', 'green', 'blue']
            
            if hasattr(self.image.data_xyztc, 'levels'):
                LayerKlass = PyramidLocalVolume
            else:
                LayerKlass = neuroglancer.LocalVolume
            
            for i in range(nChans):
                #d = np.atleast_3d(self.image.data[:, :, :, i].squeeze())
                d = BaseDataSource.XYZSubvolume(self.image.data_xyztc, self.do.tp, i)
                
                d_min = self.do.Offs[i]
                d_max = d_min + 1.0/self.do.Gains[i]
                
                s.layers.append(name=self.image.names[i],
                                layer=LayerKlass(data=d, dimensions=dims, volume_type='image',
                                                 max_downsampling=8**5),
                                shader="""
#uicontrol invlerp normalized(range=[%3.3f, %3.3f], window=[%3.3f, %3.3f])
#uicontrol vec3 c color(default="%s")
void main() {
  vec3 v = c*normalized();
  emitRGB(v);
}
                                """ % (d_min, d_max,d_min, d_max, cols[i]))


def Plug(dsviewer):
    return Neuroglancer(dsviewer)