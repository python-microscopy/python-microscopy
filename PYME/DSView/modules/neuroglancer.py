from ._base import Plugin
import numpy as np

class Neuroglancer(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        
        self._ng_viewer = None

        dsviewer.AddMenuItem('&3D', 'View in Neuroglancer', self.OnNeuroglancer)
    
    @property
    def ng_viewer(self):
        import neuroglancer
        import webbrowser
        
        if self._ng_viewer is None:
            self._ng_viewer = neuroglancer.Viewer()
            webbrowser.open(str(self._ng_viewer), new=1)
            
        return self._ng_viewer
    
    def OnNeuroglancer(self, event):
        import neuroglancer
        with self.ng_viewer.txn() as s:
            dims = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=list(self.image.voxelsize_nm))
            s.dimensions = dims
            
            nChans = self.image.data.shape[3]
            
            if nChans == 1:
                cols = ['white']
            else:
                cols = ['cyan', 'magenta', 'yellow', 'red', 'green', 'blue']
            
            for i in range(self.image.data.shape[3]):
                d = np.atleast_3d(self.image.data[:, :, :, i].squeeze())
                s.layers.append(name=self.image.names[i],
                                layer=neuroglancer.LocalVolume(data=d, dimensions=dims, volume_type='image'),
                                shader="""
#uicontrol invlerp normalized(range=[%3.3f, %3.3f], window=[%3.3f, %3.3f])
#uicontrol vec3 c color(default="%s")
void main() {
  vec3 v = c*normalized();
  emitRGB(v);
}
                                """ % (d.min(), d.max(),d.min(), d.max(), cols[i]))


def Plug(dsviewer):
    return Neuroglancer(dsviewer)