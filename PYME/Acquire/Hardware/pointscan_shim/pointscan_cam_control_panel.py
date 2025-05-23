
import wx
from PYME.Acquire.Hardware.pco import pco_sdk_cam_control_panel


class ScanParamControl(wx.Panel):
    def __init__(self, parent, cam):
        wx.Panel.__init__(self, parent)
        self.cam = cam
        self.scanner = cam.scanner

        # Create controls
        self.width = wx.TextCtrl(self, -1, str(self.scanner.width), size=(50, -1))
        self.height = wx.TextCtrl(self, -1, str(self.scanner.height), size=(50, -1))
        self.voxelsize_x_nm = wx.TextCtrl(self, -1, str(self.scanner.voxelsize_x_nm), size=(50, -1))
        self.voxelsize_y_nm = wx.TextCtrl(self, -1, str(self.scanner.voxelsize_y_nm), size=(50, -1))
        self.voxel_integration_time = wx.TextCtrl(self, -1, str(self.scanner.voxel_integration_time), size=(50, -1))
        self.scanning_indicator = wx.StaticText(self, -1, "Scanning: " + str(self.scanner._scanning_lock.locked()))

        # Bind events
        self.width.Bind(wx.EVT_TEXT, self.on_width_change)
        self.height.Bind(wx.EVT_TEXT, self.on_height_change)
        self.voxelsize_x_nm.Bind(wx.EVT_TEXT, self.on_voxelsize_x_change)
        self.voxelsize_y_nm.Bind(wx.EVT_TEXT, self.on_voxelsize_y_change)
        self.voxel_integration_time.Bind(wx.EVT_TEXT, self.on_voxel_integration_time_change)

        # Main vertical sizer
        self.vsizer = wx.BoxSizer(wx.VERTICAL)

        # Width and height controls
        width_height_sizer = wx.BoxSizer(wx.HORIZONTAL)
        width_height_sizer.Add(wx.StaticText(self, -1, "Width: "), 0, wx.ALL, 2)
        width_height_sizer.Add(self.width, 0, wx.ALL, 2)
        width_height_sizer.Add(wx.StaticText(self, -1, "Height: "), 0, wx.ALL, 2)
        width_height_sizer.Add(self.height, 0, wx.ALL, 2)
        self.vsizer.Add(width_height_sizer, 0, wx.EXPAND | wx.ALL, 2)

        # Voxel size controls
        voxel_size_sizer = wx.BoxSizer(wx.HORIZONTAL)
        voxel_size_sizer.Add(wx.StaticText(self, -1, "Voxel size X [nm]: "), 0, wx.ALL, 2)
        voxel_size_sizer.Add(self.voxelsize_x_nm, 0, wx.ALL, 2)
        voxel_size_sizer.Add(wx.StaticText(self, -1, "Y [nm]: "), 0, wx.ALL, 2)
        voxel_size_sizer.Add(self.voxelsize_y_nm, 0, wx.ALL, 2)
        self.vsizer.Add(voxel_size_sizer, 0, wx.EXPAND | wx.ALL, 2)

        # Voxel integration time
        voxel_integ_time_sizer = wx.BoxSizer(wx.HORIZONTAL)
        voxel_integ_time_sizer.Add(wx.StaticText(self, -1, "Voxel integ. time [s]: "), 0, wx.ALL, 2)
        voxel_integ_time_sizer.Add(self.voxel_integration_time, 0, wx.ALL, 2)
        self.vsizer.Add(voxel_integ_time_sizer, 0, wx.EXPAND | wx.ALL, 2)

        # Scanning indicator
        scanning_sizer = wx.BoxSizer(wx.HORIZONTAL)
        scanning_sizer.Add(self.scanning_indicator, 0, wx.ALL, 2)
        self.vsizer.Add(scanning_sizer, 0, wx.EXPAND | wx.ALL, 2)

        # Set the main sizer
        self.SetSizerAndFit(self.vsizer)
    
    def on_width_change(self, event=None):
        self.scanner.width = int(self.width.GetValue())
    
    def on_height_change(self, event=None):
        self.scanner.height = int(self.height.GetValue())
    
    def on_voxelsize_x_change(self, event=None):
        self.scanner.voxelsize_x_nm = float(self.voxelsize_x_nm.GetValue())
    
    def on_voxelsize_y_change(self, event=None):
        self.scanner.voxelsize_y_nm = float(self.voxelsize_y_nm.GetValue())
    
    def on_voxel_integration_time_change(self, event=None):
        self.scanner.voxel_integration_time = float(self.voxel_integration_time.GetValue())
    
    def update(self):
        self.width.SetValue(str(self.scanner.width))
        self.height.SetValue(str(self.scanner.height))
        self.voxelsize_x_nm.SetValue(str(self.scanner.voxelsize_x_nm))
        self.voxelsize_y_nm.SetValue(str(self.scanner.voxelsize_y_nm))
        self.voxel_integration_time.SetValue(str(self.scanner.voxel_integration_time))
        # self.pixel_clock_rate.SetValue(str(self.scanner.pixel_clock_rate))
        self.scanning_indicator.SetLabel("Scanning: " + str(self.scanner._scanning_lock.locked()))
        self.scanning_indicator.SetForegroundColour(wx.RED if self.scanner._scanning_lock.locked() else wx.GREEN)
    


        


class PointScanCamControl(pco_sdk_cam_control_panel.PcoSdkCamControl):
    """
    Control panel for PointScan cameras.
    Inherits from PcoSdkCamControl, which defines a flexible panel to add
    controls to, starting with a acquisition mode control.
    """
    def __init__(self, parent, cam, scope):
        """
        Initialize the PointScanCamControl panel.
        
        Parameters
        ----------
        parent : wx.Panel
            The parent panel.
        cam : Camera
            The camera object.
        scope : Scope
            The scope object.
        """
        wx.Panel.__init__(self, parent)
        
        self.cam = cam
        self.scope = scope
        
        self.ctrls = [pco_sdk_cam_control_panel.ModeControl(self, cam),
                      ScanParamControl(self, cam)]
        
        self._init_ctrls()
        