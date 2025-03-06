
import numpy as np
from PYME.Acquire.Hardware.pointscan_shim import pointscan_camera
import nidaqmx as ni
from nidaqmx.constants import TerminalConfiguration, Edge, FrequencyUnits, VoltageUnits
import threading

# drive_channels = {
#     'x': {
#         'channel': '/Dev1/ao0',
#         'min_val': -1,  # [V]
#         'max_val': 1,  # [V]
#         'min_position': -10,  # [um] The position min_val corresponds to
#         'max_position': 10,  # [um] The position max_val corresponds to
#         'units': VoltageUnits.VOLTS,
#     },
#     'y':{
#         'channel': '/Dev1/ao1',
#         'min_val': -1,
#         'max_val': 1,
#         'min_position': -5,  # [um] The position min_val corresponds to
#         'max_position': 5,  # [um] The position max_val corresponds to
#         'units': VoltageUnits.VOLTS,
#     }
# }

# signal_channels = {
#     'SX': {
#         'channel': '/Dev1/ai0',
#         'min_val': -1,
#         'max_val': 1,
#         'units': VoltageUnits.VOLTS,
#         'terminal_config': TerminalConfiguration.NRSE
#     },
# }

# counter_channel = 'Dev1/ctr0'

class NIDAQScanner(pointscan_camera.BaseScanner):
    dtype = float
    def __init__(self, drive_channels, signal_channels, 
                 counter_channel='Dev1/ctr0', kwargs=None):
        super().__init__(**kwargs)
        self._scanning_lock = threading.Lock()
        self._drive_channels = drive_channels
        self._signal_channels = signal_channels
        self._counter_channel = counter_channel

        self._duty_cycle = kwargs.pop('duty_cycle', 0.8)  # (0, 1]
        self._pixel_clock_rate = kwargs.pop('pixel_clock_rate', 1000)  # [Hz]

        self.map_drive_voltages_to_positions()
        self._x_c = self._drive_channels['x']['min_position'] + self._x_pos_range / 2
        self._y_c = self._drive_channels['y']['min_position'] + self._y_pos_range / 2

        # get device properties
        dev_name = self._counter_channel.lstrip('/').split('/')[0]
    
        self._model = ni.system.System.local().devices[dev_name].product_type
        self._serial = str(ni.system.System.local().devices[dev_name].serial_num)
    
    def map_drive_voltages_to_positions(self):
        self._x_pos_range = self._drive_channels['x']['max_position'] - self._drive_channels['x']['min_position']
        self._x_volt_range = self._drive_channels['x']['max_val'] - self._drive_channels['x']['min_val']
        self.x_um_per_volt = self._x_pos_range / self._x_volt_range
        
        self._y_pos_range = self._drive_channels['y']['max_position'] - self._drive_channels['y']['min_position']
        self._y_volt_range = self._drive_channels['y']['max_val'] - self._drive_channels['y']['min_val']
        self.y_um_per_volt = self._y_pos_range / self._y_volt_range

        self.x_volt_from_um = lambda x: (x - self._drive_channels['x']['min_position']) / self.x_um_per_volt + self._drive_channels['x']['min_val']
        self.y_volt_from_um = lambda y: (y - self._drive_channels['y']['min_position']) / self.y_um_per_volt + self._drive_channels['y']['min_val']
    
    @property
    def scan_center(self):
        return self._x_c, self._y_c
    
    @scan_center.setter
    def scan_center(self, center):
        self._x_c, self._y_c = center
    
    @property
    def n_channels(self):
        return len(self._signal_channels)

    def prepare(self):
        """
        map pixel positions to voltage values
        """
        x0, y0 = self.scan_center
        self._axes_voltages = {}
        # self.axis_voltages = {}
        # self.n_scan_positions = 1

        # generate the grid, in position space, then map to control voltages
        # FIXME - doesn't handle even sized grids
        half_x = self.width // 2
        x_pixelsize = self._scan_params['voxelsize.x']
        x_pos = np.linspace(-half_x, half_x, self.width, endpoint=True) * x_pixelsize + x0
        
        half_y = self.height // 2
        y_pixelsize = self._scan_params['voxelsize.y']
        y_pos = np.linspace(-half_y, half_y, self.height, endpoint=True) * y_pixelsize + y0

        if self.axes_order[0] == 'x':
            # make x scan faster
            self._axes_voltages['x'] = self.x_volt_from_um(np.repeat(x_pos, self.height))
            self._axes_voltages['y'] = self.y_volt_from_um(np.tile(y_pos, self.width))
        else:
            # make y fast axis
            self._axes_voltages['y'] = self.y_volt_from_um(np.repeat(y_pos, self.width))
            self._axes_voltages['x'] = self.x_volt_from_um(np.tile(x_pos, self.height))
        
        self.n_steps = len(self._axes_voltages['x'])

    def scan(self):
        with self._scanning_lock:
            self.prepare()
            self._scan()
    
    def _scan(self):
        with ni.Task() as ctr_task, ni.Task() as ai_task, ni.Task() as ao_task:
            # set up counter / pixel clock task
            ctr_task.co_channels.add_co_pulse_chan_freq(
                counter=self._counter_channel, 
                units=FrequencyUnits.HZ, freq=self._pixel_clock_rate, 
                duty_cycle=self._duty_cycle
            )
            ctr_task.timing.cfg_implicit_timing(
                sample_mode=ni.constants.AcquisitionType.FINITE,
                samps_per_chan=self.n_steps  # for finite mode this sets the number of pulses
            )

            # set up Analog Input task
            for k, v in self._signal_channels.items():
                ai_task.ai_channels.add_ai_voltage_chan(v['channel'],
                                                        terminal_config=v['terminal_config'],
                                                        min_val=v['min_val'], max_val=v['max_val'],
                                                        units=v['units'])
            # read signal on the falling edge of each pixel clock pulse
            ai_task.timing.cfg_samp_clk_timing(source=f"/{self._counter_channel}InternalOutput",
                                            rate=self._pixel_clock_rate, sample_mode=ni.constants.AcquisitionType.FINITE,
                                            samps_per_chan=self.n_steps,
                                            active_edge=Edge.FALLING)
            
            # set up Analog Output task, order of X,Y here regardless of scan speed order
            for v in [self._drive_channels['x'], self._drive_channels['y']]:
                ao_task.ao_channels.add_ao_voltage_chan(v['channel'],
                                                            min_val=v['min_val'], max_val=v['max_val'], 
                                                            units=v['units'])
            # move the stage on the rising edge of each pixel clock pulse
            ao_task.timing.cfg_samp_clk_timing(source=f"/{self._counter_channel}InternalOutput",
                                            rate=self._pixel_clock_rate, sample_mode=ni.constants.AcquisitionType.FINITE,
                                            samps_per_chan=self.n_steps,
                                            active_edge=Edge.RISING)
            
            
            # "start" AI task (it will actually start when the clock starts)
            ai_task.start()
            # "start" the AO task (it will actually start when the clock starts)
            ao_task.write(np.stack((self._axes_voltages['x'], 
                                    self._axes_voltages['y'])), auto_start=True)
            # start counter task (which should actually kick this whole thing off)
            ctr_task.start()
            # wait until we're done
            ctr_task.wait_until_done()
            # read data from AI task
            data = ai_task.read(number_of_samples_per_channel=self.n_steps)
            # print(data)
        
        # write the scan buffer to the full frame buffer
        for ind in range(self.n_channels):
            buf = self.free_buffers.get_nowait()
            # thought data should be (n_channels, n_steps), but comes as list of lists
            buf[:] = np.asarray(data[ind]).reshape(self.width, self.height)
            with self.full_buffer_lock:
                self.full_buffers.put(buf)
                self.n_full += 1
    
    def get_serial_number(self):
        return self._serial
    
    def stop(self):
        pass
