
from PYME.Acquire.ExecTools import joinBGInit, HWNotPresent, init_gui, init_hardware
import time



@init_hardware('Pointscan Camera')
def pointscan_cam(scope):
    from PYME.Acquire.Hardware.pointscan_shim import pointscan_camera
    import threading  
    import queue
    import numpy as np
    import time

    class TestScanner(pointscan_camera.BaseScanner):
        """ A (buffered) fake image generator for testing the camera shim.
        """
        n_channels = 3
        def __init__(self, scan_params=None, axes_order=('x', 'y')):
            super().__init__(scan_params, axes_order)
        
        @pointscan_camera.BaseScanner.frame_integ_time.getter
        def frame_integ_time(self):
            return 1
        
        def wait_for_finished_buffer(self, timeout):
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.n_full > 0:
                    with self.full_buffer_lock:
                        return self.full_buffers.get()
                time.sleep(0.05)
            raise TimeoutError('Timed out waiting for scanner buffer')
        
        def get_serial_number(self):
            return 'Number0'
        
        def stop(self):
            pass

        def _scan(self):
            time.sleep(self.frame_integ_time)
            for ind in range(self.n_channels):
                buf = self.free_buffers.get_nowait()
                buf[:] = ind * np.ones((self.height, self.width), dtype=self.dtype)
                with self.full_buffer_lock:
                    self.full_buffers.put(buf)
                    self.n_full += 1
        
        def scan(self):
            """raster-scan an image and add each channel as a separate frame to the full frame
            buffer

            """
            t = threading.Thread(target=self._scan)
            t.start()
        
        # def allocate_buffers(self, n_buffers):
        #     """queue up a number of single-frame buffers
        #     Note that each channel will be slotted into the queue as a separate frame (for now)
        #     Parameters
        #     ----------
        #     n_buffers : int
        #         number of single-frame (XY) buffers to allocate
            
        #     """
        #     self.free_buffers = queue.Queue()
        #     self.full_buffers = queue.Queue()
        #     self.n_full = 0
        #     for ind in range(n_buffers):
        #         self.free_buffers.put(np.zeros((self.width, self.height), 
        #                                     dtype=self.dtype))
        
        # def destroy_buffers(self):
        #     with self.full_buffer_lock:
        #         self.n_full = 0
        #         while not self.full_buffers.empty():
        #             try:
        #                 self.full_buffers.get_nowait()
        #             except queue.Empty:
        #                 pass


    cam = pointscan_camera.PointscanCameraShim(position_scanner=TestScanner(scan_params={
        'n_x': 10,  # [px]
        'n_y': 10,  # [px]
    }))

    scope.register_camera(cam, 'Test')

#must be here!!!
joinBGInit() #wait for anyhting which was being done in a separate thread


time.sleep(.5)
scope.initDone = True
