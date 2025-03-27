
from .base import register_module, OutputModule
from .traits import Input, Output, CStr, DictStrAny, Bool, Float, Int
import requests
import json
import numpy as np
import time

def format_values(d, context):
    for k in d.keys():
            if isinstance(d[k], str):
                d[k] = d[k].format(**context)
            elif isinstance(d[k], dict):
                format_values(d[k], context)

@register_module('QueueAcquisitions')
class QueueAcquisitions(OutputModule):
    """
    Queue move-to and start-spool acquisition tasks for each input position
    using the ActionServer web wrapper queue_action endpoint.

    Parameters
    ----------
    input_positions : Input
        PYME.IO.tabular containing 'x_um' and 'y_um' coordinates in units of 
        micrometers (preferred) *or* 'x' and 'y' coordinates in units of 
        nanometers.
    action_server_url : CStr
        URL of the microscope-side action server process.
    spool_settings : DictStrAny
        settings to be passed to `PYME.Acquire.SpoolController.start_spooling` as
        key-word arguments. Ones that make sense in the context of this recipe
        module include:
            max_frames : int
                number of frames to spool per series
            method : str
                'File', 'Queue' (py2 only), or 'Cluster'. 
            hdf_compression_level: int
                pytables compression level, valid for file/queue methods only.
            z_stepped : bool
                flag to toggle z stepping or standard protocol during spool
            z_dwell : int
                number of frames to acquire at each z position for z-stepped 
                spools
            cluster_h5 : bool
                spool to single h5 file on cluster (True) or pzf files (False).
                Only relevant for `Cluster` method.
            pzf_compression_settings : dict
                see PYME.IO.acquisition_backends
            protocol_name : str
                filename of the acquisition protocol to follow while spooling
            subdirectory : str
                firectory within current SpoolController set directory to spool
                a given series. The directory will be created if it doesn't 
                already exist.
    lifo: Bool
        last-in first-out behavior (True) starts at the last position in 
        `input_positions`, False starts with the 0th. Useful in instances where
        e.g. you leave the microscope at the end of a detection scan and
        reversing the position order could mean less travel.
    optimize_path : Bool
        toggle whether acquisition tasks for positions are posted in an order
        which will minimize stage travel. Still respects the `lifo` parameter to
        pick the start.
    timeout : Float
        time in seconds after which the acquisition tasks associated with these
        positions will be ignored/unqueued from the action manager.
    nice: Int
        priority at which acquisition tasks should execute (default=10)
    max_duration : float
        A generous estimate, in seconds, of how long the task might take, after
        which the lasers will be automatically turned off and the action queue
        paused.
    between_post_throttle : Float
        Time in seconds to sleep between posts to avoid bombarding the 
        microscope-side server. Can be set to zero for ~no throttling.
    """
    input_positions = Input('input')
    action_server_url = CStr('http://127.0.0.1:9393')
    spool_settings = DictStrAny()
    lifo = Bool(True)
    optimize_path = Bool(True)
    timeout = Float(np.finfo(float).max)
    max_duration = Float(np.finfo(float).max)
    nice = Int(10)
    between_post_throttle = Float(0.01)

    def save(self, namespace, context={}):
        """
        Parameters
        ----------
        namespace : dict
            The recipe namespace
        context : dict
            Information about the source file to allow pattern substitution to 
            generate the output name. At least 'basedir' (which is the fully 
            resolved directory name in which the input file resides) and 
            'file_stub' (which is the filename without any extension) should be 
            resolved.
        
        Notes
        -----
        str spool_settings values can context-substitute templated parameters,
        e.g. spool_settings = {'subdirectory': '{file_stub}',
                               'extra_metadata: {'Samples.Well': '{file_stub}'}}
        """
        # substitute spool settings
        spool_settings = self.spool_settings.copy()
        format_values(spool_settings, context)
        
        try:  # get positions in units of micrometers
            positions = np.stack((namespace[self.input_positions]['x_um'], 
                                  namespace[self.input_positions]['y_um']), 
                                 axis=1) # (N, 2), [um]
        except KeyError:  # assume x and y are in nanometers
            positions = np.stack((namespace[self.input_positions]['x'], 
                                  namespace[self.input_positions]['y']), 
                                 axis=1) / 1e3  # (N, 2), [nm] -> [um]
        
        if self.optimize_path:
            from PYME.Analysis.points.traveling_salesperson import sort
            start = -1 if self.lifo else 0
            positions = sort.tsp_sort(positions, start)
        else:
            positions = positions[::-1, :] if self.lifo else positions
        
        dest = self.action_server_url + '/queue_actions'
        session = requests.Session()
        session.trust_env = False
        actions = list()
        for ri in range(positions.shape[0]):
            actions.append({
                'CentreROIOn':{
                    'x': positions[ri, 0], 'y': positions[ri, 1],
                    'then': {
                        'SpoolSeries' : spool_settings
                    }
                }
            })
        session.post(dest + '?timeout=%f&nice=%d&max_duration=%f' % (self.timeout,
                                                                        self.nice,
                                                                        self.max_duration),
                        data=json.dumps(actions),
                        headers={'Content-Type': 'application/json'})
        
        # queue a high-nice call to shut off all lasers when we're done
        # FIXME - there has to be a better way of handling this!
        # a) protocols should turn lasers off anyway
        # b) maybe have a a specific 'safe state' fallback in the action manager for when the queue is empty?
        session.post(dest + '?timeout=%f&nice=%d&max_duration=%f' % (self.timeout,
                                                                        20,
                                                                        self.max_duration),
                        data=json.dumps([{
                            'FunctionAction': {
                                'functionName': 'turnAllLasersOff', 'args': {}
                                }
                            }]),
                        headers={'Content-Type': 'application/json'})
