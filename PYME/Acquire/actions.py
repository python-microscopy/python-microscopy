class Action(object):
    '''
    Base Action method - over-ride the __call__ function in derived classes
    '''
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def __call__(self, scope):
        pass
    
    def serialise(self):
        '''Convert to a .json serializable dictionary'''
        d = dict(self.params)
        
        then = getattr(self, '_then', None)
        if then:
            d['then'] = then.serialise()
        
        return {self.__class__.__name__: d}
    
    @property
    def _repr_then(self):
        if self._then is not None:
            return '- then -> %s' % repr(self._then)
        else:
            return ''
        
    def _estimated_duration(self, scope):
        '''Return the estimated duration of the action in seconds'''
        return 1.0
    
    def estimated_duration(self, scope):
        t =  self._estimated_duration(scope)

        then = getattr(self, '_then', None)
        if then:
            t += then.estimated_duration(scope)

        return t
    
    def finalise(self, scope):
        ''' Called after the action has executed'''
        # TODO - do then here??
        pass

        


class FunctionAction(Action):
    '''Legacy action which evals a string.

    Used for handling old -style actions
    '''
    
    def __init__(self, functionName, args=None):
        Action.__init__(self, functionName=functionName, args=args)
        self._fcn = functionName
        if args is None:
            args = {}
        self._args = args
    
    def __call__(self, scope):
        fcn = eval('.'.join(['scope', self._fcn]))
        #fcn = getattr(scope, self._fcn)
        return fcn(**self._args)
    
    def __repr__(self):
        return 'FunctionAction: %s(%s)' % (self._fcn, self._args) + self._repr_then


class StateAction(Action):
    ''' Base class for actions which modify scope state, with chaining support

    NOTE: we currently do not support chaining off the end of actions (e.g. spooling) which are likely to take some time.
    This is because functions such as start_spooling are non-blocking - they return a callback instead.
    '''
    
    def __init__(self, **kwargs):
        self._then = None
        Action.__init__(self, **kwargs)
    
    def then(self, task):
        self._then = task
        return self
    
    def _do_then(self, scope):
        if self._then is not None:
            return self._then(scope)


class UpdateState(StateAction):
    def __init__(self, state):
        self._state = state
        StateAction.__init__(self, state=state)
    
    def __call__(self, scope):
        scope.state.update(self._state)
        return self._do_then(scope)
    
    def __repr__(self):
        return 'UpdateState: %s' % self._state + self._repr_then

class MoveTo(StateAction):
    """
    Move to a specific position in absolute stage coordinates.

    Most useful when queueing actions to return to a specific
    already identified position.
    """
    def __init__(self, x, y):
        StateAction.__init__(self, x=x, y=y)
        self.x, self.y = x, y
    
    def __call__(self, scope):
        scope.SetPos(x=self.x, y=self.y)
        return self._do_then(scope)
    
    def __repr__(self):
        return 'MoveTo: %f, %f (x, y)' % (self.x, self.y) + self._repr_then

class CentreROIOn(StateAction):
    """
    Centre the ROI on a specific position in absolute stage coordinates.

    Most useful when queueing actions where target have been automatically
    identified.
    """
    def __init__(self, x, y):
        StateAction.__init__(self, x=x, y=y)
        self.x, self.y = x, y
    
    def __call__(self, scope):
        scope.centre_roi_on(self.x, self.y)
        return self._do_then(scope)
    
    def __repr__(self):
        return 'CentreROIOn: %f, %f (x, y)' % (self.x, self.y) + self._repr_then


class SpoolSeries(Action):
    def __init__(self, **kwargs):
        self._args = kwargs
        Action.__init__(self, **kwargs)
    
    def __call__(self, scope):
        return scope.spoolController.start_spooling(**self._args)
    
    def __repr__(self):
        return 'SpoolSeries(%s)' % ', '.join(['%s = %s' % (k,repr(v)) for k, v in self._args.items()])
    
    def _estimated_duration(self, scope):
        return scope.spoolController.estimate_spool_time(**self._args)
    
class RemoteSpoolSeries(Action):
    def __init__(self, remote_instance='remote_acquire_instance', **kwargs):
        self._remote_instance = remote_instance
        self._args = kwargs
        Action.__init__(self, **kwargs)
    
    def __call__(self, scope):
        # let the local spoolController handle the filename incrementing
        return getattr(scope,self._remote_instance).start_spooling(filename=scope.spoolController.seriesName, **self._args)
    
    def __repr__(self):
        return 'RemoteSpoolSeries(%s)' % ', '.join(['%s = %s' % (k,repr(v)) for k, v in self._args.items()])
    
    def _estimated_duration(self, scope):
        return 30*60 # 30 minutes
    
    def finalise(self, scope):
        super().finalise()
        scope.spoolController.SpoolStopped() #make sure file name increments

class SimultaeneousSpoolSeries(Action):
    '''Spool local and remote series simultaneously'''
    def __init__(self, remote_instance='remote_acquire_instance', local_settings = {}, remote_settings = {}, **kwargs):
        self._remote_instance = remote_instance
        self._local_settings = local_settings
        self._remote_settings = remote_settings
        self._args = kwargs
        Action.__init__(self, **kwargs)
    
    def __call__(self, scope):
        a =  scope.spoolController.start_spooling(settings = self._local_settings, **self._args)
        b =  getattr(scope,self._remote_instance).start_spooling(filename=scope.spoolController.seriesName + '_B', settings = self._remote_settings, **self._args)

        return lambda : a() and b()
    
    def __repr__(self):
        return 'SimultaeneousSpoolSeries(%s)' % ', '.join(['%s = %s' % (k,repr(v)) for k, v in self._args.items()])
    
    def _estimated_duration(self, scope):
        return scope.spoolController.estimate_spool_time(**self._args)
    


def action_from_dict(serialised):
    assert (len(serialised) == 1)
    act, params = list(serialised.items())[0]
    
    then = params.pop('then', None)
    # TODO - use a slightly less broad dictionary for action lookup (or move actions to a separate module)
    a = globals()[act](**params)
    
    if then:
        a.then(action_from_dict(then))
    
    return a
