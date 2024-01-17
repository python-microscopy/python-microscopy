def patch_log_verbosity():
    '''
    We use logging.DEBUG as the logging level within PYME.
    This is a bit high for some of our dependencies, so set their levels individually.

    Should be called after the modules in question have been imported.
    '''
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR) #clobber unhelpful matplotlib debug messages
    logging.getLogger('matplotlib.backends.backend_wx').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)