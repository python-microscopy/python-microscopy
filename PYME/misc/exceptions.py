"""
Custom, PYME-specific exceptions

"""


class UserError(Exception):
    """
    To be raised, e.g. when doing input validation if the user has entered an incorrect value, or performed an incorrect
    sequence of events. When using this exception class, the message should be easily readable by the user and should
    tell them specifically where the error occured and ideally how to fix it. UserErrors will generally be shown in a
    dialog or message box and will not show tracebacks. IE you can't rely on the traceback context when writing the error
    message.
    
    The main purpose of having these is to make user error handling agnostic of the user interface and to avoid a lot of
    special case wx.MessageBox calls.
    
    Subclassing and defining the help_url attribute to point to the relevant part of the documentation is encouraged.
    """
    
    help_url = None
    
    def __init__(self, message, help_url=None):
        if help_url is not None:
            self.help_url = help_url
            
        self.message=message
        
    def __str__(self):
        return str(self.message)