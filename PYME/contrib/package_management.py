import requests
import json
import conda.cli.python_api as pyconda
import sys

class CondaPackageManager(object):
    def __init__(self, channels=('python-microscopy')):
        self.channels = list(channels)
        self.refresh()

    def refresh(self):
        available = []
        for chan in self.channels:
            available.extend(self.get_channel_info(chan))
        self._available = available
        self._installed = self.get_installed()

    @staticmethod
    def get_channel_info(channel):
        r = requests.get('https://api.anaconda.org/packages/%s' % channel)
        return json.loads(r.content)
    
    @staticmethod
    def install_package(package, channel=None):
        if channel is not None:
            (sout, serr, code) = pyconda.run_command(pyconda.Commands.INSTALL,
                                                     '-c', channel, package,
                                                     use_exception_handler=False,  # False will raise, True will handle - return error code 
                                                     stdout=sys.stdout, # if we leave out the next two we just get the strings, doesn't pritn to stdout/err
                                                     stderr=sys.stderr)
        else:
            (sout, serr, code) = pyconda.run_command(pyconda.Commands.INSTALL,
                                                     package,
                                                     use_exception_handler=False,  # False will raise, True will handle - return error code 
                                                     stdout=sys.stdout, # if we leave out the next two we just get the strings, doesn't pritn to stdout/err
                                                     stderr=sys.stderr)
        return sout, serr, code
    
    @staticmethod
    def get_installed():
        """Search all install conda packages

        Returns
        -------
        info : list
            dictionary for each installed package
                name : str
                    package name
                version : str
                    package version
                build : str
                    build info
                channel : str
                    conda channel package was sourced from (or local)
        """
        from io import StringIO
        sout = StringIO()
        so, se, code = pyconda.run_command(pyconda.Commands.LIST,
                                           use_exception_handler=False,  # False will raise, True will handle - return error code 
                                           stdout=sout)
        so.seek(0)
        [so.readline() for ind in range(3)]
        # Name                    Version                   Build  Channel
        raw_info = so.read()
        so.close()

        info = []
        for line in raw_info.split('\n'):
            if line == '':  # should just be the last line, but just to be safe
                continue
            package_info = line.split()

            # default/anaconda channel will be blank
            channel = None if len(package_info) == 3 else package_info[3]
            info.append(dict(name=package_info[0], version=package_info[1],
                             build=package_info[2], channel=channel))
        
        return info
