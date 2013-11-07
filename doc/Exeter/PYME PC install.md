## Installing PYME and related software ##
- register required hostnames (e.g. lmenas1) via Windows host file, see http://helpdeskgeek.com/windows-7/windows-7-hosts-file/

- install EPD using the latest EPD installer, currently: epd-7.3-2-win-x86_64.msi

- update EPD:

  	     enpkg --userpass # (user: c.soeller@exeter.ac.uk, pwd: usual)

	     enpkg enstaller # updates enpkg
	     enpkg --update-all  # updates all packages

- install python-microscopy-exeter from hg repository, needs some preparation:

    - install emacs from gnu.org mirrors (note that this is easy)

    - install tortoisehg

    - generate ssh key with puttykeygen (from http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html) + use pageant (supplied with tortoisehg) + use kitty.exe to install key in phy-lmsrv1, see also:
https://confluence.atlassian.com/display/BITBUCKET/Set+up+SSH+for+Mercurial

	- on server run:	 emacs authorized_keys2
	and paste public key into file (note I tend to use the dsa key)

    - clone repository using tortoisehg, source: ssh://csoelle@phy-lmsrv1.ex.ac.uk/src/python-microscopy-exeter

    - now run 'remove_old_wx.py'

    - install proper wx, should be 2.8.12.x, currently: wxPython2.8-win64-unicode-2.8.12.1-py27.exe

    - goto CMD shell, cd into python-microscopy-exeter and execute
       python setup.py develop

- get 7-zip to unpack some tar.gz's from http://www.7-zip.org/

- additional Python modules to get

        easy_install Pyro

        easy install pip
        # pip is a replacement for easy_install that supports uninstall

    - install django, note we need 1.2.x version for compatibility (this bit is for sample database compatibility)
    - may also require the 'south' module
    - install mysql-python, used MySQL-python-1.2.3.win-amd64-py2.7.exe
    - get FFTW 3 libs from http://www.fftw.org/install/windows.html and unpack into suitable directory

    - get PyFFTW3  from https://launchpad.net/pyfftw/

    - build it by unpacking the tar.gz, cd into dir and
  	set to dir where you unpacked the FFTW3 dlls, e.g. set FFTW_PATH=C:\python-support-files\fftw-3.3.3-dll64
	python setup.py install

- final steps to set default open for h5s etc and include PYMEnf

    - set defaults for .h5, .h5r and .psf (using dh5view.cmd and Visgui.cmd)

    - expand PYMEnf into a suitable directory
  	- run 'python setup.py build_ext -i' in the DriftCorrection subdirectory
	     	     (note that setup.py install does not work!)

	- add the directory in which PYMEnf (i.e. the dir *above* PYMEnf) is located to the PYTHONPATH, e.g. 'C:\python-support-files'



-------------------------------------------------------------------------
### History items ###
- merged changes from default branch into exeter;

- todo, run 'Merge with Local' in tortoiseHg workbench from latest revision of merged branch (normally default); note that I had to edit .hgrc to change default path to google-code repo:

        [paths]
        default = https://code.google.com/p/python-microscopy/
        # default = ssh://csoelle@phy-lmsrv1.ex.ac.uk/src/python-microscopy-exeter

- this can be done from within workbench 'Settings' dialog and using 'Edit File' button

- needed to run cython on 'illuminate.pyx'

- also tried the conflict resolver that came up automatically; need to edit the default editor for hgworkbench
