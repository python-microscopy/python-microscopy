- get the driver for the freetronics leosticK at  http://www.freetronics.com/pages/installing-the-usb-driver-file-for-windows
		- here we need to edit the hardware IDs to make the installer pick up the drivers 
- Filter Wheel
	- need to install drivers for filter wheel -> check how to
- install Nikon Ti Driver; needed to allow unsigned driver. This seems to work:

		C:\Windows\system32>bcdedit -set loadoptions DDISABLE_INTEGRITY_CHECKS
		C:\Windows\system32>bcdedit -set TESTSIGNING ON
		C:\Windows\system32>bcdedit /set nointegritychecks ON

	- Needs to be done in a CMD window started as administrator, see also http://superuser.com/questions/124112/use-an-unsigned-driver-in-windows-7-x64. I also needed to go to 'update drivcer' and just follow the update prompt. That installed the driver for good.

	-	in addition needed to get new Pywin32 as ran into error "'module' object has no attribute 'VARIANT'". Got that from installer site http://sourceforge.net/projects/pywin32/files/pywin32/Build%20218/ and used pywin32-218.win-amd64-py2.7.exe. After that object loads ok.

- when using both Ixon and Zyla the Ixon DLL path must come first!! Otherwise we get an issue with the ixon dll being picked up from the zyla dir incorrectly!
	
Things to do:

- connect to sample database - what is required?
	- done: needed to install MySQL-python, got that from caosguest as MySQL-python-1.2.3.win-amd64-py2.7
- record new EM gain calibration for ixus; we also may need a size calibration; anything else?
  	 - actually avoided doing by just copying PYMEsettings.db to PYME/Acquire from old computer
- set up nightly backups
      - need to share Data drive and add user caosguest to machine
