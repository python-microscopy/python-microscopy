- get the driver for the freetronics leosticK at  http://www.freetronics.com/pages/installing-the-usb-driver-file-for-windows
- Filter Wheel
	- need to install drivers for filter wheel -> check how to
- install Nikon Ti Driver; needed to allow unsigned driver. This seems to work
		C:\Windows\system32>bcdedit -set loadoptions DDISABLE_INTEGRITY_CHECKS
		C:\Windows\system32>bcdedit -set TESTSIGNING ON
	Needs to be done in a CMD window started as administrator, see also http://superuser.com/questions/124112/use-an-unsigned-driver-in-windows-7-x64!
	
