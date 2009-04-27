from distutils.core import setup
try:
	import py2exe

	import matplotlib

	setup(console=['taskWorkerME.py'], data_files=matplotlib.get_py2exe_datafiles())
except:
	pass
