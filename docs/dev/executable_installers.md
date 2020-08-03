# Building executable installers

We use conda constructor [https://github.com/conda/constructor](https://github.com/conda/constructor) along with the configuration file located at 
`python-microscopy/construct.yaml` to build executable installers for Windows and OSX.

1) Make sure conda packages for `python-microscopy`, `pyme-depends` and `pymecompress` are up to date 
2) [optional] Create a clean conda environment (or just install constructor into the conda base environment)
3) `conda install constructor` **NOTE:** At one point it was neccesary to git clone the constructor source, checkout the "v3" branch
   and do a `python setup.py install` instead of the above. I think that the v3 branch has now been merged into main, but 
   it might be worth trying the source install if you run into issues.
4) Download the latest `conda.exe` for the platform you are building for from 
[https://repo.anaconda.com/pkgs/misc/conda-execs/](https://repo.anaconda.com/pkgs/misc/conda-execs/) see also 
[https://github.com/conda/constructor/issues/257](https://github.com/conda/constructor/issues/257)
5) Change to the `python-microscopy` directory
6) Edit `construct.yaml` to set the version to the latest conda package version
6) Run `constructor --conda-exe PATH_TO_CONDA.EXE .`

### Known problems / issues

- Generated installer on Windows displays a bunch of complaint dialogs saying that it's not working. These can safely
  be dismissed whilst still resulting in a functional install
- Setting the installer icon is not working
- Might be getting a framework error (or other dependency problem) on OSX (it's been a while since I checked)
- There was a problem with packages wanting to run post-install scripts on OSX (not sure if this has gone away)
