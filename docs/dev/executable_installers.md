# Building executable installers

We use conda constructor [https://github.com/conda/constructor](https://github.com/conda/constructor) along with the configuration file located at 
`python-microscopy/construct.yaml` to build executable installers for Windows and OSX.

Instructions work for `miniconda3` with `conda --version` output of `conda 4.8.3`.

1) Create a clean conda environment (e.g. `conda create -n pyme_build python=2.7`)
2) Activate the new environment and run `conda install constructor`. This should install `constructor>=3.0.0`.
3) Change to the `python-microscopy` directory
4) Edit `construct.yaml` to set the version to the latest conda package version
5) Run `constructor`

### Known problems / issues

- There is a framework error on OSX, but this might be fixed by a new `conda-build` call with the latest version of `conda-build`.
