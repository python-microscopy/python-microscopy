import importlib.resources
import pathlib
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# import.resources should be used instead of using __path__ and similar package variables directly,
# as direct access breaks with pip/meson editable installs
# see also https://mesonbuild.com/meson-python/how-to-guides/editable-installs.html#data-files
# the code below should be fully backwards compatible with non-editable installs, older setup.py based installs
# and newer py3s

def package_files_matching(calling_package,patterns):
    modules = []
    if not isinstance(patterns,list):
        patterns = [patterns]
    logger.debug("Local modules for package %s" % calling_package)
    for file in importlib.resources.files(calling_package).iterdir():
        for pattern in patterns:
            if pathlib.PurePath(file.name).match(pattern):
                modules.append(file.stem)
                break # make sure each file is only added once

    logger.debug("Matching modules " + repr(modules))

    return modules
