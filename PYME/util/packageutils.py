import importlib.resources
import pathlib
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# the use of __path__ is not recommended any more, the package_files_matching funcs below aims for replacing the previous use
# TODO: similarly, the use of __file__ is discouraged, see specifically
#                  https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package/58941536#58941536
#       there are still plenty of uses of __file__ in the dist; the current use of __file__ is quite heterogneous,
#       and may need different replacements; cirrent uses should be replaced over time;
#       if common use cases lend itself to encapsulation the functionally could be collected here

# import.resources should be used instead of using __path__ and similar package variables directly,
# as direct access breaks with pip/meson editable installs
# see also https://mesonbuild.com/meson-python/how-to-guides/editable-installs.html#data-files
# the code below should be fully backwards compatible with non-editable installs, older setup.py based installs
# when using newer py3s (post 3.9 or so)

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
