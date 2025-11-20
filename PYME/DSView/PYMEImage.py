"""
Entry point for PYMEImage application

Minimal stub to ensure that on macOS we have a framework build of python
"""

from PYME.util.uilaunch import ensure_macos_framework_build

def main():
    # make sure we have a framework build on macOS so that ui elements work properly
    # run this before importing any ui elements (or anything that might take time to import)
    ensure_macos_framework_build()

    from PYME.DSView import dsviewer
    dsviewer.main()

if __name__ == '__main__':
    main()
