#!/bin/bash

$PYTHON setup.py install
#$PYTHON pymecompress/setup.py install

# Add more build steps here, if they are necessary.

echo "Trying to install OSX Launchers"
echo $OSX_ARCH
echo $RECIPE_DIR
echo $PREFIX

if [ -n "$OSX_ARCH" ]
	then
		cd $RECIPE_DIR/../../osxLaunchers
		xcodebuild -alltargets

		cp -r ./build/Release/*.app $PREFIX

		install_name_tool -change @rpath/libpython2.7.dylib @executable_path/../../../lib/libpython2.7.dylib $PREFIX/VisGUI.app/Contents/MacOS/VisGUI
		install_name_tool -change @rpath/libpython2.7.dylib @executable_path/../../../lib/libpython2.7.dylib $PREFIX/dh5view.app/Contents/MacOS/dh5view
fi

#echo "Attempting to build and install go components"
#
#curdir=`pwd`
#
#PYMEGOdir="$GOPATH/src/github.com/mrd0ll4r/pyme"
#if [ -d "$PYMEGOdir" ]
#    then
#        echo "installing pyme go components"
#
#        cd $PYMEGOdir/cmd/distributor
#        go install .
#
#        cd $PYMEGOdir/cmd/nodeserver
#        go install .
#
#        cd $curdir
#
#        cp $GOBIN/distributor $PREFIX/bin
#        cp $GOBIN/nodeserver $PREFIX/bin
#fi


# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
