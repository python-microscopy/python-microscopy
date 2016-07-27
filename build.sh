#!/bin/bash

$PYTHON setup.py install
#$PYTHON pymecompress/setup.py install

# Add more build steps here, if they are necessary.

echo "Trying to install OSX Launchers"
echo $OSX_ARCH
echo $RECIPE_DIR

if [ -n "$OSX_ARCH" ]
	then
		cd $RECIPE_DIR/osxLaunchers
		xcodebuild -alltargets

		cp -r ./build/Release/*.app $PREFIX
fi

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
