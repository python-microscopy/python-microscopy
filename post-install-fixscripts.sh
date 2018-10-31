PYMEVER=`python -c 'import PYME.version ; print PYME.version.version'`
PYPREF=`python -c 'import sys; print sys.prefix'`
for filename in dh5view VisGUI fitMonP PYMEAcquire dh5view.py VisGUI.py
do
    echo "patching $PYPREF/bin/$filename..."
    if egrep -q '^#!(.*)MacOS/python$' $PYPREF/bin/$filename; then
	echo '     this is a app/MacOS install - no pythonw needed'
    else
	echo '     patching with pythonw...'
	sed -i '' -E 's|^#!(.*)python$|#!/bin/bash \1\pythonw|' $PYPREF/bin/$filename
    fi
    sed -i '' -E "s/PYME==[0-9.]+/PYME==$PYMEVER/" $PYPREF/bin/$filename
done


#echo "patching fitmonp in PYME tree..."
#sed -i '' -E 's|^#!(.*)python$|#!/bin/bash \1\pythonw|' $HOME/Documents/src/python-microscopy-exeter/PYME/ParallelTasks/fitMonP.py
