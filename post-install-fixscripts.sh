PYMEVER=`python -c 'import PYME.version ; print PYME.version.version'`
for filename in dh5view VisGUI VisGUI.py PYMEAcquire.py dh5view.py fitmonP.py launchWorkers.py taskServerMP.py taskWorkerMP.py
do
    echo "patching $filename..."
    sed -i '' -E 's|^#!(.*)python$|#!/bin/bash \1\pythonw|' $HOME/anaconda/bin/$filename
    sed -i '' -E "s/PYME==[0-9.]+/PYME==$PYMEVER/" $HOME/anaconda/bin/$filename
done


echo "patching fitmonp in PYME tree..."
sed -i '' -E 's|^#!(.*)python$|#!/bin/bash \1\pythonw|' $HOME/Documents/src/python-microscopy-exeter/PYME/ParallelTasks/fitMonP.py
