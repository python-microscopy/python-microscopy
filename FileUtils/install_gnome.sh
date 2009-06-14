#!/bin/bash

sudo cp pyme.xml /usr/share/mime/packages/
sudo update-mime-database /usr/share/mime

gconftool --install-schema-file=h5r.schema.xml

ln -s `pwd`/h5-thumbnailer.py ~/bin/h5-thumbnailer
ln -s `pwd`/h5r-thumbnailer.py ~/bin/h5r-thumbnailer
