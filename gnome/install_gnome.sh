#!/bin/bash

sudo cp pyme.xml /usr/share/mime/packages/
sudo update-mime-database /usr/share/mime

gconftool --install-schema-file=h5r.schema.xml

sudo xdg-desktop-menu install pyme-pyme.directory *.desktop

gconftool-2 -t string -s /desktop/gnome/url-handlers/pyme/command "urlOpener.py %s"
gconftool-2 -s /desktop/gnome/url-handlers/pyme/needs_terminal false -t bool
gconftool-2 -t bool -s /desktop/gnome/url-handlers/pyme/enabled true

#ln -s `pwd`/urlOpener.py ~/bin/pymeUrlOpener
#ln -s `pwd`/../DSView/dh5view.py ~/bin/dh5view
#ln -s `pwd`/../Analysis/LMVis/VisGUI.py ~/bin/VisGUI
#ln -s `pwd`/h5-thumbnailer.py ~/bin/h5-thumbnailer
#ln -s `pwd`/h5r-thumbnailer.py ~/bin/h5r-thumbnailer
#ln -s `pwd`/kdf-thumbnailer.py ~/bin/kdf-thumbnailer
#ln -s `pwd`/sf-thumbnailer.py ~/bin/sf-thumbnailer
