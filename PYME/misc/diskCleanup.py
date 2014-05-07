#!/usr/bin/python

###############
# diskCleanup.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################
#!/usr/bin/python
import sys
import os

micrDataDir = '/smb/MicrData/'
lDataDir = len(micrDataDir)
nasDir = '/smb/NasData/'
#backupDir = '/media/disk/LMNAS1/'

def deleteFiles(directory):
    #dir_size = 0
    for (path, dirs, files) in os.walk(directory):
        for file in files:
            filename = os.path.join(path, file)
            #print filename

            nFilename = nasDir + filename[lDataDir:]
            #bFilename = backupDir + filename[lDataDir:]

            #print nFilename, bFilename

            if os.path.exists(nFilename) and os.path.getsize(nFilename) == os.path.getsize(filename): #and os.path.exists(bFilename) and os.path.getsize(filename) == os.path.getsize(nFilename):
                print(('Deleting %s' % filename))
                try:
                    os.remove(filename)
                except OSError:
                    import traceback
                    traceback.print_exc()
                    
            else:
                print(('Keeping %s' % filename))

        for dir in dirs:
            dirname = os.path.join(path, dir)

            if len(os.listdir(dirname)) == 0 and not 'System Volume Information' in dirname:
                print(('Removing %s' % dirname))
                os.rmdir(dirname)


if __name__ == '__main__':
    deleteFiles(sys.argv[1])
