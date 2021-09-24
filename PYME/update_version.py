#!/usr/bin/python

###############
# update_version.py
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


from datetime import datetime
import os
import subprocess
import urllib
import json

def hook(ui, repo, **kwargs):
    update_version()
    return 0

def update_version_hg():
    now = datetime.now()
    
    p = subprocess.Popen('hg id -i', shell=True, stdout = subprocess.PIPE)
    id = p.stdout.readline().strip().decode()
    
    f = open(os.path.join(os.path.split(__file__)[0], 'version.py'), 'w')
    
    f.write('#PYME uses date based versions (yy.m.d)\n')    
    f.write("version = '%d.%02d.%02d'\n\n" % (now.year - 2000, now.month, now.day))
    f.write('#Mercurial changeset id\n')
    f.write("changeset = '%s'\n" % id)
    f.close()


def update_version():
    now = datetime.utcnow()
    
    p = subprocess.Popen('git describe --abbrev=12 --always --dirty=+', shell=True, stdout=subprocess.PIPE, encoding='utf8')
    id = p.stdout.readline().strip()
    
    f = open(os.path.join(os.path.split(__file__)[0], 'version.py'), 'w')

    new_version = '%d.%02d.%02d' % (now.year - 2000, now.month, now.day)

    # check to see if there is already a release / tag with this version number,
    # if there is, this is a post-release, append post<n>
    p = subprocess.Popen('git tag', shell=True, stdout=subprocess.PIPE, encoding='utf8')
    git_tags = [l.strip() for l in p.stdout.readlines()]

    post_count=0
    nv = new_version
    while nv in git_tags:
        nv = '%s.post%d' %(new_version, post_count)
        post_count += 1

    new_version = nv
    
    f.write('#PYME uses date based versions (yy.m.d)\n')
    f.write("version = '%s'\n\n" % new_version)
    f.write('#Git changeset id\n')
    f.write("changeset = '%s'\n" % id)
    f.close()
    
    print('PYMEVERSION=%s' % new_version)
    
if __name__ == '__main__':
    update_version()