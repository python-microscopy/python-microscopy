#!/usr/bin/python

###############
# __init__.py
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
__version__ = '1.0'

def get_installed_version():
    import pkg_resources
    try:
        autocomplete = pkg_resources.get_distribution('django-autocomplete')
    except pkg_resources.DistributionNotFound:
        return __version__
    return autocomplete.version


def get_mercurial_version():
    import os
    path = os.path.join(__path__[0], os.pardir)
    try:
        from mercurial.hg import repository
        from mercurial.ui import ui
        from mercurial import node, error

        repo = repository(ui(), path)
    except:
        return None
    tip = repo.changelog.tip()
    rev = repo.changelog.rev(tip)
    return '%s.dev%d' % (__version__, rev)


def get_version():
    return get_mercurial_version() or get_installed_version()

