#!/usr/bin/python
##################
# autocomp_settings.py
#
# Copyright David Baddeley, 2011
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
##################

#from django.contrib.auth.models import User
from django.shortcuts import render_to_response
from autocomplete.views import AutocompleteSettings, autocomplete
from samples.models import *
#from test_project.testapp.models import Dummy
#from test_project.testapp.views import autocomplete

from autocomplete.views import AutocompleteView

#autocomplete = AutocompleteView('samples')

class CreatorAutocomplete(AutocompleteSettings):
    queryset = Slide.objects.all()
    search_fields = ('^creator',)
    key='creator'

    def label(self, o):
        return unicode(o.creator)
    value = label

class StructureAutocomplete(AutocompleteSettings):
    queryset = Labelling.objects.all()
    search_fields = ('^structure',)
    key='structure'

    def label(self, o):
        return unicode(o.structure)
    value = label

#class CustomRenderingAutocomplete(SimpleAutocomplete):
#    key = 'first_name'
#    label = u'<em>%(email)s</em>'
#
#    def value(self, u):
#        return u.username.upper()
#
#class EmailAutocomplete(AutocompleteSettings):
#    queryset = User.objects.all()
#    search_fields = ('^email', '^username')
#    key = value = 'email'
#
#    def label(self, u):
#        return u'%s %s \u003C%s\u003E' % (u.first_name, u.last_name, u.email)

#autocomplete.register(Slide._meta.get_field('creator'), CreatorAutocomplete)
autocomplete.register('Slide.creator', CreatorAutocomplete)
autocomplete.register('Labelling.structure', StructureAutocomplete)
#autocomplete.register('testapp.loginreq', LoginRequiredAutocomplete)
#autocomplete.register('testapp.hasperm', HasPermAutocomplete)
#autocomplete.register('testapp.customrender', CustomRenderingAutocomplete)
#autocomplete.register(Dummy.user2, User2Autocomplete)
#autocomplete.register('testapp.limit', LimitAutocomplete)
#autocomplete.register(Dummy.friends, FriendsAutocomplete)
#autocomplete.register('testapp.email', EmailAutocomplete)


