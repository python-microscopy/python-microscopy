#!/usr/bin/python

###############
# admin.py
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
from django.db import models
from django.contrib import admin
from django.conf import settings
from django.core.urlresolvers import reverse

from autocomplete import widgets
from autocomplete.views import autocomplete as default_view
from autocomplete.utils import autocomplete_formfield


class AdminMedia:
    extend = False
    js = (settings.AUTOCOMPLETE_MEDIA_PREFIX + 'js/jquery_autocomplete.js',)
    css = {'all':
        (settings.AUTOCOMPLETE_MEDIA_PREFIX + 'css/jquery-ui.css',)
    }

class AdminAutocompleteWidget(widgets.AutocompleteWidget):
    Media = AdminMedia

class AdminMultipleAutocompleteWidget(widgets.MultipleAutocompleteWidget):
    Media = AdminMedia


class AutocompleteAdmin(object):
    autocomplete_autoconfigure = True
    autocomplete_view = default_view
    autocomplete_fields = {}


    def autocomplete_formfield(self, ac_id, formfield=None, **kwargs):
        return autocomplete_formfield(ac_id, formfield, self.autocomplete_view,
                AdminAutocompleteWidget, AdminMultipleAutocompleteWidget,
                **kwargs)

    def formfield_for_dbfield(self, db_field, **kwargs):
        #print db_field, self.autocomplete_view.settings
        if db_field.name in self.autocomplete_fields:
            ac_id = self.autocomplete_fields[db_field.name]
            return self.autocomplete_formfield(ac_id, db_field.formfield, **kwargs)
        elif self.autocomplete_autoconfigure:
            if db_field in self.autocomplete_view.settings:
                return self.autocomplete_formfield(db_field, **kwargs)
        return super(AutocompleteAdmin, self).formfield_for_dbfield(db_field, **kwargs)

    def _media(self):
        # little hack to include autocomplete's js before jquery.init.js
        media = super(AutocompleteAdmin, self).media
        media._js.insert(3, settings.AUTOCOMPLETE_MEDIA_PREFIX + 'js/jquery-ui.min.js')
        return media
    media = property(_media)

    def _autocomplete_view(self, request, field):
        info = self.model._meta.app_label, self.model._meta.module_name, field

        if field in self.autocomplete_fields:
            ac_id = self.autocomplete_fields[field]
        else:
            ac_id = '/'.join(info)
        return self.autocomplete_view(request, ac_id)

    def get_urls(self):
        # This ensures that `admin_site.admin_view` is applied to the
        # autocomplete_view.
        from django.conf.urls import patterns, url

        info = self.model._meta.app_label, self.model._meta.module_name

        urlpatterns = super(AutocompleteAdmin, self).get_urls()
        urlpatterns += patterns('',
            url(r'^autocomplete/(?P<field>[\w]+)/$',
                self.admin_site.admin_view(self._autocomplete_view),
                name='%s_%s_autocomplete' % info)
        )
        return urlpatterns

    def urls(self):
        return self.get_urls()
    urls = property(urls)

    @classmethod
    def _validate(cls):
        pass
