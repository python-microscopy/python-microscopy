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
from samples.models import *
from django.contrib import admin

from autocomplete.admin import AutocompleteAdmin
from samples import autocomp_settings

#admin.site.register(Slide)
admin.site.register(Image)
admin.site.register(File)
admin.site.register(Labelling)
admin.site.register(ImageTag)
admin.site.register(FileTag)
admin.site.register(SlideTag)
admin.site.register(Species)
admin.site.register(Sample)
admin.site.register(Dye)

class LabellingInline(AutocompleteAdmin, admin.StackedInline):
    model = Labelling
    autocomplete_fields = {'structure': 'Labelling.structure'}
    #limit=10

class SlideAdmin(AutocompleteAdmin, admin.ModelAdmin):
    inlines = [LabellingInline]
    autocomplete_fields = {'creator': 'Slide.creator'}
    list_display = ('creator', 'reference', 'timestamp', 'labels')
    list_filter = ('creator', 'timestamp')

    def save_model(self, request, obj, form, change):
        obj.slideID = hashString32(obj.creator + obj.reference)
        obj.save()

admin.site.register(Slide, SlideAdmin)
