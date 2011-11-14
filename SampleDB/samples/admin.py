from SampleDB.samples.models import *
from django.contrib import admin

from SampleDB.autocomplete.admin import AutocompleteAdmin
from SampleDB.samples import autocomp_settings

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

    def save_model(self, request, obj, form, change):
        obj.slideID = hashString32(obj.creator + obj.reference)
        obj.save()

admin.site.register(Slide, SlideAdmin)