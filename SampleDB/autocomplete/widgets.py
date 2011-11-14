from django import forms
from django.forms.util import flatatt
from django.utils import simplejson
from django.utils.safestring import mark_safe
from django.conf import settings

from autocomplete.views import autocomplete as default_view


class AutocompleteWidget(forms.Widget):

    js_options = {
        'source': None,
        'multiple': False,
        'force_selection': True,
    }
    
    class Media:
        js = tuple(settings.AUTOCOMPLETE_MEDIA_PREFIX + js for js in (
            'js/jquery.min.js',
            'js/jquery-ui.min.js',
            'js/jquery_autocomplete.js',
        ))
        css = {'all':
            (settings.AUTOCOMPLETE_MEDIA_PREFIX + 'css/jquery-ui.css',)
        }

    def __init__(self, ac_id, view=default_view, attrs=None, using=None, **js_options):
        self.settings = view.get_settings(ac_id)
        self.db = using
        self.js_options = self.js_options.copy()
        self.js_options.update(self.settings.js_options)
        self.js_options.update(js_options)
        super(AutocompleteWidget, self).__init__(attrs)

    def render(self, name, value, attrs=None, hattrs=None, initial_objects=u''):
        if value is None:
            value = ''
        hidden_id = 'id_hidden_%s' % name
        hidden_attrs = self.build_attrs(type='hidden', name=name, value=value, id=hidden_id)
        normal_attrs = self.build_attrs(attrs, type='text')
        if value:
            if self.settings.reverse_label:
                normal_attrs['value'] = self.label_for_value(value)
            else:
                normal_attrs['value'] = value
        if not self.js_options.get('source'):
            self.js_options['source'] = self.settings.get_absolute_url()
        options = simplejson.dumps(self.js_options)
        return mark_safe(u''.join((
            u'<input%s />\n' % flatatt(hidden_attrs),
            u'<input%s />\n' % flatatt(normal_attrs),
            initial_objects,
            u'<script type="text/javascript">',
            u'django.autocomplete("#id_%s", %s);' % (name, options),
            u'</script>\n',
        )))

    def label_for_value(self, value):
        # XXX MultipleObjectsReturned could be raised if the field is not unique.
        settings = self.settings
        try:
            obj = settings.queryset.get(**{settings.key: value})
            return settings.value(obj)
        except settings.model.DoesNotExist:
            return value


class MultipleAutocompleteWidget(AutocompleteWidget):

    def __init__(self, ac_id, view=default_view, attrs=None, using=None, **js_options):
        js_options['multiple'] = True
        super(MultipleAutocompleteWidget, self).__init__(ac_id, view, attrs,
            using, **js_options)

    def render(self, name, value, attrs=None, hattrs=None):
        if value:
            initial_objects = self.initial_objects(value)
            value = ','.join([str(v) for v in value])
        else:
            value = None
            initial_objects = u''
        return super(MultipleAutocompleteWidget, self).render(
            name, value, attrs, hattrs, initial_objects)

    def label_for_value(self, value):
        return ''

    def value_from_datadict(self, data, files, name):
        value = data.get(name)
        if value:
            return value.split(',')
        return value
    
    def initial_objects(self, value):
        settings = self.settings
        output = [u'<ul class="ui-autocomplete-values">']
        for obj in settings.queryset.filter(**{'%s__in' % settings.key: value}):
            output.append(u'<li>%s</li>' % settings.label(obj))
        output.append(u'</ul>\n')
        return mark_safe(u'\n'.join(output))
