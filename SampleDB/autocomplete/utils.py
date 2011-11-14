from django import forms
from django.forms.models import modelform_factory
from django.db import models

from autocomplete import widgets
from autocomplete.views import autocomplete as default_view


def autocomplete_formfield(ac_id, formfield=None, view=default_view,
        widget_class=None, multiple_widget_class=None, **kwargs):
    """
    >>> # uses the default formfield (CharField)
    >>> autocomplete_formfield('myapp.email')

    >>> # uses a custom formfield
    >>> autocomplete_formfield('myapp.email', forms.EmailField)

    >>> # uses ForeignKey.formfield
    >>> autocomplete_formfield(Message.user)

    >>> # uses ManyToManyField.formfield
    >>> autocomplete_formfield(User.permissions)

    Use a custom view::
        >>> autocomplete_formfield('myapp.ac_id', view=my_autocomplete_view)
        
    """
    kwargs.pop('request', None) # request can be passed by contrib.admin
    db = kwargs.get('using')
    settings = view.get_settings(ac_id)
    if widget_class is None:
        widget_class = widgets.AutocompleteWidget
    if multiple_widget_class is None:
        multiple_widget_class = widgets.MultipleAutocompleteWidget

    if formfield is None:
        formfield = getattr(settings.field, 'formfield', forms.CharField)

    if isinstance(settings.field, models.ForeignKey):
        kwargs['widget'] = widget_class(ac_id, view, using=db)
        kwargs['to_field_name'] = settings.key
    elif isinstance(settings.field, models.ManyToManyField):
        kwargs['widget'] = multiple_widget_class(ac_id, view, using=db)
        kwargs['to_field_name'] = settings.key
    else:
        js_options = dict(force_selection=False)
        js_options.update(settings.js_options)
        kwargs['widget'] = widget_class(ac_id, view, using=db, **js_options)

    return formfield(**kwargs)


def _formfield_callback(autocomplete_fields, **ac_options):
    view = ac_options.get('view', default_view)

    def autocomplete_callback(field, **kwargs):
        acargs = dict(ac_options, **kwargs)
        if field.name in autocomplete_fields:
            ac_id = autocomplete_fields[field.name]
            return autocomplete_formfield(ac_id, field.formfield, **acargs)
        elif view.has_settings(field):
            return autocomplete_formfield(field, **acargs)
        return field.formfield(**kwargs)
    return autocomplete_callback


def autocompleteform_factory(model, autocomplete_fields={},
        form=forms.ModelForm, fields=None, exclude=None, **ac_options):
    """
    autocompleteform_factory(MyModel, {'friends':'myapp.friends'})
    """
    # XXX maybe we should make formfield_callback customizable
    callback = _formfield_callback(autocomplete_fields, **ac_options)
    return modelform_factory(model, form, fields, exclude, callback)
