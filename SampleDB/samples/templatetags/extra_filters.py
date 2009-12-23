from django import template
from django.utils.html import conditional_escape
from django.utils.safestring import mark_safe
from django.template.defaultfilters import stringfilter

register = template.Library()

@register.filter
def is_changed(value): # Only one argument.
    #is_changed.oldValue=None
    #is_changed.currentState=False

    #print value, is_changed.oldValue

    #while True:
    if not (value == is_changed.oldValue):
        is_changed.currentState = not is_changed.currentState

    is_changed.oldValue = value

    if is_changed.currentState:
        return 'odd'
    else:
        return 'even'

is_changed.oldValue=None
is_changed.currentState=False

@register.filter
def sum(value, arg): 
    s = 0
    #print value
    for v in value:
        s += v.__getattribute__(arg)
        #print s

    return s

@register.filter
@stringfilter
def tagify(value, autoescape=None):
    if autoescape:
        esc = conditional_escape
    else:
        esc = lambda x: x
        
    s = value
    if value.startswith('Protocol_'):
        s = '<b>P</b> %s' % esc(value.split('_')[1])

    if value.startswith('FitModule_'):
        s = '<b>F</b> %s' % esc(value.split('_')[1])

    return mark_safe(s)

tagify.needs_autoescape=True
    