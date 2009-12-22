from django import template

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
        print s

    return s
    