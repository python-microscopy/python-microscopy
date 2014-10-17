#!/usr/bin/python

###############
# extra_filters.py
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

    