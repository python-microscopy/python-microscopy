#!/usr/bin/python

###############
# colorsys.py
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
"""Conversion functions between RGB and other color systems.

This modules provides two functions for each color system ABC:

  rgb_to_abc(r, g, b) --> a, b, c
  abc_to_rgb(a, b, c) --> r, g, b

All inputs and outputs are triples of floats in the range [0.0...1.0]
(with the exception of I and Q, which covers a slightly larger range).
Inputs outside the valid range may cause exceptions or invalid outputs.

Supported color systems:
RGB: Red, Green, Blue components
YIQ: Luminance, Chrominance (used by composite video signals)
HLS: Hue, Luminance, Saturation
HSV: Hue, Saturation, Value

hacked from the original colorsys module to work with numpy arrays
"""

import numpy as np

# References:
# http://en.wikipedia.org/wiki/YIQ
# http://en.wikipedia.org/wiki/HLS_color_space
# http://en.wikipedia.org/wiki/HSV_color_space

__all__ = ["rgb_to_yiq","yiq_to_rgb","rgb_to_hls","hls_to_rgb",
           "rgb_to_hsv","hsv_to_rgb"]

# Some floating point constants

ONE_THIRD = 1.0/3.0
ONE_SIXTH = 1.0/6.0
TWO_THIRD = 2.0/3.0

# YIQ: used by composite video signals (linear combinations of RGB)
# Y: perceived grey level (0.0 == black, 1.0 == white)
# I, Q: color components

def rgb_to_yiq(r, g, b):
    y = 0.30*r + 0.59*g + 0.11*b
    i = 0.60*r - 0.28*g - 0.32*b
    q = 0.21*r - 0.52*g + 0.31*b
    return (y, i, q)

def yiq_to_rgb(y, i, q):
    r = y + 0.948262*i + 0.624013*q
    g = y - 0.276066*i - 0.639810*q
    b = y - 1.105450*i + 1.729860*q
    #if r < 0.0: r = 0.0
    r = r*(r > 0.0)
    #if g < 0.0: g = 0.0
    g = g*(g > 0.)
    #if b < 0.0: b = 0.0
    b = b*(b < 0.)
    #if r > 1.0: r = 1.0
    r = np.minimum(r, 1.0)
    #if g > 1.0: g = 1.0
    g = np.minimum(g, 1.0)
    #if b > 1.0: b = 1.0
    b = np.minimum(b, 1.0)
    return (r, g, b)


# HLS: Hue, Luminance, Saturation
# H: position in the spectrum
# L: color lightness
# S: color saturation

def rgb_to_hls(r, g, b):
    maxc = np.max([r, g, b],0)
    minc = np.min([r, g, b],0)
    # XXX Can optimize (maxc+minc) and (maxc-minc)
    l = (minc+maxc)/2.0
    if minc == maxc: return 0.0, l, 0.0
    if l <= 0.5: s = (maxc-minc) / (maxc+minc)
    else: s = (maxc-minc) / (2.0-maxc-minc)
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc: h = bc-gc
    elif g == maxc: h = 2.0+rc-bc
    else: h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, l, s

def hls_to_rgb(h, l, s):
    if s == 0.0: return l, l, l
    if l <= 0.5: m2 = l * (1.0+s)
    else: m2 = l+s-(l*s)
    m1 = 2.0*l - m2
    return (_v(m1, m2, h+ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h-ONE_THIRD))

def _v(m1, m2, hue):
    hue = hue % 1.0
    if hue < ONE_SIXTH: return m1 + (m2-m1)*hue*6.0
    if hue < 0.5: return m2
    if hue < TWO_THIRD: return m1 + (m2-m1)*(TWO_THIRD-hue)*6.0
    return m1


# HSV: Hue, Saturation, Value
# H: position in the spectrum
# S: color saturation ("purity")
# V: color brightness

def rgb_to_hsv(r, g, b):
    maxc = np.max([r, g, b],0)
    minc = np.min([r, g, b],0)

    print((maxc.shape))

    v = maxc
    h = np.zeros(v.shape)
    s = np.zeros(v.shape)
    #if minc == maxc: return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    #if r == maxc: h = bc-gc
    #elif g == maxc: h = 2.0+rc-bc

    h = 4.0+gc-rc
    h[g == maxc] = (2.0 + rc - bc)[g == maxc]
    h[r == maxc] = (bc-gc)[r == maxc]


    h = (h/6.0) % 1.0

    h[minc == maxc] = 0.0
    s[minc == maxc] = 0.0
    return h, s, v

def hsv_to_rgb(h, s, v):
    #if s == 0.0: return v, v, v
    i = np.floor(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6

    r = np.zeros(h.shape)
    g = np.zeros(h.shape)
    b = np.zeros(h.shape)

    #if i == 0: return v, t, p
    r[i==0] = v[i==0]
    g[i==0] = t[i==0]
    b[i==0] = p[i==0]
    #if i == 1: return q, v, p
    r[i==1] = q[i==1]
    g[i==1] = v[i==1]
    b[i==1] = p[i==1]
    #if i == 2: return p, v, t
    r[i==2] = p[i==2]
    g[i==2] = v[i==2]
    b[i==2] = t[i==2]
    #if i == 3: return p, q, v
    r[i==3] = p[i==3]
    g[i==3] = q[i==3]
    b[i==3] = v[i==3]
    #if i == 4: return t, p, v
    r[i==4] = t[i==4]
    g[i==4] = p[i==4]
    b[i==4] = v[i==4]
    #if i == 5: return v, p, q
    r[i==5] = v[i==5]
    g[i==5] = p[i==5]
    b[i==5] = q[i==5]

    r[s==0] = v[s==0]
    g[s==0] = v[s==0]
    b[s==0] = v[s==0]

    return r, g, b
    # Cannot get here
