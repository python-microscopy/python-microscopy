#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

from PYME.IO.FileUtils.pickledLoad import np_load_legacy

class MDParam(object):
    def __init__(self):
        pass
        
    def formFields(self):
        return [self.formField()]
        #, mdToUpdate=None):
        #self.mdToUpdate = mdToUpdate

    #def OnChange(self, event=None):
    #    if not self.mdToUpdate is None:
    #        self.retrieveValue(self.mdToUpdate)


class FloatParam(MDParam):
    def __init__(self, paramName, guiName, default=0, helpText='', **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default

        MDParam.__init__(self, **kwargs)
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))

        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)
            self.retrieveValue(mdh)
            self.tValue.Bind(wx.EVT_TEXT, lambda e : self.retrieveValue(mdh))

        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = float(self.tValue.GetValue())

    def updateValue(self, mdh, **kwargs):
        self.tValue.SetValue('%3.2f' % mdh.getOrDefault(self.paramName, self.default))

    def formField(self):
        from django import forms
        return self.paramName , forms.FloatField(label=self.guiName, initial=self.default)
        

class IntParam(MDParam):
    def __init__(self, paramName, guiName, default=0, helpText='', **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default

        MDParam.__init__(self, **kwargs)
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))
        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)
            self.retrieveValue(mdh)
            self.tValue.Bind(wx.EVT_TEXT, lambda e : self.retrieveValue(mdh))
        
        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = int(self.tValue.GetValue())

    def updateValue(self, mdh, **kwargs):
        self.tValue.SetValue('%d' % mdh.getOrDefault(self.paramName, self.default))

    def formField(self):
        from django import forms
        return self.paramName, forms.IntegerField(label=self.guiName, initial=self.default)

class RangeParam(MDParam):
    def __init__(self, paramName, guiName, default=[0, 0], helpText='', **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default

        MDParam.__init__(self, **kwargs)
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='', size=(50, -1))

        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)
            self.retrieveValue(mdh)
            self.tValue.Bind(wx.EVT_TEXT, lambda e : self.retrieveValue(mdh))

        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = tuple([int(v) for v in self.tValue.GetValue().split(':')])

    def updateValue(self, mdh, **kwargs):
        val = ':'.join(['%d' % v for v in mdh.getOrDefault(self.paramName, self.default)])
        self.tValue.SetValue(val)

    def formField(self):
        from django import forms
        class RangeField(forms.CharField):
            def clean(self, value):
                import json
                val = json.loads(value)
                return val

        return self.paramName, RangeField(label=self.guiName, initial=repr(self.default))
        
        
class StringParam(MDParam):
    def __init__(self, paramName, guiName, default='', helpText='', **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default

        MDParam.__init__(self, **kwargs)
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))
        
        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)
            self.retrieveValue(mdh)
            self.tValue.Bind(wx.EVT_TEXT, lambda e : self.retrieveValue(mdh))

        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = self.tValue.GetValue()

    def updateValue(self, mdh, **kwargs):
        self.tValue.SetValue(mdh.getOrDefault(self.paramName, self.default))

    def formField(self):
        from django import forms
        return self.paramName, forms.CharField(label=self.guiName, initial=self.default, required=False)

        
class FloatListParam(MDParam):
    def __init__(self, paramName, guiName, default='', helpText='', **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default

        MDParam.__init__(self, **kwargs)
        
    def _valToString(self, values):
        return ''.join(['%3.2f' % v for v in values])
        
    def _strToVal(self, s):
        sl = s.split(',')
        return [float(si) for si in sl]
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))

        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)
            self.retrieveValue(mdh)
            self.tValue.Bind(wx.EVT_TEXT, lambda e : self.retrieveValue(mdh))

        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = self._strToVal(self.tValue.GetValue())

    def updateValue(self, mdh, **kwargs):
        self.tValue.SetValue(self._valToString(mdh.getOrDefault(self.paramName, self.default)))

    def formField(self):
        from django import forms
        class FloatListField(forms.CharField):
            def clean(self, value):
                import json
                val = json.loads(value)
                return [float(v) for v in val]

        return self.paramName, forms.FloatListField(label=self.guiName, initial=repr(self.default))
        

class ChoiceParam(MDParam):
    def __init__(self, paramName, guiName, default='', helpText='', choices = [], choiceNames = [], **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default
        self.choices = choices
        self.choiceNames = choiceNames
        if choiceNames == []:
            self.choiceNames = choices

        MDParam.__init__(self, **kwargs)
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(parent, -1, self.guiName), 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.cValue = wx.Choice(parent, -1, choices = self.choiceNames, size=(100, -1))

        hsizer.Add(self.cValue, 1,wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)
            self.retrieveValue(mdh)
            self.cValue.Bind(wx.EVT_CHOICE, lambda e : self.retrieveValue(mdh))

        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = self.choices[self.cValue.GetSelection()]

    def updateValue(self, mdh, **kwargs):
        self.cValue.SetSelection(self.choices.index(mdh.getOrDefault(self.paramName, self.default)))

    def formField(self):
        from django import forms
        return self.paramName, forms.ChoiceField(label=self.guiName, choices=[(c, c) for c in self.choiceNames], initial=self.default)
        
        
        
class FilenameParam(MDParam):
    def __init__(self, paramName, guiName, default='<none>', helpText='', 
                    prompt='Please select file', wildcard='All files|*.*', filename = None, **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default
        self.filename = default
        if not filename is None:
            self.filename = filename
        self.prompt = prompt
        self.wildcard = wildcard

        MDParam.__init__(self, **kwargs)
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx #, os
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        FieldText = '%s %s' %(self.guiName, self.default)

        self.stFilename = wx.StaticText(parent, -1, FieldText)
        hsizer.Add(self.stFilename, 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)

        bSetFile = wx.Button(parent, -1, 'Set', style=wx.BU_EXACTFIT)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)
            self.retrieveValue(mdh, False)
            bSetFile.Bind(wx.EVT_BUTTON, lambda e : self._setFile(mdh))
        hsizer.Add(bSetFile, 0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 0)

        return hsizer
        
    def _setFile(self, mdh):
        import wx,os
        fdialog = wx.FileDialog(None, self.prompt,
                    #defaultDir=os.path.split(self.image.filename)[0],
                    wildcard=self.wildcard, style=wx.FD_OPEN)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            #self.ds = example.CDataStack(fdialog.GetPath().encode())
            #self.ds =
            #psfFilename = fdialog.GetPath()
            #self.image.mdh.setEntry('PSFFile', getRelFilename(psfFilename))
            self.filename = fdialog.GetPath()
            #self.md.setEntry('PSFFile', psfFilename)
            self.stFilename.SetLabel('%s %s' % (self.guiName, os.path.split(self.filename)[1]))
            self.stFilename.SetForegroundColour(wx.Colour(0, 128, 0))

            #if self.mdToUpdate and not (self.filename == self.default):
            #    self.OnChange()

            if not self.filename == self.default:
                #mdh[self.paramName] = self.filename
                self.retrieveValue(mdh, False)

            return True
        else:
            return False
        
    def retrieveValue(self, mdh, must_be_defined=True):
        if not self.filename == self.default:
            mdh[self.paramName] = self.filename
        elif must_be_defined:
            if self._setFile(mdh):
                #try to call this manually
                mdh[self.paramName] = self.filename
            else:
                raise RuntimeError('Required fit filename %s not defined' % self.paramName)

    def updateValue(self, mdh, **kwargs):
        import os, wx
        haveFile = False

        if self.paramName in mdh.getEntryNames():
            self.filename = mdh[self.paramName]
            FieldText = '%s ' % self.guiName + os.path.split(mdh[self.paramName])[1]
            haveFile = True
            #print(FieldText)

            self.stFilename.SetLabel(FieldText)
            #self.stFilename.SetForegroundColour(wx.Colour(0, 128, 0))
        else:
            FieldText = '%s %s' %(self.guiName, self.default)
            self.stFilename.SetLabel(FieldText)
            #self.stFilename.SetForegroundColour(wx.Colour(0, 0, 0))

        #print self.filename, self.default
        if (self.filename == self.default) or (self.filename == ''):
            self.stFilename.SetForegroundColour(wx.Colour(0, 0, 0))
        else:
            self.stFilename.SetForegroundColour(wx.Colour(0, 128, 0))

    def formField(self):
        from django import forms
        from PYME.misc import django_widgets
        return self.paramName, forms.CharField(label=self.guiName, initial=self.default, required=False, widget=django_widgets.ClusterFileInput())
       

class ShiftFieldParam(FilenameParam):    
    def retrieveValue(self, mdh, *args, **kwargs):
        import numpy as np
        
        oldfn = mdh.getOrDefault(self.paramName, None)
        if 'chroma.dx' in mdh.getEntryNames():
            if self.filename == self.default:
                self.filename = 'legacy'
                oldfn = 'legacy'
        FilenameParam.retrieveValue(self, mdh, *args, **kwargs)
        
        if not self.filename == oldfn and not self.filename in ['<none>', '']:
            dx, dy = np_load_legacy(self.filename)
            mdh.setEntry('chroma.dx', dx)
            mdh.setEntry('chroma.dy', dy)
        

class BoolParam(MDParam):
    def __init__(self, paramName, guiName, default=False, helpText='', **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default

        MDParam.__init__(self, **kwargs)
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.cbValue = wx.CheckBox(parent, -1, self.guiName)
        #self.cbValue.SetValue(self.default)
        
        hsizer.Add(self.cbValue, 1,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)
            self.retrieveValue(mdh)
            self.cbValue.Bind(wx.EVT_CHECKBOX, lambda e : self.retrieveValue(mdh))

        return hsizer
        
    def retrieveValue(self, mdh):
        mdh[self.paramName] = self.cbValue.GetValue()  

    def updateValue(self, mdh, **kwargs):
        self.cbValue.SetValue(mdh.getOrDefault(self.paramName, self.default))

    def formField(self):
        from django import forms
        return self.paramName, forms.BooleanField(label=self.guiName, initial=self.default, required=False)
 

class BoolFloatParam(MDParam):
    def __init__(self, paramName, guiName, default=False, helpText='', ondefault=0, offvalue=0, **kwargs):
        self.paramName = paramName
        self.guiName = guiName
        self.default = default
        self.ondefault = ondefault
        self.offvalue = offvalue

        MDParam.__init__(self, **kwargs)
        
    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        self.cbValue = wx.CheckBox(parent, -1, self.guiName)
        hsizer.Add(self.cbValue, 1,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        
        self.tValue = wx.TextCtrl(parent, -1, value='0', size=(50, -1))
        hsizer.Add(self.tValue, 0,wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.updateValue(mdh)

        if syncMdh:
            if mdhChangedSignal:
                mdhChangedSignal.connect(self.updateValue)

            self.retrieveValue(mdh)

            self.cbValue.Bind(wx.EVT_CHECKBOX, lambda e : self.retrieveValue(mdh))
            self.tValue.Bind(wx.EVT_TEXT, lambda e : self.retrieveValue(mdh))

        return hsizer
        
    def retrieveValue(self, mdh):
        if self.cbValue.GetValue():
            mdh[self.paramName] = float(self.tValue.GetValue())
        else:
            mdh[self.paramName] = float(self.offvalue)

    def updateValue(self, mdh, **kwargs):
        try:
            fval = mdh[self.paramName]
            if (fval == self.offvalue):
                fval = self.ondefault
                cval = False
            else:
                cval = True
        except (KeyError, AttributeError):
            fval = self.ondefault
            cval = self.default

        self.tValue.SetValue('%3.2f' % fval)
        self.cbValue.SetValue(cval)

    def formField(self):
        from django import forms
        #FIXME - find a suitable way to represent this
        return self.paramName, forms.FloatField(label=self.guiName, initial=float(self.default)*self.ondefault)



class ParamGroup(object):
    def __init__(self, name, items, helpText='', folded=True):
        self.name = name
        self.items = items
        self.folded = folded
        
    def _createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx
        
        box = wx.StaticBox(parent, label=self.name)
        sbSizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        for item in self.items:
            it = item.createGUI(box, mdh, syncMdh, mdhChangedSignal)
            sbSizer.Add(it, 0, wx.RIGHT|wx.EXPAND, 5)
        
        return sbSizer

    def createGUI(self, parent, mdh, syncMdh=False, mdhChangedSignal=None):
        import wx

        from PYME.ui import manualFoldPanel as afp

        clp = afp.collapsingPane(parent, caption=self.name, folded=self.folded)
        cp = wx.Panel(clp, -1)
        vsizer = wx.BoxSizer(wx.VERTICAL)
    
        for item in self.items:
            it = item.createGUI(cp, mdh, syncMdh, mdhChangedSignal)
            vsizer.Add(it, 0, wx.RIGHT | wx.EXPAND, 5)
    
        cp.SetSizer(vsizer)
        clp.AddNewElement(cp)
        return clp
    
    def formFields(self):
        return [item.formField() for item in self.items]
        
