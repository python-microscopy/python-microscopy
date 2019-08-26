#!/usr/bin/python

###############
# workspaceTree.py
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
import wx
import wx.gizmos
from wx.lib.mixins.treemixin import VirtualTree
#from pylab import cm, array

from PYME.IO import tabular

class WorkWrap:
    def __init__(self, dict):
        self.dict = dict
        self.keysToReport = []
        self.keyColours = {}
        self.nColours = 0

    def addKey(self, keyName, colour=None):
        self.keysToReport.append(keyName)
        if not colour is None:
            self.keyColours[keyName] = colour

    def newColour(self):
        ret =  self.nColours
        self.nColours += 1
        return ret

    def keys(self):
        return self.keysToReport

    def __getitem__(self, key):
        return self.dict[key]

class WorkspaceTree(VirtualTree, wx.gizmos.TreeListCtrl):
    def __init__(self, *args, **kwargs):
        self.workspace = kwargs.pop('workspace')
        self.shell = kwargs.pop('shell')

        #wx.TreeCtrl.__init__(self, *args, **kwargs)
        VirtualTree.__init__(self, *args, **kwargs)

        isz = (16,16)
        il = wx.ImageList(isz[0], isz[1])
        self.fldridx     = il.Add(wx.ArtProvider_GetBitmap(wx.ART_FOLDER,      wx.ART_OTHER, isz))
        self.fldropenidx = il.Add(wx.ArtProvider_GetBitmap(wx.ART_FILE_OPEN,   wx.ART_OTHER, isz))
        self.fileidx     = il.Add(wx.ArtProvider_GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, isz))

        self.SetImageList(il)
        self.il = il

        self.AddColumn("Name")
        self.AddColumn("Class")
        self.AddColumn("Repr")
        self.SetMainColumn(0) # the one with the tree in it...
        self.SetColumnWidth(0, 175)

        self.GetMainWindow().Bind(wx.EVT_RIGHT_UP, self.OnRightUp)

        try:
            #this bails on windows - but also doesn't seem to be required
            #under windows - so let it fail quietly
            self.ExpandAllChildren(self.GetRootItem())
        except:
            pass
        self.RefreshItems()

        
        
    def _getChild(self, item, n):
        """get the nth child of an item"""
        if '__getitem__' in dir(item):
            #either list like or dict like
            if 'keys' in dir(item):
                #dict like
                return item[list(item.keys())[n]]
            else:
                return item[n]
        else:
            return list(item.__dict__.values())[n]

    def _getChildName(self, item, n):
        """get the nth child of an item"""
        if '__getitem__' in dir(item):
            #either list like or dict like
            if 'keys' in dir(item):
                #dict like
                return item.keys()[n]
            else:
                return '[%d]' %n
        else:
            return list(item.__dict__.keys())[n]

    def _getChildPathPart(self, item, n):
        """get the nth child of an item"""
        if '__getitem__' in dir(item):
            #either list like or dict like
            if 'keys' in dir(item):
                #dict like
                return "['%s']" % list(item.keys())[n]
            else:
                return '[%d]' %n
        else:
            return '.' + list(item.__dict__.keys())[n]

    def _getChildPath(self, index):
        """get the nth child of an item"""
        curItem = self.workspace
        path = self._getChildName(curItem, index[0])

        #walk down children
        for i in range(len(index) - 1):
            curItem = self._getChild(curItem,index[i])
            path += self._getChildPathPart(curItem, index[i+1])    

        return path

    def _getNumChildren(self, item):
        """get the nth child of an item"""
        if type(item).__name__ in ['str', 'unicode', 'shmarray', 'ndarray']:
            #special case for things we don't want to expand
            return 0
        elif '__getitem__' in dir(item):
            #either list like or dict like
            if '__len__' in dir(item):
                #dict like
                return len(item)
            elif 'keys' in dir(item):
                return len(item.keys())
        elif '__dict__' in dir(item):
            return len(item.__dict__)
        else:
            return 0


    def OnGetChildrenCount(self, index):
        if index is None:
            return len(self.workspace)
        else:
            curItem = self.workspace

            #walk down children
            for i in index:
                curItem = self._getChild(curItem, i)

            return self._getNumChildren(curItem)

    def OnGetItemText(self, index, column=0):
        if index is None:
            return ''
        else:
            curItem = self.workspace

            #walk down children
            for i in index[:-1]:
                curItem = self._getChild(curItem, i)

            if column ==0:
                return self._getChildName(curItem, index[-1])
            elif column == 1:
                if isinstance(curItem, tabular.TabularBase):
                    return 'ndarray'
                else:
                    curItem = self._getChild(curItem, index[-1])
                    return curItem.__class__.__name__
            elif column == 2:
                if isinstance(curItem, tabular.TabularBase):
                    return ''
                else:
                    curItem = self._getChild(curItem, index[-1])
                    r = repr(curItem)
                    if len(r) > 100:
                        r = r[:100] + ' ...'

                    return r

    def OnGetItemImage(self, index, which = 0, column=0):
        if not column == 0:
            return -1
        
        curItem = self.workspace
        
        if not index is None:
            #walk down children
            for i in index:
                curItem = self._getChild(curItem, i)

        nChildren = self._getNumChildren(curItem)

        if nChildren == 0:
            return self.fileidx
        else:
            if which == wx.TreeItemIcon_Normal or which == wx.TreeItemIcon_Selected:
                return self.fldridx
            elif which == wx.TreeItemIcon_Expanded or which == wx.TreeItemIcon_SelectedExpanded:
                return self.fldropenidx
            else:
                return self.fileidx

#    def OnGetItemBackgroundColour(self, index):
#        name =  self._getChildName(self.workspace, index[0])
#        if 'keyColours' in dir(self.workspace) and name in self.workspace.keyColours.keys():
#            c = float(self.workspace.keyColours[name])/self.workspace.nColours
#            #print c
#            c = ((127*array(cm.hsv(c)) + 128)[:3]).astype('i')
#            #print c
#            return wx.Colour(*c)
#        else:
#            return wx.NullColour

        

    def OnRightUp(self, evt):
        pos = evt.GetPosition()
        item, flags, col = self.HitTest(pos)
        self.shell.AddText(self._getChildPath(self.GetIndexOfItem(item)))




