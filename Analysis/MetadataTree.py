#!/usr/bin/python

##################
# MetadataTree.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import  wx
import  wx.gizmos   as  gizmos

from PYME.Acquire.MetaDataHandler import NestedClassMDHandler

class MetadataPanel(wx.Panel):
    def __init__(self, parent, mdh):
        self.mdh=mdh
        wx.Panel.__init__(self, parent, -1)
        self.Bind(wx.EVT_SIZE, self.OnSize)

        self.tree = gizmos.TreeListCtrl(self, -1, style =
                                        wx.TR_DEFAULT_STYLE
                                        #| wx.TR_HAS_BUTTONS
                                        #| wx.TR_TWIST_BUTTONS
                                        #| wx.TR_ROW_LINES
                                        #| wx.TR_COLUMN_LINES
                                        #| wx.TR_NO_LINES
                                        | wx.TR_FULL_ROW_HIGHLIGHT
                                   )

        
        # create some columns
        self.tree.AddColumn("Entry")
        self.tree.AddColumn("Value")
        
        self.tree.SetMainColumn(0) # the one with the tree in it...
        self.tree.SetColumnWidth(0, 200)


        self.root = self.tree.AddRoot("Metadata")
        self.tree.SetItemText(self.root, "root", 0)

        nmdh = NestedClassMDHandler(mdh)
        self.addEntries(nmdh, self.root)

        self.paths = {}

        #entryNames = self.mdh.getEntryNames()
        
#        for k in nmdh.__dict__.keys():
#            #txt = "Item %d" % x
#            child = self.tree.AppendItem(self.root, k)
#            self.tree.SetItemText(child, txt + "(c1)", 1)
                                                

        self.tree.Expand(self.root)

        self.tree.GetMainWindow().Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)
        #self.tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.OnActivate)

    def addEntries(self, mdh, node, entrypath=''):
        #en = []
        for k in mdh.__dict__.keys():
            child = self.tree.AppendItem(node, k)
            self.tree.SetItemText(child, k, 0)
            if mdh.__dict__[k].__class__ == NestedClassMDHandler:
                self.addEntries(mdh.__dict__[k], child, '.'.join((entrypath, k)))
            else:
                self.tree.SetItemText(child, str(mdh.getEntry(k)), 1)
                self.paths[child] = '.'.join((entrypath, k))


    #def OnActivate(self, evt):
        #self.log.write('OnActivate: %s' % self.tree.GetItemText(evt.GetItem()))


    def OnRightDown(self, evt):
        pos = evt.GetPosition()
        item, flags, col = self.tree.HitTest(pos)
        if item:
            self.log.write('Flags: %s, Col:%s, Text: %s' %
                           (flags, col, self.tree.GetItemText(item, col)))

    def OnSize(self, evt):
        self.tree.SetSize(self.GetSize())


