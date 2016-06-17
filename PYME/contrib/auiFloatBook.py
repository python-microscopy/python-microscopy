#!/usr/bin/python
"""
auinotebookwithfloatingpages.py

This module provides the class AuiNotebookWithFloatingPages, which is a
subclass of wx.aui.AuiNotebook. It allows the user to drag tabs of the
notebook out of the notebook and have the contents of the tab being
transferred to a frame, thus creating a floating page. When the user
closes a floating page, the page is placed back in the active pane of
the notebook.

Known limitation: when the notebook is more or less full screen, tabs
cannot be dragged far enough outside of the notebook to become
floating pages.

Author: Frank Niessink <frank@...>
License: wxWidgets license
Version: 0.1
Date: August 8, 2007

"""

import wx, wx.aui


class AuiNotebookWithFloatingPages(wx.aui.AuiNotebook):
    def __init__(self, *args, **kwargs):
        super(AuiNotebookWithFloatingPages, self).__init__(*args, **kwargs)
        self.__auiManager = self.GetAuiManager()
        self.Bind(wx.aui.EVT_AUINOTEBOOK_END_DRAG, self.onEndDrag)
        self.Bind(wx.aui.EVT_AUINOTEBOOK_DRAG_MOTION, self.onDragMotion)

    def onDragMotion(self, event):
        self.__auiManager.HideHint()
        if self.IsMouseWellOutsideWindow():
            x, y = wx.GetMousePosition()
            hintRect = wx.Rect(x, y, 400, 300)
            # Use CallAfter so we overwrite the hint that might be
            # shown by our superclass:
            wx.CallAfter(self.__auiManager.ShowHint, hintRect)
        event.Skip()

    def onEndDrag(self, event):
        self.__auiManager.HideHint()
        if self.IsMouseWellOutsideWindow():
            # Use CallAfter so we our superclass can deal with the event first
            wx.CallAfter(self.FloatPage, self.Selection)
        event.Skip()

    def IsMouseWellOutsideWindow(self):
        screenRect = self.GetScreenRect()
        screenRect.Inflate(50, 50)
        return not screenRect.Contains(wx.GetMousePosition())

    def FloatPage(self, pageIndex):
        pageTitle = self.GetPageText(pageIndex)
        pageContents = self.GetPage(pageIndex)
        frame = wx.MiniFrame(self, title=pageTitle,
            style=wx.DEFAULT_FRAME_STYLE)#|wx.FRAME_TOOL_WINDOW)
        frame.SetClientSize(pageContents.GetEffectiveMinSize())
        pageContents.Reparent(frame)
        self.RemovePage(pageIndex)
        frame.Bind(wx.EVT_CLOSE, self.onCloseFloatingPage)
        frame.Move(wx.GetMousePosition())
        frame.Show()

    def onCloseFloatingPage(self, event):
        event.Skip()
        frame = event.GetEventObject()
        pageTitle = frame.GetTitle()
        pageContents = frame.GetChildren()[0]
        pageContents.Reparent(self)
        self.AddPage(pageContents, pageTitle, select=True)


if __name__ == '__main__':

    def createPanel(parent, ContentClass, *args, **kwargs):
        panel = wx.Panel(parent)
        content = ContentClass(panel, *args, **kwargs)
        sizer = wx.BoxSizer()
        sizer.Add(content, flag=wx.EXPAND, proportion=1)
        panel.SetSizerAndFit(sizer)
        return panel

    app = wx.App(redirect=False)
    frame = wx.Frame(None)
    notebook = AuiNotebookWithFloatingPages(frame)
    for index in range(5):
        page = createPanel(notebook, wx.TextCtrl,
            value='This is page %d'%index, style=wx.TE_MULTILINE)
        notebook.AddPage(page, 'Page %d'%index)
    frame.Show()
    app.MainLoop() 
