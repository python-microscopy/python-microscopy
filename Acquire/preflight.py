import wx

def ShowPreflightResults(parent, failedChecks):
    if len(failedChecks) == 0:
        return True

    else:
        #print failedChecks
        errormsgs = '\n'.join(['- ' + c.message for c in failedChecks])
        msg = 'Preflight check found the following potential problems:\n\n' + errormsgs + '\n\nDo you wish to continue?'
        dlg = wx.MessageDialog(parent, msg, 'Preflight Check:', wx.YES_NO|wx.NO_DEFAULT|wx.ICON_ERROR)

        ret = dlg.ShowModal()
        dlg.Destroy()

        return ret == wx.ID_YES



