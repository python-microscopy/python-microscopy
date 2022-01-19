def show_about_dlg(component_name='python-microscopy', desc_addendum=''):
    from PYME.version import detailed_version
    from PYME.resources import getIconPath
    import wx.adv
    # msg = "PYME Visualise\n\n Visualisation of localisation microscopy data\nDavid Baddeley 2009"
    
    # dlg = wx.MessageDialog(self, msg, "About PYME Visualise",
    #                        wx.OK | wx.ICON_INFORMATION)
    # dlg.SetFont(wx.Font(8, wx.NORMAL, wx.NORMAL, wx.NORMAL, False, "Verdana"))
    # dlg.ShowModal()
    # dlg.Destroy()
    
    if desc_addendum == '':
        desc = 'A suite of tools for microscopy data collection and processing'
    else:
        desc = desc_addendum + '\n\nPart of the python-microscopy suite'
    
    dlg = wx.adv.AboutDialogInfo()
    dlg.SetName(component_name)
    dlg.SetVersion(detailed_version())
    dlg.SetDescription(desc)
    dlg.SetCopyright("(C)2009-2021")
    dlg.SetIcon(wx.Icon(getIconPath('pymeLogo.png')))
    #dlg.SetLicense("GPLv3") # I think we need to either expand or omit
    # TODO: should this be the issues page or the website
    dlg.SetWebSite("https://github.com/python-microscopy/python-microscopy/issues", desc="Report an issue")
    #dlg.AddDeveloper("David Baddeley") #should probably be all or none here, punting full list for now
    
    wx.adv.AboutBox(dlg)