import wx

def CheckWxPhoenix():
    if 'phoenix' in wx.version():
        return True
    return False

wxPythonPhoenix = CheckWxPhoenix()

# currently for initial testing simply forcing to 4.x
# could replace with test that figures out if newer or older wx present
# wxPythonPhoenix = True

def BitmapFromImage(image, depth=-1):
    if wxPythonPhoenix:
        return wx.Bitmap(img=image, depth=depth)
    else:
        return wx.BitmapFromImage(image, depth=depth)


def ImageFromBitmap(bitmap):
    if wxPythonPhoenix:
        return bitmap.ConvertToImage()
    else:
        return wx.ImageFromBitmap(bitmap)

def ImageFromData(width,height,data):
    if wxPythonPhoenix:
        return wx.Image(width,height,data)
    else:
        return wx.ImageFromData(width,height,data)
    
def EmptyBitmap(width, height, depth=-1):
    if wxPythonPhoenix:
        return wx.Bitmap(width=width, height=height, depth=depth)
    else:
        return wx.EmptyBitmap(width=width, height=height, depth=depth)

def EmptyImage(width, height, clear=True):
    if wxPythonPhoenix:
        return wx.Image(width=width, height=height, clear=clear)
    else:
        return wx.EmptyImage(width=width, height=height, clear=clear)
