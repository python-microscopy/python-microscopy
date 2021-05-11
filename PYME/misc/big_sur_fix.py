# from bht at https://stackoverflow.com/questions/63475461/unable-to-import-opengl-gl-in-python-on-macos
try:
    import OpenGL as ogl
    try:
        import OpenGL.GL   # this fails in <=2020 versions of Python on OS X 11.x
    except ImportError:
        print('Drat, patching for Big Sur')
        from ctypes import util
        orig_util_find_library = util.find_library
        def new_util_find_library( name ):
            res = orig_util_find_library( name )
            if res: return res
            if name == 'c':
                # work-around for ifaddr
                # exploits the fact that ctypes.CDLL(None) opens the System library
                return None
            return '/System/Library/Frameworks/'+name+'.framework/'+name
        util.find_library = new_util_find_library
except ImportError:
    pass
