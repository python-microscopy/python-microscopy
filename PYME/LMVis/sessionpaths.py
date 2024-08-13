from pathlib import Path

# the idea of this module is to provide two functions to make paths either relative or absolute in session objects
# typically this will be relative to the '.pvs' session file path
# it also provides a method register_path_modulechecks to allow the user/module coder to request path
# translation for some recipe module arguments

# a dict of recipe module names with arguments to check
checkmodules = {}

# it seems best to request this directly in the python files in which the modules are defined
# example:     register_modulecheck('PYMEcs.MBMcorrection','mbmfile','mbmsettings')
# can this possibly be autoregistered by modifying the Traits.File object suitably?
def register_path_modulechecks(module,*entries):
    checkmodules[module] = entries

# split filename?query type string in fname and query parts
def parse_fnq(dsfnstr):
    parts = dsfnstr.split('?')
    parts0 = parts[0]
    if len(parts) > 1:
        parts1 = parts[1] # we assume there are no further '?'s in the query string
    else:
        parts1 = None
    return (parts0,parts1)

# join filename and query to form the expected string for session loading
def fnq_string(fn,query):
    if query is None:
        return fn
    else:
        return fn + '?' + query

def fnstring_relative(fnstring,sessiondir):
    fnamep = Path(fnstring)
    if fnamep.is_absolute():
        fnamerel = str(fnamep.relative_to(sessiondir)) # now make the filenames relative to the session dir path
    else:
        fnamerel = fnstring
    return fnamerel

def fnstring_absolute(fnstring,sessiondir):
    fnamep = Path(fnstring)
    if fnamep.is_absolute():
        fnameabs = fnstring
    else:
        fnameabs = str(sessiondir / fnamep) # now make the filenames relative to the session dir path
    return fnameabs

def get_session_dirP(sessionpath):
    sessionpathp = Path(sessionpath)
    abspath = sessionpathp.resolve()
    sessiondir = abspath.parent # get the directory part of this absolute path

    return sessiondir # return sessiondir as Path object
    
# note that we do not touch any paths that are already relative
# this means we can repeatedly call this on session objects without doing any damage
def make_session_paths_relative(session,sessionpath):
    sessionrel = session.copy()
    sessiondir = get_session_dirP(sessionpath)
    for ds in session['datasources']:
        fname,query = parse_fnq(session['datasources'][ds])
        sessionrel['datasources'][ds] = fnq_string(fnstring_relative(fname,sessiondir),query)
    # users can request certain recipe module arguments to be also "path-translated" by registering the module/arguments
    for checkmod in checkmodules:
        for module in sessionrel['recipe']:
            if checkmod in module:
                # print("PYMEcs.MBMcorrection in module")
                for field in checkmodules[checkmod]:    
                    if field in module[checkmod]:
                        module[checkmod][field] = fnstring_relative(module[checkmod][field],sessiondir)
    return sessionrel

# note that we do not touch any paths that are already absolute
# this means we can repeatedly call this on session objects without doing any damage
def make_session_paths_absolute(session,sessionpath):
    sessionabs = session.copy()
    sessiondir = get_session_dirP(sessionpath)
    for ds in session['datasources']:
        fname,query = parse_fnq(session['datasources'][ds])
        sessionabs['datasources'][ds] = fnq_string(fnstring_absolute(fname,sessiondir),query)
    # users can request certain recipe module arguments to be also "path-translated" by registering the module/arguments
    for checkmod in checkmodules:
        for module in sessionabs['recipe']:
            if checkmod in module:
                # print("PYMEcs.MBMcorrection in module")
                for field in checkmodules[checkmod]:    
                    if field in module[checkmod]:
                        module[checkmod][field] = fnstring_absolute(module[checkmod][field],sessiondir)
    return sessionabs
