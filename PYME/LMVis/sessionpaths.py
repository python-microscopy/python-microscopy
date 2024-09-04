from pathlib import Path
import os

# the idea of this module is to provide two functions to make paths either relative or absolute in session objects
# typically this will be relative to the '.pvs' session file path
# it also provides a method register_path_modulechecks to allow the user/module coder to request path
# translation for some recipe module arguments

# a dict of recipe module names with arguments to check
checkmodules = {}

# it seems best to request this directly in the python files in which the modules are defined
# example:     register_modulecheck('PYMEcs.MBMcorrection','mbmfile','mbmsettings')
# can this possibly be autoregistered by modifying the FileOrURI object suitably?
def register_path_modulechecks(module,*entries):
    checkmodules[module] = entries

SESSIONDIR_TOKEN = '$session_dir$'

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
    

# note that we do not touch any paths that are already absolute
# this means we can repeatedly call this on session objects without doing any damage
# this function only briefly kept for compatibility until all pvs rewritten
def make_session_paths_absolute_compat(session,sessionpath):
    sessionabs = session.copy()
    sessiondir = get_session_dirP(sessionpath)
    for ds in session['datasources']:
        fname,query = parse_fnq(session['datasources'][ds])
        sessionabs['datasources'][ds] = fnq_string(fnstring_absolute(fname,sessiondir),query)
    # users can request certain recipe module arguments to be also "path-translated" by registering the module/arguments
    for checkmod in checkmodules:
        for module in sessionabs['recipe']:
            if checkmod in module:
                for field in checkmodules[checkmod]:    
                    if field in module[checkmod]:
                        module[checkmod][field] = fnstring_absolute(module[checkmod][field],sessiondir)
    return sessionabs

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
        try:
            fnamerel = str(fnamep.relative_to(sessiondir)) # now make the filenames relative to the session dir path
        except ValueError: # value error implies it is not possible to make a "clean" relative path, i.e. that the file fnstring is not truely below sessiondir
            return None # signal via None, we do the further handling in the calling function
    else:
        fnamerel = fnstring
    return fnamerel

def allpaths_relative_to(session,sessiondir):
    # currently we only check data sources, NOT recipe FileOrURI entries
    for ds in session['datasources']:
        fname,query = parse_fnq(session['datasources'][ds])
        if not Path(fname).is_relative_to(sessiondir):
            return False
    return True

# now a little nicer
def process_recipe_paths(recipe,sessiondir):
    # users can request certain recipe module arguments to be "path-translated" by registering the module/arguments
    for module in recipe:
        [(modname,paramdict)] = module.items()
        for param in paramdict:
            if param in checkmodules.get(modname,[]):
                relstring = fnstring_relative(paramdict[param],sessiondir)
                if relstring is not None:
                    paramdict[param] = os.path.join(SESSIONDIR_TOKEN,relstring)

from PYME.IO.unifiedIO import is_cluster_uri
def resolve_relative_session_paths(session):
    for ds in session['datasources']:
        fname,query = parse_fnq(session['datasources'][ds])
        if not is_cluster_uri(fname) and not Path(fname).is_absolute() and not fname.startswith(SESSIONDIR_TOKEN):
            session['datasources'][ds] = fnq_string(Path(fname).resolve(),query)

def check_session_paths(session,sessiondir):
    resolve_relative_session_paths(session) # if started from command line some ds paths may be relative
    if allpaths_relative_to(session,sessiondir):
        session['relative_paths'] = True
        for ds in session['datasources']:
            fname,query = parse_fnq(session['datasources'][ds])
            pathstring = os.path.join(SESSIONDIR_TOKEN,fnstring_relative(fname,sessiondir))
            session['datasources'][ds] = fnq_string(pathstring,query)
        process_recipe_paths(session['recipe'],sessiondir)
    else:
        session['relative_paths'] = False
        
