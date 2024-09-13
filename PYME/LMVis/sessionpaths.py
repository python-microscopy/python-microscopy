from pathlib import Path
import os
from PYME.recipes import base
from PYME.IO.unifiedIO import is_cluster_uri

import logging
logger = logging.getLogger(__name__)

SESSIONDIR_TOKEN = '$session_dir$'

# split filename?query type string in fname and query parts
def parse_fnq(dsfnstr):
    parts = dsfnstr.split('?')
    if len(parts) > 1:
        return  (parts[0],parts[1])# we assume there are no further '?'s in the query string
    else:
        return  (parts[0],None)

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

def path_is_safe_and_relativeto(fpath,sessiondir):
    is_safe = (not is_cluster_uri(fpath)) and (not fpath.startswith(SESSIONDIR_TOKEN)) and Path(fpath).is_relative_to(sessiondir)
    return is_safe

def allpaths_relative_to(session,sessiondir):
    for ds in session['datasources']:
        fname,query = parse_fnq(session['datasources'][ds])
        if not path_is_safe_and_relativeto(fname,sessiondir):
            return False
    for module in session['recipe']:
        [(mn, mod_dict)] = module.items()
        # we only check registered modules - session loading will anyway fail with an unregistered module
        if mn in base.all_modules: # it is possible that there is an unregistered module in the recipe
            for k in base.all_modules[mn].file_or_uri_traits():
                if k in mod_dict:
                    if not path_is_safe_and_relativeto(mod_dict[k],sessiondir):
                        return False
    return True

def process_session_paths(session,sessiondir):
    for ds in session['datasources']:
            fname,query = parse_fnq(session['datasources'][ds])
            pathstring = os.path.join(SESSIONDIR_TOKEN,fnstring_relative(fname,sessiondir))
            session['datasources'][ds] = fnq_string(pathstring,query)
    for module in session['recipe']:
        [(mn, mod_dict)] = module.items()
        # we only check registered modules - session loading will anyway fail with an unregistered module
        if mn in base.all_modules: # it is possible that there is an unregistered module in the recipe
            for k in base.all_modules[mn].file_or_uri_traits():
                if k in mod_dict:
                    relstring = fnstring_relative(mod_dict[k],sessiondir)
                    if relstring is not None: # do not translate if path is not below sessiondir
                        mod_dict[k]  = os.path.join(SESSIONDIR_TOKEN,relstring)

def resolve_relative_session_paths(session):
    for ds in session['datasources']:
        fname,query = parse_fnq(session['datasources'][ds])
        if not is_cluster_uri(fname) and not Path(fname).is_absolute() and not fname.startswith(SESSIONDIR_TOKEN):
            session['datasources'][ds] = fnq_string(Path(fname).resolve(),query)

def check_session_paths(session,sessiondir):
    resolve_relative_session_paths(session) # if started from command line some ds paths may be relative
    if allpaths_relative_to(session,sessiondir):
        logger.debug('path are all below session dir, rewriting paths with SESSIONDIR_TOKEN')
        session['relative_paths'] = True
        process_session_paths(session,sessiondir)
    else:
        logger.debug('some paths not below session dir, leaving paths unchanged')
        session['relative_paths'] = False
        
