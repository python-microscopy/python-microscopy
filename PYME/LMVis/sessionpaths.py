from pathlib import Path, PurePosixPath
import os
from PYME.recipes import base
from PYME.IO.unifiedIO import is_cluster_uri

import logging
logger = logging.getLogger(__name__)

SESSIONDIR_TOKEN = '$session_dir$'

# split filename?query type string in fname and query parts
def _parse_fnq(dsfnstr):
    parts = str(dsfnstr).split('?')
    if len(parts) > 1:
        return  (parts[0],parts[1])# we assume there are no further '?'s in the query string
    else:
        return  (parts[0],None)

# join filename and query to form the expected string for session loading
def _join_fnq_string(fn,query):
    if query is None:
        return str(fn)
    else:
        return str(fn) + '?' + query

def _get_relative_filename(fnstring,sessiondir):
    fnamep = Path(fnstring)
    if fnamep.is_absolute():
        try:
            fnamerel = fnamep.relative_to(sessiondir).as_posix() # now make the filenames relative to the session dir path
            return fnamerel
        except ValueError: # value error implies it is not possible to make a "clean" relative path, i.e. that the file fnstring is not truely below sessiondir
            return None # signal via None, we do the further handling in the calling function
    else:
        return None


def _path_is_safe_and_relativeto(fpath,sessiondir):
    is_safe = (not is_cluster_uri(fpath)) and (not fpath.startswith(SESSIONDIR_TOKEN)) and Path(fpath).is_relative_to(sessiondir)
    return is_safe

def allpaths_relative_to(session,sessiondir):
    """Check if all paths in session are below sessiondir"""
    for ds in session['datasources']:
        fname,query = _parse_fnq(session['datasources'][ds])
        if not _path_is_safe_and_relativeto(fname,sessiondir):
            return False
    
    for module in session['recipe']:
        [(mn, mod_dict)] = module.items()
        
        for k in base.all_modules[mn].file_or_uri_traits():
            if k in mod_dict:
                if not _path_is_safe_and_relativeto(mod_dict[k],sessiondir):
                    return False
                    
    return True

def _process_session_paths(session,sessiondir):
    """Rewrite paths in session to replace sessiondir with SESSIONDIR_TOKEN"""

    #TODO - if we really want to be portable, should we also be doing sep replacement?
    # e.g. replace os.sep with '/' in paths?
    #UPDATE - the as_posix() conversion here and in _get_relative_filename should now take care of this

    for ds in session['datasources']:
            fname,query = _parse_fnq(session['datasources'][ds])
            pathstring = PurePosixPath(SESSIONDIR_TOKEN).joinpath(_get_relative_filename(fname,sessiondir)).as_posix()
            session['datasources'][ds] = _join_fnq_string(pathstring,query)

    for module in session['recipe']:
        [(mn, mod_dict)] = module.items()

        for k in base.all_modules[mn].file_or_uri_traits():
            if k in mod_dict:
                relstring = _get_relative_filename(mod_dict[k],sessiondir)
                if relstring is None:
                    raise ValueError('Path %s is not below sessiondir %s' % (mod_dict[k],sessiondir))
                
                mod_dict[k]  = PurePosixPath(SESSIONDIR_TOKEN).joinpath(relstring).as_posix()

def _make_paths_absolute(session):
    """Make all paths in session absolute - if started from command line some ds paths
    may be relative to current working directory
    
    FIXME: Paths may also be relative to PYMEDATADIR, which is not handled here
    """
    for ds in session['datasources']:
        fname,query = _parse_fnq(session['datasources'][ds])
        if not is_cluster_uri(fname) and not Path(fname).is_absolute() and not fname.startswith(SESSIONDIR_TOKEN):
            session['datasources'][ds] = _join_fnq_string(Path(fname).resolve(),query)

def attempt_relative_session_paths(session,sessiondir):
    """Re-write paths to be relative to sessiondir if possible, otherwise leave them unchanged

    NOTE: undoing this operation on load is accomplished usin session_txt.replace(SESSIONDIR_TOKEN,sessiondir)
    
    Parameters
    ----------
    session : dict
        session dictionary
    sessiondir : str
        path to the session directory (i.e. the directory containing the session file)

    """

    _make_paths_absolute(session) # if started from command line some ds paths may be relative to current working directory
    
    if allpaths_relative_to(session,sessiondir):
        logger.debug('path are all below session dir, rewriting paths with SESSIONDIR_TOKEN')
        session['relative_paths'] = True
        _process_session_paths(session,sessiondir)
    else:
        logger.debug('some paths not below session dir, leaving paths unchanged')
        session['relative_paths'] = False

def substitute_sessiondir(session_txt, session_filename):
    """Substitute SESSIONDIR_TOKEN with sessiondir in session_txt

    Parameters
    ----------
    session_txt : str
        session dictionary as a string
    session_filename : str
        path to the session file

    Returns
    -------
    str
        session_txt with SESSIONDIR_TOKEN replaced by sessiondir
    """
    from pathlib import Path
    sessiondir = Path(session_filename).resolve().parent.as_posix()

    return session_txt.replace(SESSIONDIR_TOKEN+"\\",SESSIONDIR_TOKEN+'/').replace(SESSIONDIR_TOKEN,sessiondir)

