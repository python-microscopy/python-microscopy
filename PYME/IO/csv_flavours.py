import os
from PYME.IO import MetaDataHandler

import logging
logger = logging.getLogger(__file__)

def isnumber(s):
    try:
        float(s)
        return True
    except:
        return False


csv_flavours = {
    'elyra' : {
        'idnames' : ['Index','First Frame'], # column names that identify this flavour
        'comment_char' : '#', # it does not really work to use these the comment char to ignore lines with missing data
        'delimiter' : '\t',
        'column_name_mappings' : {  # a dictionary of name substitutions
            'Position X [nm]': 'x',
            'Position Y [nm]': 'y',
            'Number Photons': 'nPhotons',
            'First Frame': 't',
            'Chi square': 'nchi2',
            'PSF half width [nm]': 'fwhm', # this one needs a mapping to sig really
            'Precision [nm]': 'error_x',
            'Channel' : 'probe',
        },
        'ignore_errors' : True, # these files bomb unless we tell np.genfromtxt to ignore errors
        # 'column_translations' : {}, # a dict to populate a MappingFilter (To be confirmed)
    },
    'simple' : { # placeholder for more vanilla QuickPALM/simple ThunderSTORM
        'idnames' : ['frame','int'],
        'delimiter' : ',',
        'comment_char' : '#', # it does not really work to use these the comment char to ignore lines with missing data
        'column_name_mappings' : {  # a dictionary of name substitutions
            'frame' : 't',
            'int' : 'nPhotons',
            'X' : 'x',
            'Y' : 'y',
            'Z' : 'z',
        },
    },
    'thunderstorm' : {
        'idnames' : ['frame','x [nm]'],
        'comment_char' : '#', # it does not really work to use these the comment char to ignore lines with missing data
        'delimiter' : ',',
        'column_name_mappings' : {  # a dictionary of name substitutions
            'frame' : 't',
            'x [nm]' : 'x',
            'y [nm]' : 'y',
            'z [nm]' : 'z',
            'uncertainty_xy [nm]' : 'error_x',
            'uncertainty_z [nm]' : 'error_z',
            'intensity [photon]' : 'nPhotons',
            'sigma1 [nm]' : 'sigx',
            'sigma2 [nm]' : 'sigy',
            'sigma [nm]' : 'sig',
        },
    },
    'visp_3d' : {
        'idnames' : ['column_4'],
        'delimiter' : '\t',
        'ext': '.3d',
        'column_name_mappings' : {
            'column_0': 'x',
            'column_1': 'y',
            'column_2': 'z',
            'column_3': 'A',
            'column_4': 't'
        }
    },
    'visp_3dlp' : {
        'idnames' : ['column_7'],
        'delimiter' : '\t',
        'ext': '.3dlp',
        'column_name_mappings' : {
            'column_0': 'x',
            'column_1': 'y',
            'column_2': 'z',
            'column_3': 'error_x',
            'column_4': 'error_y',
            'column_5': 'error_z',
            'column_6': 'A',
            'column_7': 't'
        }
    },
    'default' : {
        'column_name_mappings' : {},
    },
}

requiredNames = {'x':'x position [nm]',
                    'y':'y position [nm]'}


def parse_csv_header(filename):
    n = 0
    commentLines = []
    dataLines = []
    headerNameLines = []

    def is_header_candidate(line, delims):
        guessedDelim = guess_delim(line,delims)
        if guessedDelim is not None:
            return not isnumber(line.split(guessedDelim)[0])
        else:
            return False

    def guess_delim(line,delims):
        maxItems = 1
        guessedDelim = None
        
        for delim in delims:
            # simple heuristic that the proper delimiter will produce more items when used for split
            if len(line.split(delim)) > maxItems:
                maxItems = len(line.split(delim))
                guessedDelim = delim

        return guessedDelim
    
    fid = open(filename, 'r')

    # NB - we could previously read whitespace-delimited data where the whitespace was not a tab
    # This is probably safer, but could result in regressions. 
    delims = [',','\t']
    delim = None # default

    while n < 10: # only look at first 10 data lines max
        line = fid.readline()
        if line.startswith('#'): #check for comments
            commentLines.append(line[1:])
        elif is_header_candidate(line, delims): # textual header that is not a comment
            # we later assume that the last comment line contains the column headers
            # this is required, as the default .csv format uses a commented header for the column names.
            commentLines.append(line)
            delim = guess_delim(line,delims)
        else:
            # upon first encounter we may need to check if ','-delimited or '\t'-delimited!
            if delim is None:
                delim = guess_delim(line,delims)
            dataLines.append(line.split(delim))
            n += 1
            
    numCols = len(dataLines[0])
    
    if len(commentLines) > 0 and len(commentLines[-1].split(delim)) == numCols:
        colNamesRaw = [s.strip() for s in commentLines[-1].split(delim)]
        # the stuff below seemed necessary since (1) some names came with byte order mark, or BOM, prepended
        # and (2) some had quotes around the names
        colNames = [name.encode('utf-8').decode('utf-8-sig').strip('"').replace('-', '').rstrip() for name in colNamesRaw]
    else:
        colNames = ['column_%d' % i for i in range(numCols)]

    return colNames, dataLines, len(commentLines), delim


def replace_names(old_names, flavour):
    newnames = []

    repdict = csv_flavours[flavour]['column_name_mappings']

    for name in old_names:
        if name in repdict.keys():
            newname = repdict[name]
        else:
            newname = name
        newnames.append(newname.replace(' ','_')) # in names we replace spaces with underscores
    
    return newnames

def guess_text_options(filename):
    colNames, _, n_skip, delim = parse_csv_header(filename)
    ext = os.path.splitext(filename)[-1]
    flavour = guess_flavour(colNames, delim, ext)

    logger.info('Guessed text file flavour: %s' % flavour)
    colNames = replace_names(colNames, flavour)

    text_options = {'columnnames': colNames,
                    'skiprows' : n_skip,
                    'delimiter' : delim,
                    'invalid_raise' : not csv_flavours[flavour].get('ignore_errors', False)
                    }

    return text_options


def check_required_names(self):
    reqNotDef = [name for name in self.requiredNames.keys() if not name in self.translatedNames]
    if len(reqNotDef) > 0:
        raise RuntimeError("some required names are not defined in file header: " + repr(reqNotDef))
        # this is a stopgap; the proper implementation will need to call into the textimportdialog
        # at this stage

        

def guess_flavour(colNames, delim=None, ext=None):
    # guess csv flavour by matching column names
    fl = None
    for flavour in csv_flavours:
        if (not flavour == 'default') and all(idn in colNames for idn in csv_flavours[flavour]['idnames']):
            if fl is not None:
                raise RuntimeError('Ambiguous flavour database: file matches both %s and %s' % (fl, flavour))
            fl = flavour

    # If this failed, guess csv flavor by matching file type. This means it's a
    # headerless CSV/TXT.
    if (fl is None) and (ext is not None):
        for flavour in csv_flavours:
            if (not flavour == 'default') and ext in csv_flavours[flavour].get('ext',[]):
                if fl is not None:
                    raise RuntimeError('Ambiguous flavour database: file matches both %s and %s' % (fl, flavour))
                if not all(idn in colNames for idn in csv_flavours[flavour]['column_name_mappings'].keys()):
                    continue
                fl = flavour
    
    if (fl is not None) and (delim is not None):
        # consistency check
        if csv_flavours[fl].get('delimiter',',') != delim:
            raise RuntimeError('guessed delimiter %s and flavour delimiter %s do not match' %
                                (delim, csv_flavours[fl].get('delimiter',',')))

    if fl is None:
        fl = 'default'

    return fl




def gen_mdh(self): # generate some metaData that will be passed up the chain
                    # to record some bits of this import
    mdh = MetaDataHandler.NestedClassMDHandler()
    mdh['SMLMImporter.flavour'] = self.flavour
    mdh['SMLMImporter.originalNames'] = self.colNames
    mdh['SMLMImporter.translatedNames'] = self.translatedNames
    if len(self.commentLines) > 0:
        mdh['SMLMImporter.commentLines'] = self.commentLines 
    
    self._mdh = mdh
    
