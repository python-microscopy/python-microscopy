import numpy as np
def isnumber(s):
    try:
        float(s)
        return True
    except:
        return False


class CSVSMLMReader(object):

    csv_flavours = {
        'elyra' : {
            'idnames' : ['Index','First Frame'], # column names that identify this flavour
            'comment_char' : '#', # it does not really work to use these the comment char to ignore lines with missing data
            'column_name_mappings' : {  # a dictionary of name substitutions
                'Position X [nm]': 'x',
                'Position Y [nm]': 'y',
                'Number Photons': 'nPhotons',
                'First Frame': 't',
                'Chi square': 'nchi2',
                'PSF half width [nm]': 'sigfwhm',
                'Precision [nm]': 'error_x'
            },
            # 'column_translations' : {}, # a dict to populate a MappingFilter (To be confirmed)
        },
        'simple' : { # placeholder for more vanilla QuickPALM/ThunderSTORM, but not quite sure about TS
            'idnames' : ['frame','int'],
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
    }

    def __init__(self, file):
        self.filename = file # do we allow file to be something other than a name?
        self.delimiter = ','
        self.comment_char = '#'
        
    def parse_header_csv(self):
        n = 0
        commentLines = []
        dataLines = []

        fid = open(self.filename, 'r')
        delim = ','

        while n < 10: # only look at first 10 lines max
            line = fid.readline()
            if line.startswith('#'): #check for comments
                commentLines.append(line[1:])
            elif not isnumber(line.split(delim)[0]): #or textual header that is not a comment
                commentLines.append(line)
            else:
                dataLines.append(line.split(delim))
                n += 1
                
        numCommentLines = len(commentLines)
        numCols = len(dataLines[0])
        
        if len(commentLines) > 0 and len(commentLines[-1].split(delim)) == numCols:
            colNamesRaw = [s.strip() for s in commentLines[-1].split(delim)]
            # the stuff below seemed necessary since (1) some names came with byte order mark, or BOM, prepended
            # and (2) some had quotes around the names
            colNames = [name.encode('utf-8').decode('utf-8-sig').strip('"') for name in colNamesRaw]
        else:
            colNames = ['column_%d' % i for i in range(numCols)]

        self.colNames = colNames
        self.dataLines = dataLines
        self.nCommentLines = numCommentLines


    def replace_names(self):
        if self.flavour is not None:
            newnames = []
            repdict = self.csv_flavours[self.flavour]['column_name_mappings']
            for name in self.colNames:
                if name in repdict.keys():
                    newnames.append(repdict[name])
                else:
                    newnames.append(name)
            self.translatedNames = newnames
        else:
            self.translatedNames = self.colNames


    def read_csv_data(self):
        return np.genfromtxt(self.filename,
                         comments=self.comment_char, delimiter=self.delimiter,
                         skip_header=self.nCommentLines, skip_footer=0,
                         names=self.translatedNames, dtype='f4', 
                         missing_values=None, filling_values=np.nan) # may have to use NaN rather than negative no
    

    
    def check_flavour(self):
        self.flavour = None
        for flavour in self.csv_flavours:
            if all(idn in self.colNames for idn in self.csv_flavours[flavour]['idnames']):
                self.flavour = flavour


    def print_flavour(self):
        if self.flavour is None:
            print('Flavour is None (default)')
        else:
            print('Flavour is %s' % self.flavour)

    def read_csv_flavour(self):
        self.parse_header_csv()
        self.check_flavour()
        self.replace_names()
        data = self.read_csv_data()
        col0 = self.translatedNames[0]
        if np.any(np.isnan(data[col0])):
            return data[np.logical_not(np.isnan(data[col0]))] # delete rows with missing values
        else:
            return data
