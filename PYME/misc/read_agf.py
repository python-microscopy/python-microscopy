#! /usr/bin/env python
## This file reads a Zemax glass catalog file and creates a collection of NumPy
## variables containing the glass data. Later, you can create a Python interface
## to grab any or all of these variables according to a called function.
#
#From: http://www.owlnet.rice.edu/~tt3/Algorithms.html
import os, glob, sys
from numpy import *

def read_cat_from_lines(lines, catalog_name = None):
    #glassnames = []

    j = 0
    nlines = alen(lines)

#    for j in arange(nlines):
#        line = lines[j]
#        if line.startswith('NM'):
#            nm = line.split()
#            glassnames.append((nm[1]).upper())

    #glasscat[catalog_name] = dict.fromkeys(glassnames)
    
    glasses = {}
    #nglasses = alen(glassnames)
    k = 0

    for j in arange(nlines):
        line = lines[j]
        if line.startswith('NM'):
            nm = line.split()
            glassname = (nm[1]).upper()# glassnames[k]
            
            glasses[glassname] = {}
            k += 1

            glasses[glassname]['name'] = glassname
            glasses[glassname]['catalog'] = catalog_name
            glasses[glassname]['dispform'] = int(nm[2])
            glasses[glassname]['nd'] = float(nm[4])
            glasses[glassname]['vd'] = float(nm[5])
            if alen(nm) >= 8:
                glasses[glassname]['status'] = int(nm[7])
            else:
                status = 0
            if alen(nm) >= 9 and nm.count('-') < 0:
                glasses[glassname]['meltfreq'] = int(nm[8])
            else:
                meltfreq = 0
        if line.startswith('ED'):
            ed = line.split()
            glasses[glassname]['tce'] = float(ed[1])
            glasses[glassname]['density'] = float(ed[3])
            glasses[glassname]['dpgf'] = float(ed[4])
        if line.startswith('CD'):
            cd = line.split()
            cd.remove('CD')
            glasses[glassname]['cd'] = [float(a) for a in cd]
        if line.startswith('TD'):
            td = line.split()
            td.remove('TD')
            glasses[glassname]['td'] = [float(a) for a in td]
        if line.startswith('OD'):
            od = line.split()
            od.remove('OD')
            if od.count('-') > 0: od[od.index('-')] = '-1'
            glasses[glassname]['od'] = [float(a) for a in od]
        if line.startswith('LD'):
            ld = line.split()
            ld.remove('LD')
            glasses[glassname]['ld'] = [float(a) for a in ld]
            
    return glasses
    
def read_cat_from_string(data, catalog_name = None):
    return read_cat_from_lines(data.splitlines(), catalog_name)
    

## =============================================================================
def read_glasscat(glassdir='/home/nh/Zemax/Glasscat/'):
    if os.path.isdir(glassdir):
        ## Get a list of all '*.agf' files in the directory.
        ## Usage:
        ## >>> glasscat = read_zemax.read_glasscat('/home/nh/Zemax/Glasscat/')
        ## >>> nd = glasscat['schott']['N-BK7']['nd']
        glassdir = os.path.dirname(glassdir)
        files = glob.glob(os.path.join(glassdir,'*.[Aa][Gg][Ff]'))

        ## Get the set of catalog names. These keys will initialize the glasscat dictionary.
        catalog_names = []
        nfiles = alen(files)

        for i in arange(nfiles):
            ifile = files[i]
            (head,tail) = os.path.split(ifile)
            catalog_names.append((os.path.splitext(tail)[0]).lower())

        glasscat = dict.fromkeys(catalog_names)

        ## Next, in each catalog we will need to get the list of glass names. This will
        ## initialize the dictionary of glasses.
        for i in arange(nfiles):
            ifile = files[i]
            catalog_name = catalog_names[i]

            f = open(ifile, 'r')
            glassnames = []

            j = 0
            lines = f.readlines()
            nlines = alen(lines)

            for j in arange(nlines):
                line = lines[j]
                if line.startswith('NM'):
                    nm = line.split()
                    glassnames.append((nm[1]).upper())

            glasscat[catalog_name] = dict.fromkeys(glassnames)
            nglasses = alen(glassnames)
            k = 0

            for j in arange(nlines):
                line = lines[j]
                if line.startswith('NM'):
                    nm = line.split()
                    glassname = glassnames[k]
                    glasscat[catalog_name][glassname] = {}
                    k += 1

                    glasscat[catalog_name][glassname]['name'] = glassname
                    glasscat[catalog_name][glassname]['catalog'] = catalog_name
                    glasscat[catalog_name][glassname]['dispform'] = int(nm[2])
                    glasscat[catalog_name][glassname]['nd'] = float(nm[4])
                    glasscat[catalog_name][glassname]['vd'] = float(nm[5])
                    if alen(nm) >= 8:
                        glasscat[catalog_name][glassname]['status'] = int(nm[7])
                    else:
                        status = 0
                    if alen(nm) >= 9 and nm.count('-') < 0:
                        glasscat[catalog_name][glassname]['meltfreq'] = int(nm[8])
                    else:
                        meltfreq = 0
                if line.startswith('ED'):
                    ed = line.split()
                    glasscat[catalog_name][glassname]['tce'] = float(ed[1])
                    glasscat[catalog_name][glassname]['density'] = float(ed[3])
                    glasscat[catalog_name][glassname]['dpgf'] = float(ed[4])
                if line.startswith('CD'):
                    cd = line.split()
                    cd.remove('CD')
                    glasscat[catalog_name][glassname]['cd'] = [float(a) for a in cd]
                if line.startswith('TD'):
                    td = line.split()
                    td.remove('TD')
                    glasscat[catalog_name][glassname]['td'] = [float(a) for a in td]
                if line.startswith('OD'):
                    od = line.split()
                    od.remove('OD')
                    if od.count('-') > 0: od[od.index('-')] = '-1'
                    glasscat[catalog_name][glassname]['od'] = [float(a) for a in od]
                if line.startswith('LD'):
                    ld = line.split()
                    ld.remove('LD')
                    glasscat[catalog_name][glassname]['ld'] = [float(a) for a in ld]

        f.close()
    elif os.path.isfile(glassdir):
        catalog_filename = glassdir
        (head,tail) = os.path.split(catalog_filename)
        catalog_name = (os.path.splitext(tail)[0]).lower()
        if not os.path.isfile(catalog_filename):
            catalog_name = catalog_name.upper()

        f = open(catalog_filename, 'r')
        glassnames = []

        j = 0
        lines = f.readlines()
        nlines = alen(lines)

        for j in arange(nlines):
            line = lines[j]
            if line.startswith('NM'):
                nm = line.split()
                glassnames.append((nm[1]).upper())

        glasscat = dict([(catalog_name, dict.fromkeys(glassnames))])
        nglasses = alen(glassnames)
        k = 0

        for j in arange(nlines):
            line = lines[j]
            if line.startswith('NM'):
                nm = line.split()
                glassname = glassnames[k]
                glasscat[catalog_name][glassname] = {}
                k += 1

                glasscat[catalog_name][glassname]['name'] = glassname
                glasscat[catalog_name][glassname]['catalog'] = catalog_name
                glasscat[catalog_name][glassname]['dispform'] = int(nm[2])
                glasscat[catalog_name][glassname]['nd'] = float(nm[4])
                glasscat[catalog_name][glassname]['vd'] = float(nm[5])
                if alen(nm) >= 8:
                    glasscat[catalog_name][glassname]['status'] = int(nm[7])
                else:
                    status = 0
                if alen(nm) >= 9 and nm.count('-') < 0:
                    glasscat[catalog_name][glassname]['meltfreq'] = int(nm[8])
                else:
                    meltfreq = 0
            if line.startswith('ED'):
                ed = line.split()
                glasscat[catalog_name][glassname]['tce'] = float(ed[1])
                glasscat[catalog_name][glassname]['density'] = float(ed[3])
                glasscat[catalog_name][glassname]['dpgf'] = float(ed[4])
            if line.startswith('CD'):
                cd = line.split()
                cd.remove('CD')
                glasscat[catalog_name][glassname]['cd'] = [float(a) for a in cd]
            if line.startswith('TD'):
                td = line.split()
                td.remove('TD')
                glasscat[catalog_name][glassname]['td'] = [float(a) for a in td]
            if line.startswith('OD'):
                od = line.split()
                od.remove('OD')
                if od.count('-') > 0: od[od.index('-')] = '-1'
                glasscat[catalog_name][glassname]['od'] = [float(a) for a in od]
            if line.startswith('LD'):
                ld = line.split()
                ld.remove('LD')
                glasscat[catalog_name][glassname]['ld'] = [float(a) for a in ld]

        f.close()
    else:
        print(('read_glasscat(): The input filename "' + glassdir + '" is not a valid file or directory.'))
        sys.exit(1)

    return(glasscat)

## =============================================================================
def total_nglasses(glasscat):
    total_nglasses = 0
    for catalog in glasscat:
        for glass in glasscat[catalog]:
            total_nglasses += 1
    return total_nglasses

## =============================================================================
def print_catalog_list(glasscat):
    for cat in glasscat: print(cat)
    return

## =============================================================================
def print_glass_list(glasscat, catalog):
    for glass in glasscat[catalog]: print(glass)
    return

## =============================================================================
def get_all_glasses(glasscat, key):
    keylist = []
    for catalog in glasscat:
        for glass in glasscat[catalog]:
            keylist.append(glasscat[catalog][glass][key])

    if (key != 'name') and (key != 'catalog'): keylist = asarray(keylist)
    return(keylist)

## =============================================================================
def dispersion_data(cd, dispform, ld, wavemin=None, wavemax=None, nwaves=None, sampling_domain='wavelength'):
    ## Zemax's dispersion formulas all use wavelengths in um. So, to compare "ld"
    ## and wavemin,wavemax we first convert the former to nm and then, when done
    ## we convert to um.
    if (wavemin is None):
        lambda_min = ld[0]
    else:
        lambda_min = wavemin / 1000.0
    if (wavemax is None):
        lambda_max = ld[1]
    else:
        lambda_max = wavemax / 1000.0
    if (nwaves is None):
        nwaves = max([10, round(1000.0 * (lambda_max - lambda_min))])

    ## Choose which domain is the one in which we sample uniformly. Regardless
    ## of choice, the returned vector "w" is wavelength in um.
    if (sampling_domain == 'wavelength'):
        w = lambda_min + (lambda_max - lambda_min) * arange(nwaves) / (nwaves - 1.0)
    elif (sampling_domain == 'wavenumber'):
        sigma_min = 1.0 / lambda_max
        sigma_max = 1.0 / lambda_min
        s = sigma_min + (sigma_max - sigma_min) * arange(nwaves) / (nwaves - 1.0)
        w = 1.0 / s
    else:
        print(('The sampling domain "' + sampling_domain + '" defined as input ' \
              + 'to the "dispersion_data()" function is invalid.'))
        sys.exit(1)

    if (dispform == 1):
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + \
                (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8)
        indices = sqrt(formula_rhs)
    elif (dispform == 2): ## Sellmeier1
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                    (cd[4] * w**2 / (w**2 - cd[5]))
        indices = sqrt(formula_rhs + 1.0)
    elif (dispform == 3): ## Herzberger
        L = 1.0 / (w**2 - 0.028)
        indices = cd[0] + (cd[1] * L) + (cd[2] * L**2) + (cd[3] * w**2) + \
                (cd[4] * w**4) + (cd[5] * w**6)
    elif (dispform == 4): ## Sellmeier2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - (cd[2])**2)) + \
                    (cd[3] * w**2 / (w**2 - (cd[4])**2))
        indices = sqrt(formula_rhs + 1.0)
    elif (dispform == 5): ## Conrady
        indices = cd[0] + (cd[1] / w) + (cd[2] / w**3.5)
    elif (dispform == 6): ## Sellmeier3
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                    (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7]))
        indices = sqrt(formula_rhs + 1.0)
    elif (dispform == 7): ## HandbookOfOptics1
        formula_rhs = cd[0] + (cd[1] / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = sqrt(formula_rhs)
    elif (dispform == 8): ## HandbookOfOptics2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = sqrt(formula_rhs)
    elif (dispform == 9): ## Sellmeier4
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) + \
                    (cd[3] * w**2 / (w**2 - cd[4]))
        indices = sqrt(formula_rhs)
    elif (dispform == 10): ## Extended1
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + \
                    (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8) + \
                    (cd[6] * w**-10) + (cd[7] * w**-12)
        indices = sqrt(formula_rhs)
    elif (dispform == 11): ## Sellmeier5
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                    (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7])) + \
                    (cd[8] * w**2 / (w**2 - cd[9]))
        indices = sqrt(formula_rhs + 1.0)
    elif (dispform == 12): ## Extended2
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + \
                    (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8) + \
                    (cd[6] * w**4) + (cd[7] * w**6)
        indices = sqrt(formula_rhs)

    ## Convert waves in um back to waves in nm for output.
    return(w*1000.0, indices)

## =============================================================================
def get_index(glassData, wavelength=635.):
    ## Zemax's dispersion formulas all use wavelengths in um. So, to compare "ld"
    ## and wavemin,wavemax we first convert the former to nm and then, when done
    ## we convert to um.
    
    cd = glassData['cd']
    dispform = glassData['dispform']
    
    w = wavelength/1e3

    if (dispform == 1):
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + \
                (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8)
        indices = sqrt(formula_rhs)
    elif (dispform == 2): ## Sellmeier1
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                    (cd[4] * w**2 / (w**2 - cd[5]))
        indices = sqrt(formula_rhs + 1.0)
    elif (dispform == 3): ## Herzberger
        L = 1.0 / (w**2 - 0.028)
        indices = cd[0] + (cd[1] * L) + (cd[2] * L**2) + (cd[3] * w**2) + \
                (cd[4] * w**4) + (cd[5] * w**6)
    elif (dispform == 4): ## Sellmeier2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - (cd[2])**2)) + \
                    (cd[3] * w**2 / (w**2 - (cd[4])**2))
        indices = sqrt(formula_rhs + 1.0)
    elif (dispform == 5): ## Conrady
        indices = cd[0] + (cd[1] / w) + (cd[2] / w**3.5)
    elif (dispform == 6): ## Sellmeier3
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                    (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7]))
        indices = sqrt(formula_rhs + 1.0)
    elif (dispform == 7): ## HandbookOfOptics1
        formula_rhs = cd[0] + (cd[1] / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = sqrt(formula_rhs)
    elif (dispform == 8): ## HandbookOfOptics2
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) - (cd[3] * w**2)
        indices = sqrt(formula_rhs)
    elif (dispform == 9): ## Sellmeier4
        formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) + \
                    (cd[3] * w**2 / (w**2 - cd[4]))
        indices = sqrt(formula_rhs)
    elif (dispform == 10): ## Extended1
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + \
                    (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8) + \
                    (cd[6] * w**-10) + (cd[7] * w**-12)
        indices = sqrt(formula_rhs)
    elif (dispform == 11): ## Sellmeier5
        formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                    (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7])) + \
                    (cd[8] * w**2 / (w**2 - cd[9]))
        indices = sqrt(formula_rhs + 1.0)
    elif (dispform == 12): ## Extended2
        formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + \
                    (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8) + \
                    (cd[6] * w**4) + (cd[7] * w**6)
        indices = sqrt(formula_rhs)

    ## Convert waves in um back to waves in nm for output.
    return indices

## =============================================================================
def index_data(glass_dict, sampling_domain='wavelength', wavemin=None, wavemax=None, nwaves=None):
    return(dispersion_data(glass_dict['cd'], glass_dict['dispform'], \
                           glass_dict['ld'], wavemin=wavemin, wavemax=wavemax, \
                           nwaves=nwaves, sampling_domain=sampling_domain))

## =============================================================================
def index_coeffs(wavemin, wavemax, nwaves, cd, dispform, ld, basis='Taylor', \
                              sampling_domain='wavelength'):
    ## Generate a vector of wavelengths in nm, with samples every 1 nm.
    (waves, indices) = dispersion_data(cd, dispform, ld, sampling_domain=sampling_domain)

    if (wavemin < waves.min()):
        print(('index_coeffs(): The minimum wavelength (' + str(wavemin) +
              'nm)\n is outside the transmission range given for the glass (' +
              str(waves.min()) + '--' + str(waves.max()) + ' nm)'))
        return(zeros(8), zeros(alen(waves)), waves)
    if (wavemax > waves.max()):
        print(('index_coeffs(): The maximum wavelength (' + str(wavemax) +
              'nm)\n is outside the transmission range given for the glass (' +
              str(waves.min()) + '--' + str(waves.max()) + ' nm)'))
        return(zeros(8), zeros(alen(waves)), waves)

    okay = (waves >= wavemin) & (waves <= wavemax)
    waves = waves[okay]
    indices = indices[okay]

    N = 9
    M = alen(waves)

    ## x ranges from -1 to +1.
    x = -1.0 + arange(M) * 2.0 / (M - 1.0)

    if (basis == 'Taylor'):
        H = zeros((M,N))
        H[:,0] = 1.0
        H[:,1] = x
        H[:,2] = x**2
        H[:,3] = x**3
        H[:,4] = x**4
        H[:,5] = x**5
        H[:,6] = x**6
        H[:,7] = x**7
        H[:,8] = x**8
    elif (basis == 'Legendre'):
        H = zeros((M,N))
        H[:,0] = 1.0
        H[:,1] = x
        H[:,2] = 0.5 * (3.0 * x**2 - 1.0)
        H[:,3] = 0.5 * (5.0 * x**3 - 3.0 * x)
        H[:,4] = 0.125 * (35.0 * x**4 - 30.0 * x**2 + 3.0)
        H[:,5] = 0.125 * (63.0 * x**5 - 70.0 * x**3 + 15.0 * x)
        H[:,6] = 0.0625 * (231.0 * x**6 - 315.0 * x**4 + 105.0 * x**2 - 5.0)
        H[:,7] = 0.0625 * (429.0 * x**7 - 693.0 * x**5 + 315.0 * x**3 - 35.0 * x)
        H[:,8] = 0.0078125 * (6435.0 * x**8 - 12012.0 * x**6 + 6930.0 * x**4 - 1260.0 * x**2 + 35.0)
    else:
        print(('The basis chosen (' + basis + ') is invalid for the ' + \
              '"index_coeffs()" function.'))
        sys.exit(3)

    g = indices
    coeffs = dot(linalg.pinv(H), g)
    interp_index = coeffs[0] + (coeffs[1] * x) + (coeffs[2] * H[:,2]) + \
        (coeffs[3] * H[:,3]) + (coeffs[4] * H[:,4]) + (coeffs[5] * H[:,5]) + \
        (coeffs[6] * H[:,6]) + (coeffs[7] * H[:,7]) + (coeffs[8] * H[:,8])

    return(coeffs, interp_index, waves)

## =============================================================================
def insert_dispersion_coeffs(glasscat, wavemin, wavemax, nwaves=None, basis='Taylor', sampling_domain='wavelength'):
    if (nwaves is None):
        nwaves = round(wavemin - wavemax)

    for catalog in glasscat:
        for glass in glasscat[catalog]:
            cd = glasscat[catalog][glass]['cd']
            dispform = glasscat[catalog][glass]['dispform']
            ld = glasscat[catalog][glass]['ld']

            if (ld[0]*1000.0) <= wavemin and (ld[1]*1000.0) >= wavemax:
                (coeffs, interp_index, interp_waves) = index_coeffs(wavemin, wavemax, nwaves, cd, dispform, ld, basis, sampling_domain)
                glasscat[catalog][glass]['coeffs'] = coeffs
                glasscat[catalog][glass]['waveinfo'] = (wavemin, wavemax, basis)
            else:
                glasscat[catalog][glass]['coeffs'] = zeros(9)
                glasscat[catalog][glass]['waveinfo'] = (0.0, 0.0, 0.0, '')
    return(glasscat)

## =============================================================================
def print_glass_report(glass_dict):
    print(('name     = ' + str(glass_dict['name'])))
    print(('nd       = ' + str(glass_dict['nd'])))
    print(('vd       = ' + str(glass_dict['vd'])))
    print(('dispform = ' + str(glass_dict['dispform'])))
    print(('tce      = ' + str(glass_dict['tce'])))
    print(('density  = ' + str(glass_dict['density'])))
    print(('dpgf     = ' + str(glass_dict['dpgf'])))
    print(('cd       = ' + str(glass_dict['cd'])))
    print(('td       = ' + str(glass_dict['td'])))
    print(('od       = ' + str(glass_dict['od'])))
    print(('ld       = ' + str(glass_dict['ld'])))
    if 'coeffs' in glass_dict: print(('coeffs   = ' + str(glass_dict['coeffs'])))
    return

## =============================================================================
def find_glassdict(glasscat, glassname):
    for catalog in glasscat:
        if (glassname in glasscat[catalog]):
            glassdict = glasscat[catalog][glassname]
    if (alen(glassdict) == 0):
        print('That glass name does not exist in the catalog.')
        sys.exit(1)
    return(glassdict)

## =============================================================================
## Reduce all catalogs in the dictionary such that no two glasses are simultaneously
## within (+/- tol1) of key1 and (+/- tol2) of key2.
def reduce_catalog(glasscat, key1, tol1, key2=None, tol2=None):
    nglass = total_nglasses(glasscat)
    names_all = get_all_glasses(glasscat, 'name')
    names_new = names_all[:]
    catalogs_all = get_all_glasses(glasscat, 'catalog')

    keyval1 = get_all_glasses(glasscat, key1)
    if (key2 is not None): keyval2 = get_all_glasses(glasscat, key2)
    items_to_remove = []

    for i in arange(nglass-1):
        skip = (array(where(items_to_remove == i)))[0]
        if (alen(skip) > 0): continue

        for j in arange(i+1, nglass):
            if (key2 is None):
                if (abs(keyval1[j] - keyval1[i]) < tol1):
                    items_to_remove.append(j)
            else:
                if (abs(keyval1[j] - keyval1[i]) < tol1) and (abs(keyval2[j] - keyval2[i]) < tol2):
                    items_to_remove.append(j)


    ## Remove the duplicates from the "remove" list, and then delete those glasses
    ## from the glass catalog.
    items_to_remove = unique(items_to_remove)
    for i in items_to_remove:
        name = names_new[i]
        catalog = catalogs_all[i]
        del glasscat[catalog][name]

    return(glasscat)

## =============================================================================
def simplify_catalog(glasscat, zealous=False):
    ## Remove the "inquiry glasses", the "high-transmission" duplications of
    ## regular glasses, and the "soon-to-be-inquiry" glasses from the Schott catalog.
    names_all = get_all_glasses(glasscat, 'name')
    I_glasses = ['FK3', 'N-SK10', 'N-SK15', 'BAFN6', 'N-BAF3', 'N-LAF3', 'SFL57', 'SFL6', 'SF11', 'N-SF19', 'N-PSK53', 'N-SF64', 'N-SF56', 'LASF35']
    num_i = alen(I_glasses)
    H_glasses = ['LF5HT', 'BK7HT', 'LLF1HT', 'N-SF57HT', 'SF57HT', 'LF6HT', 'N-SF6HT', 'F14HT', 'LLF6HT', 'SF57HHT', 'F2HT', 'K5HT', 'SF6HT', 'F8HT', 'K7HT']
    num_h = alen(H_glasses)
    N_glasses = ['KZFSN5', 'P-PK53', 'N-LAF36', 'UBK7', 'N-BK7']
    num_n = alen(N_glasses)
    Z_glasses = ['N-F2', 'N-LAF7', 'N-SF1', 'N-SF10', 'N-SF2', 'N-SF4', 'N-SF5', 'N-SF57', 'N-SF6', 'N-ZK7', 'P-LASF50', 'P-LASF51', 'P-SF8', 'P-SK58A', 'P-SK60']
    num_z = alen(Z_glasses)
    ZN_glasses = ['CLEARTRAN_OLD', 'ZNS_VIS']
    num_zn = alen(ZN_glasses)

    for n in arange(alen(names_all)):
        remove = False
        for i in arange(num_h):
            if (names_all[n].find(H_glasses[i])) != -1: remove = True
        for i in arange(num_n):
            if (names_all[n] == N_glasses[i]): remove = True
        for i in arange(num_i):
            if (names_all[n] == I_glasses[i]): remove = True
        for i in arange(num_zn):
            if (names_all[n] == ZN_glasses[i]): remove = True
        if zealous:
            for i in arange(num_z):
                if (names_all[n] == Z_glasses[i]): remove = True
        if remove:
            catalog = find_catalog(glasscat, names_all[n])
            del glasscat[catalog][names_all[n]]

    return(glasscat)

## =============================================================================
def find_catalog(glasscat, glassname):
    for catalog in glasscat:
        if glassname in glasscat[catalog].keys():
            catalog1 = catalog
            break
    return(catalog)
