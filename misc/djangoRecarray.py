import numpy as np



def qsToRecarray(qs):
    '''converts a django query set into a record array'''

    fields = qs.model._meta.fields

    names = []
    formats = []

    for f in fields:
        fmt = np.array(f.to_python('1')).dtype.str
        if fmt == '|S1':
            fmt = '|S50'
        formats.append(fmt)
        names.append(f.name)

    #print formats
    dtype = np.dtype({'names':names, 'formats':formats})
    #print dtype
    
    data = np.zeros(len(qs), dtype)

    for i, r in enumerate(qs):
        for n in names:
            data[n][i] = r.__getattribute__(n)
        #l.append(np.array(vals, dtype))

    #return hstack(l)

    return data


