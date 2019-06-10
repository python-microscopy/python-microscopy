##########################################################

"""
Script to test the cluster side of dcimg spooling independantly of labview.
Copies a single series repeatedly to the spool directory whilst incrementally increasing the filename.

Arguments:
      template_loc : a directory containing template files
      output_loc : the directory to spool to

The template directory should contain 3 files:

XXX.json
XXX_Chan001.dcimg
XXX_zsteps.json

"""

import os
import sys
import shutil
import time

N_SERIES = 10000 # number of series to spool
SERIES_DELAY_S = 8 #delay between series


def main():
    template_loc = sys.argv[1]
    output_loc = sys.argv[2]

    start_i = 0
    if len(sys.argv) > 3:
        start_i = int(sys.argv[3])
    
    template_json = os.path.join(template_loc, 'XXX.json')
    template_dcimg = os.path.join(template_loc, 'XXX.dcimg')
    template_zsteps = os.path.join(template_loc, 'XXX_zsteps.json')

    #print template_loc, output_loc

    with open(template_json, 'rb') as f:
        json_data = f.read()

    with open(template_dcimg, 'rb') as f:
        dcimg_data = f.read()

    with open(template_zsteps, 'rb') as f:
        zsteps_data = f.read()
    
    t_last = time.time()
    time.sleep(0.1)
    for i in range(start_i, N_SERIES):
        t_now = time.time()
        delay = SERIES_DELAY_S - max(t_now - t_last, 0)
        print('delay: %f' % delay)
        if delay > 0:
            time.sleep(delay)
        print('spooling series %d' % i)
        #shutil.copy(template_json, os.path.join(output_loc, 'Series%04d.json' % i))
        with open(os.path.join(output_loc, 'Series%04d.json' % i), 'wb') as f:
            f.write(json_data)

        #note that this is slowish (probably due to having to allocate lots of ram). Might be quicker to read in once and write repeatedly.
        #shutil.copy(template_dcimg, os.path.join(output_loc, 'Series%04d_ch001.dcimg' % i))
        with open(os.path.join(output_loc, 'Series%04d_ch001.dcimg' % i), 'wb') as f:
            f.write(dcimg_data)

        #shutil.copy(template_zsteps, os.path.join(output_loc, 'Series%04d_zsteps.json' % i))
        with open(os.path.join(output_loc, 'Series%04d_zsteps.json' % i), 'wb') as f:
            f.write(zsteps_data)

        print('Sppoled series %d' % i)
        t_last = t_now


if __name__ == '__main__':
    main()