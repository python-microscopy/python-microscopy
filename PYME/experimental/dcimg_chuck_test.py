import os
import sys
import shutil
import time

N_SERIES = 10000
SERIES_DELAY_S = 10 #delay between series


def main():
    template_loc = sys.argv[1]
    output_loc = sys.argv[2]
    
    template_json = os.path.join(template_loc, 'XXX.json')
    template_dcimg = os.path.join(template_loc, 'XXX.dcimg')
    template_zsteps = os.path.join(template_loc, 'XXX_zsteps.json')
    
    t_last = time.time()
    for i in range(N_SERIES):
        t_now = time.time()
        time.sleep(SERIES_DELAY_S - (t_now - t_last))
        shutil.copy(template_json, os.path.join(output_loc, 'Series%4d.json' % i))
        shutil.copy(template_dcimg, os.path.join(output_loc, 'Series%4d.dcimg' % i))
        shutil.copy(template_zsteps, os.path.join(output_loc, 'Series%4d_zsteps.json' % i))
        print('Sppoled series %d' % i)
        t_last = t_now
    