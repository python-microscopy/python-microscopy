#!/usr/bin/python

##################
# shiftDisp.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################




def get_r():
    if tq.getNumberTasksCompleted() > 0:
        t = tq.getCompletedTask()
        xr = []
        yr = []
        xg = []
        yg = []
        for i in range(len(t.results[0])):
            r_r = t.results[0][i]
            r_g = t.results[1][i]
            if not r_r.fitErr == None and not r_g.fitErr == None and r_r.fitErr[1] < 50 and r_g.fitErr[1] < 50:
                xr.append(r_r.x0())
                yr.append(r_r.y0())
                xg.append(r_g.x0())
                yg.append(r_g.y0())
        xr = array(xr)
        yr = array(yr)
        xg = array(xg)
        yg = array(yg)
        clf()
        quiver(xr, yr, (xr-xg), yr-yg)
