#!/usr/bin/python

##################
# genCRBcurves.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np

from PYME.Analysis.PSFGen import fourierHNA
from PYME.Analysis import cramerRao


def plotCRB(zs, crb, methodName):
    plt.figure()
    plt.plot(zs, np.sqrt(crb), lw=2)
    plt.ylim(0, 15)
    plt.xlabel('Defocus[nm]')
    plt.ylabel('Localisation precision [nm]')
    plt.legend(['x', 'y', 'z'])
    plt.title(methodName)


def genCurves():


    zs = 25.*np.arange(-40, 40)

    #vanilla widefield PSF
    ps_wf = fourierHNA.GenWidefieldPSF(zs, 70.)

    I = ps_wf[:,:,40].sum()

    def genCRB(ps):
        #calculated for a photon number of 2000, bg of 0
        #I = ps[:,:,ps.shape[2]/2].sum()
        return cramerRao.CalcCramerReoZ(cramerRao.CalcFisherInformZn2(ps*2000/I, 500, voxelsize=[70, 70, 25]))

    crb_wf = genCRB(ps_wf + 1/2e3)


    #astigmatic PSF
    #stength of 1.5 gives approximately 500 nm axial separation
    ps_as = fourierHNA.GenAstigPSF(zs, 70., 2)
    crb_as = genCRB(ps_as + 1/2e3)


    #biplane PSF
    #generate 2 latterally (and axially) displaced psfs and add them into the same image
    #not strictly biplane, but equivalent for calculating CRB etc ...
    def genBiplanePSF(zs, vs=70, sep=500):
        ps1 = fourierHNA.GenShiftedPSF(zs + sep/2., vs)[::-1,:,:]#shift doesn't have parameters, so flip the image to reverse shift
        ps2 = fourierHNA.GenShiftedPSF(zs - sep/2., vs)

        return 0.5*ps1 + 0.5*ps2

    ps_bp = genBiplanePSF(zs, 70, 450)
    crb_bp = genCRB(ps_bp+ 1./2e3)

    #phase ramp PSF
    ps_pr = fourierHNA.GenPRIPSF(zs, 70., 0.5)
    crb_pr = genCRB(ps_pr+ 1./2e3)

    #double helix PSF
    #NB whilst qualitatively similar to published DH-PSFs the selection of vortex
    #locations is somewhat ad-hoc.
    ps_dh = fourierHNA.GenDHPSF(zs, 70., 1.5*array([-1.2, -.9, -.6, -.3, 0, .3, .6, .9, 1.2]))
    crb_dh = genCRB(ps_dh+ 1/2e3)



    #plotCRB(zs, crb_wf, 'Widefield')
    #plotCRB(zs, crb_as, 'Astigmatic')
    #plotCRB(zs, crb_bp, 'Biplane')
    #plotCRB(zs, crb_pr, 'Phase Ramp')
    #plotCRB(zs, crb_dh, 'Double Helix')
    #
    ##plot 3D CRBs as per DH papers
    #figure()
    #plot(zs, sqrt(crb_wf.sum(1)), lw=2)
    #plot(zs, sqrt(crb_as.sum(1)), lw=2)
    #plot(zs, sqrt(crb_bp.sum(1)), lw=2)
    #plot(zs, sqrt(crb_pr.sum(1)), lw=2)
    #plot(zs, sqrt(crb_dh.sum(1)), lw=2)
    #
    #xlabel('Defocus[nm]')
    #ylabel('3D CRB [nm]')
    #ylim(0, 20)
    #legend(['Widefield', 'Astigmatic', 'Biplane', 'Phase Ramp', 'Double Helix'])
    #
    #
    ##plot volume of localisation elipse
    #figure()
    #plot(zs, sqrt(crb_wf).prod(1), lw=2)
    #plot(zs, sqrt(crb_as).prod(1), lw=2)
    #plot(zs, sqrt(crb_bp).prod(1), lw=2)
    #plot(zs, sqrt(crb_pr).prod(1), lw=2)
    #plot(zs, sqrt(crb_dh).prod(1), lw=2)
    #
    #xlabel('Defocus[nm]')
    #ylabel('Volume of Localisation elipse [nm^3]')
    #legend(['Widefield', 'Astigmatic', 'Biplane', 'Phase Ramp', 'Double Helix'])
    #ylim(0, sqrt(crb_pr).prod(1).max()*1.5)

    #crb vs background - use 3D crb over 1um axially
    def crb3DvBG(ps, bgvals):
        crb3d = []
        vol = []
        #I = ps[:,:,ps.shape[2]/2].sum()
        for bg in bgvals:
            #print bg
            crb = genCRB(ps + I*bg/2000.)
            crb3d.append(np.sqrt(crb.sum(1))[20:-20].mean())
            vol.append(np.sqrt(crb).prod(1)[20:-20].mean())

        return np.array(crb3d), np.array(vol)

    bgvals = np.hstack([0, np.logspace(-1, 2, 50)])

    bg_wf = crb3DvBG(ps_wf[13:46, 13:46, :], bgvals)
    bg_as = crb3DvBG(ps_as[13:46, 13:46, :], bgvals)
    bg_bp = crb3DvBG(ps_bp[13:46, 13:46, :], bgvals)
    bg_pr = crb3DvBG(ps_pr[13:46, 13:46, :], bgvals)
    bg_dh = crb3DvBG(ps_dh[13:46, 13:46, :], bgvals)

    plt.figure()
    plt.plot(bgvals, bg_wf[0], lw=2)
    plt.plot(bgvals, bg_as[0], lw=2)
    plt.plot(bgvals, bg_bp[0], lw=2)
    plt.plot(bgvals, bg_pr[0], lw=2)
    plt.plot(bgvals, bg_dh[0], lw=2)

    plt.ylim(0, bg_pr[0].max()*1.5)

    plt.xlabel('Background [photons/pixel]')
    plt.ylabel('3D CRB [nm]')
    plt.legend(['Widefield', 'Astigmatic', 'Biplane', 'Phase Ramp', 'Double Helix'])

    plt.figure()
    plt.plot(bgvals, bg_wf[1])
    plt.plot(bgvals, bg_as[1])
    plt.plot(bgvals, bg_bp[1])
    plt.plot(bgvals, bg_pr[1])
    plt.plot(bgvals, bg_dh[1])
    plt.ylim(0, bg_pr[1].max()*1.5)

    #plot(bgvals, bg_pr[0], 'k', lw=2)
    #plot(bgvals, bg_as[0], 'g', lw=2)

    sc = np.logspace(-2, 0, 50)

    def genScSurface(sc, genfcn):
        crb_vs_sc = []
        for s in sc:
            print(s)
            ps = genfcn(zs, 70., s)
            crb_vs_sc.append(crb3DvBG(ps[13:46, 13:46, :], bgvals)[0])

        return np.array(crb_vs_sc)


    #from mpl_toolkits.mplot3d import Axes3D
    #f = figure()
    #crb_vs_sc = genScSurface(sc, fourierHNA.GenPRIPSF)
    #contour(bgvals, sc, log(crb_vs_sc), 100, cmap=cm.spectral)
    #ax = Axes3D(f)
    #ax.plot_surface(bgvals[None, :]*ones_like(sc)[:,None], sc[:,None]*ones_like(bgvals)[None, :], log(crb_vs_sc), rstride=1, cstride=1, cmap=cm.spectral)
    #draw()