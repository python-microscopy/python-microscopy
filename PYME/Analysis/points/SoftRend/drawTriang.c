/*
##################
# drawTriang.c
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
 */

//#include "Python.h"
//#include <complex.h>
#include <math.h>
//#include "numpy/arrayobject.h"
#include <stdio.h>

void drawTriangle (double* pImage, int sizeX, int sizeY, double x0, double y0, double x1, double y1, double x2, double y2, float val)
{
    double tmp;
    double y01, y02, y12;
    double m01, m02, m12;
    int x, y;

    // Sort the points so that x0 <= x1 <= x2
    if (x0 > x1) { tmp=x0; x0=x1; x1=tmp; tmp=y0; y0=y1; y1=tmp;}
    if (x0 > x2) { tmp=x0; x0=x2; x2=tmp; tmp=y0; y0=y2; y2=tmp;}
    if (x1 > x2) { tmp=x1; x1=x2; x2=tmp; tmp=y1; y1=y2; y2=tmp;}

    if ((x0 < 0.0) || (x1 < 0.0) || (x2 < 0.0) || (y0 < 0.0) || (y1 < 0.0) || (y2 < 0.0)
            || (x0 >= (double)sizeX) || (x1 >= (double)sizeX) || (x2 >= (double)sizeX)
            || (y0 >= (double)sizeY) || (y1 >= (double)sizeY) || (y2 >= (double)sizeY)
            )
    {
        return; //drop any triangles which extend over the boundaries
    }


    //calculate gradient
    m01 = (y1-y0)/(x1-x0);
    m02 = (y2-y0)/(x2-x0);
    m12 = (y2-y1)/(x2-x1);

    y01 = y0;
    y02 = y0;
    y12 = y1;

    // Draw vertical segments
    for (x = (int)x0; x < (int)x1; x++)
    {
        if (y01 < y02)
        {
            for (y = (int)y01; y < (int)y02; y++)
                pImage[sizeY*x + y] += val;
        }
        else
        {
            for (y = (int)y02; y < (int)y01; y++)
                pImage[sizeY*x + y] += val;
        }

        y01 += m01;
        y02 += m02;

    }


    for (x = (int)x1; x < (int)x2; x++)
    {
        if (y12 < y02)
        {
            for (y = (int)y12; y < (int)y02; y++)
                pImage[sizeY*x + y] += val;
        }
        else
        {
            for (y = (int)y02; y < (int)y12; y++)
                pImage[sizeY*x + y] += val;
        }

        y12 += m12;
        y02 += m02;

    }

}

void drawTetrahedron (double* pImage, int sizeX, int sizeY, int sizeZ, double x0,
        double y0, double z0, double x1, double y1, double z1, double x2, double y2,
        double z2, double x3, double y3, double z3, float val)
{
    double tmp;
    double y01, y02, y03, y12, y13, y23;
    double x01, x02, x03, x12, x13, x23;
    double m01x, m02x, m03x, m12x, m13x, m23x;
    double m01y, m02y, m03y, m12y, m13y, m23y;
    int z;

    // Sort the points so that z0 <= z1 <= z2 <= z3
    if (z0 > z1) { tmp=x0; x0=x1; x1=tmp; tmp=y0; y0=y1; y1=tmp; tmp=z0; z0=z1; z1=tmp;}
    if (z0 > z2) { tmp=x0; x0=x2; x2=tmp; tmp=y0; y0=y2; y2=tmp; tmp=z0; z0=z2; z2=tmp;}
    if (z0 > z3) { tmp=x0; x0=x3; x3=tmp; tmp=y0; y0=y3; y3=tmp; tmp=z0; z0=z3; z3=tmp;}
    if (z1 > z2) { tmp=x1; x1=x2; x2=tmp; tmp=y1; y1=y2; y2=tmp; tmp=z1; z1=z2; z2=tmp;}
    if (z1 > z3) { tmp=x1; x1=x3; x3=tmp; tmp=y1; y1=y3; y3=tmp; tmp=z1; z1=z3; z3=tmp;}
    if (z2 > z3) { tmp=x2; x2=x3; x3=tmp; tmp=y2; y2=y3; y3=tmp; tmp=z2; z2=z3; z3=tmp;}


    if (//(x0 < 0.0) || (x1 < 0.0) || (x2 < 0.0) || (x3 < 0.0)
            //|| (y0 < 0.0) || (y1 < 0.0) || (y2 < 0.0) || (y3 < 0.0)
            (z0 < 0.0) || (z1 < 0.0) || (z2 < 0.0) || (z3 < 0.0)
            //|| (x0 >= (double)sizeX) || (x1 >= (double)sizeX) || (x2 >= (double)sizeX) || (x3 >= (double)sizeX)
            //|| (y0 >= (double)sizeY) || (y1 >= (double)sizeY) || (y2 >= (double)sizeY) || (y3 >= (double)sizeY)
            || (z0 >= (double)sizeZ) || (z1 >= (double)sizeZ) || (z2 >= (double)sizeZ) || (z3 >= (double)sizeZ)
            )
    {
        //printf("drop: %f, %f, %f, %f\n", z0, z1, z2, z3);
        return; //drop any triangles which extend over the boundaries
    }


    //calculate gradient
    m01x = (x1-x0)/(z1-z0);
    m01y = (y1-y0)/(z1-z0);
    m02x = (x2-x0)/(z2-z0);
    m02y = (y2-y0)/(z2-z0);
    m03x = (x3-x0)/(z3-z0);
    m03y = (y3-y0)/(z3-z0);
    m12x = (x2-x1)/(z2-z1);
    m12y = (y2-y1)/(z2-z1);
    m13x = (x3-x1)/(z3-z1);
    m13y = (y3-y1)/(z3-z1);
    m23x = (x3-x2)/(z3-z2);
    m23y = (y3-y2)/(z3-z2);

    y01 = y0;
    x01 = x0;
    y02 = y0;
    x02 = x0;
    y03 = y0;
    x03 = x0;
    y12 = y1;
    x12 = x1;
    y13 = y1;
    x13 = x1;
    y23 = y2;
    x23 = x2;

    //printf("z; %f, %f, %f, %f\n", z0, z1, z2, z3);

    // Draw triangles
    for (z = (int)z0; z < (int)z1; z++)
    {
        drawTriangle(&(pImage[sizeX*sizeY*z]), sizeX, sizeY, x01, y01, x02, y02, x03, y03, val);

        //printf("%f, %f, %f\n", x01, x02, x03);

        y01 += m01y;
        x01 += m01x;
        y02 += m02y;
        x02 += m02x;
        y03 += m03y;
        x03 += m03x;
    }


    for (z = (int)z1; z < (int)z2; z++)
    {
        drawTriangle(&(pImage[sizeX*sizeY*z]), sizeX, sizeY, x12, y12, x02, y02, x13, y13, val);
        drawTriangle(&(pImage[sizeX*sizeY*z]), sizeX, sizeY, x13, y13, x02, y02, x03, y03, val);

        //printf("%f, %f, %f, %f\n", x12, x13, x02, x03);

        y02 += m02y;
        x02 += m02x;
        y03 += m03y;
        x03 += m03x;
        y12 += m12y;
        x12 += m12x;
        y13 += m13y;
        x13 += m13x;
    }

    for (z = (int)z2; z < (int)z3; z++)
    {
        drawTriangle(&(pImage[sizeX*sizeY*z]), sizeX, sizeY, x13, y13, x23, y23, x03, y03, val);
        //printf("%f, %f, %f\n", x13, x23, x03);

        y13 += m13y;
        x13 += m13x;
        y23 += m23y;
        x23 += m23x;
        y03 += m03y;
        x03 += m03x;
    }

}