.. _installationanacondagui:

.. warning::

    These installation instructions are outdated, and will likely not work. Use the
    :ref:`new instructions<installation>` instead. TODO: remove this page and all links.

Completely graphical installation of PYME on OSX
################################################

PYME requires python (version 2.7) and a number of additional scientific packages.
Although it is possible to install all packages individually, and then install PYME,
by far the easiest way to get a system up and running is to install a pre-packaged 
'scientfic python' distribution. `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ 
is one such distribution which is free for both academic and commercial use and includes 
extensive package management capabilities which allow us to easily distribute and update 
PYME on a variety of platforms. We currently provide compiled packages for 64 bit windows, OSX, and Linux. 

STEP 1: Installing Anaconda
===========================

Download and install `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ from 
https://store.continuum.io/cshop/anaconda/. Choose the **64 bit version of Python 2.7**
for your prefered platform. *NB: Anaconda is available in both Python 2.7 and Python 3.x flavours.
PYME will only work with the Python 2.7 version.*

STEP 2: Installing PYME using Anaconda-navigator
================================================

NB: Those who are not afraid of the console would probably be better served by following the instructions in :ref:`installationanaconda`.
These instructions are for those who want a completely graphical method of installation.
At present these instructions only work on OSX (windows to follow soon).

Having installed anaconda, run Anaconda-Navigator from the folder where you installed anaconda. This should give you a
window like the one below.

.. image:: /images/anaconda_navigator.png

Click on the "channels" button to open the channels dialog.

.. image:: /images/anaconda_nav_add_channel.png

Click add to add the channel containing python-microscopy.

.. image:: /images/anaconda_nav_add_channel_david_baddeley.png

Type "david_baddeley" into the channel name box and hit enter / return.

.. image:: /images/anaconda_nav_add_chan_update.png

Click update channels. This should return you the main Anaconda-Navigator window, which should refresh / reload and show
a new 'app', python-microscopy. Click install.

.. image:: /images/anaconda_nav_install_python-microscopy.png



