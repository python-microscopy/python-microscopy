.. _cluster_install:

Setting up PYME cluster hardware and software
*********************************************

This document details the steps to set up a linux cluster for data analysis. In this document we focus on high-thoughput
analysis of full data rate (800MB/s) sCMOS data, although the same infrastructure can be used for conventional PALM/STORM
experiments, see :ref:`localization analysis<_localization_analysis>` docs for key differences.

Recommended Hardware configuration
==================================

For streaming at full frame rate from an sCMOS camera, we recommend the following configuration. A lower spec config
will work if full sCMOS data rate is not needed.

Instrument computer
-------------------

* a modern CPU with at least 4 (ideally 6) physical cores and support for the AVX command set

* >=32GB RAM

* a 10GbE network card for connection to the cluster

* running a 64 bit version of either linux or windows

* it is recommended that this machine not be connected to the institutional network

Network switch
--------------

* a dedicated switch with at least 1 10Gb uplink port and sufficient downstream ports for the cluster

* downstream ports may be 1GbE, but backplane bandwidth should be at least 10Gb, preferably higher [#switch]_

Cluster nodes
-------------

Our development cluster has 10 nodes, in general nodes should have:

* a modern CPU

* >= 32 GB RAM

* a 1GbE network connection [#network]_

* enough hard-drives to reach the desired storage capacity. A strong recommendation would be to go with the largest drives
  you can get and to configure them using LVM so that the storage can be easily expanded by adding additional drives.

* ideally a (small) SSD boot/OS drive. Python code runs significantly faster off a SSD than a hard drive.

* a CUDA compatible GPU with compute capacity >=5.2 [can be omitted for moderate throughput analysis]

* 64 bit linux

Software installation
=====================

Setting up the software on the cluster is simple for someone with experience with Linux and Python. We recommend using
a dedicated cluster with limited connection to the external network (nodes need internet access through a proxy /NAT
for software updates, but should not be directly accessible from outside the cluster). This is both for performance
reasons to limit any extraneous network traffic and for security reasons.

.. warning::

    The PYME cluster architecture should be considered insecure and should only be used on a trusted network.

.. note::

    For testing, all the processes that make up the cluster be run on the same computer, and on operating systems other
    than Linux. Much of the development work was done on OSX. Performance on windows hosts is likely to be poor due to
    file system limitations (although this can be partially mitigated by packing individual pzf 'files' inside an HDF5
    container - see "cluster of one" docs).

In short the following steps should be followed:


On each node:
-------------

#. Set up an identical install of Linux (64 bit, we recommend Ubuntu)

#. Create a user for PYME

#. Install the python 3.6 version of miniconda

#. Add the ``david_baddeley`` conda channel

#. :code:`conda install python-microscopy`

#. Create ``/etc/PYME/config.yaml`` (or ``~/.PYME/config.yaml``) with the following contents (modified appropriately):

   .. code-block:: yaml

     dataserver-root: "/path/to/directory/to/serve"
     dataserver-filter: ""

   .. note::

     ``dataserver-root`` should point to a directory which will be dedicated to cluster data (not ``home`` or similar)
     and which must be writeable by the PYME user. Anything in this directory will be made visible through the cluster
     file system. This should ideally be on a hard mount (not an auto-mount under ``/media/``) to ensure that permissions
     don't get screwed up. Note: It should be sufficient for the directory to be writeable by the user, but if in doubt, a directory *owned* by the user is arguably safer. 

     ``dataserver-filter`` lets you specify a filter that will allow multiple distinct clusters to run on the same network.
     The default value of ``""`` will match all running servers. This is appropriate in the recommended case where the cluster
     is isolated from the general network behind a dedicated switch. If this is not possible, setting ``dataserver-filter``
     is recommended (the typical use case here would be a "Cluster of One" on an acquisition computer for standard low
     throughput analysis).

#. *Optional, but strongly recommended for high-throughput - enable GPU fitting (PYME will allow CPU based fitting in the absence of these steps)*

   #. Install CUDA

   #. Install ``pyme-warp-drive`` following instructions on [github](https://github.com/python-microscopy/pyme-warp-drive)

   #. *Optional*, Install ``pyNVML`` so GPU usage can be graphically displayed in the clusterUI web interface. A Python 2
      package is hosted in the ``david_baddeley`` conda channel, and installable with :code:`conda install nvidia-ml-py`.





On the master/interface node:
-----------------------------

The master node runs 3 extra server processes that do not run on standard cluster nodes - a Web UI to the cluster,
a task scheduler for distributed compute tasks, and, optionally, a WebDAV server to permit the cluster to be mapped as
a drive on windows or OSX. It is also reasonable to use the master node as a gateway/proxy into the cluster, in which
case it should have 2 network interfaces. In our installs to date the master node is one of the standard cluster nodes,
just running the extra processes but it could also be a standalone machine.

9. Follow the individual node steps (optionally without configuring the data server if this is not also a storage node)

#. Checkout the PYME source from [github](github.com/python-microscopy/python-microscopy) to get the ``clusterUI`` sources. ``clusterUI`` is a Django web app for browsing the cluster.

#. ``conda install django=1.11``


Running the software
====================

The following steps should be ideally added to init scripts so that the cluster automatically comes back up after a power outage.
For testing purposes, they can be executed manually. All these processes should run as an unprivileged user - in no
circumstances should they run as root.

On each node:
-------------
1. Run ``PYMEDataServer`` to launch the distributed file system server

2. *[optional]* run ``PYMEClusterDup`` to start the data duplication processes

   .. warning::

      PYMEClusterDup is not particularly well tested (we ran out of space on our development cluster and disabled duplication).
      It might not play well with files saved using the ``__aggregate_`` endpoints.


On the master node:
-------------------

3. Run ``PYMERuleServer`` to launch the process which oversees the task distribution

4. Change to the ``clusterUI`` directory in PYME source distribution and run ``python manage.py runserver 9000`` to run
   ``clusterUI`` using the Django builtin development server.

   .. note::

     This will launch a webserver on port 9000 (the django default of 8080 is the default port for the dataserver,
     and so should be avoided). Ideally the ``clusterUI`` app should be deployed behind a webserver  - e.g. apache -
     following the Django instructions, although this currently results in unresolved performance problems.

   .. tip::

     The ``clusterUI`` app can be run from any computer with an interface on the cluster subnet, PYME installed (from
     source), and the same ``dataserver-filter`` entry in the ``config.yaml`` file (see above).

5. *[optional]* Run ``PYMEWebDav`` for the WebDAV server to enable the cluster to be mapped as a network drive on windows
   and mac. The webdav server will bind to port 9090, and has a default **username:password** combo of **test:test**.

   .. warning::

     PYMEWebDav is really buggy, and just barely functional. In order to use it on modern versions of windows you will
     need to set a registry key enabling support for the (insecure) authentication model it uses (googling windows and
     WebDAV turns up the relevant instructions pretty quickly). Look at ``PYME/ParallelTasks/webdav.py`` for info on
     setting custom passwords.

#. *[optional]* Install the svgwrite package to display recipes graphically in the cluster user interface. We do not
   currently maintain a conda package for svgwrite, but it can be found in, e.g., the conda-forge channel.

On each node:
-------------
7. Run ``PYMERuleNodeServer`` to launch the distributed analysis clients.

   .. note::

      ``PYMERuleServer`` should be running on the master before the node server is launched. **TODO** - make the nodeserver wait
      for a ruleserver to become available so that startup scripts are more robust.

Spooling data
=============

On the instrument computer
--------------------------

#. Make a development install of PYME following the instructions at http://python-microscopy.org/doc/Installation/InstallationFromSource.html#installationfromsource .

#. Either use the ``PYMEAcquire`` acquisition program, or adapt the code in ``PYME/experimental/dcimgFileChucker.py`` to interface with your acquisition program.


Troubleshooting
===============

mDNS server advertisements point to loopback, rather than external interface
----------------------------------------------------------------------------

Example symptom: running `PYMEDataServer` logs `INFO:root:Serving HTTP on 127.0.1.1 port 15348 ...` 
rather than an IP address on the cluster network. 

PYME binds to the IP address associated with the host computer name. On linux this
is association is set in the `/etc/hosts` file, which often defaults to

.. code-block::
    127.0.0.1	localhost
    127.0.1.1	<hostname>

This configuration is incomplete, and there are two ways to resolve it:

**The right way:**

* Make sure DNS (e.g. dnsmasq) and, optionally DHCP, are configured correctly within the cluster

* Comment out / delete the ``127.0.1.1 <hostname>`` line in ``/etc/hosts``


**The quick and dirty way:**

**NOTE:** this only works if you have assigned static IPs to your nodes

* Change the ``127.0.1.1 <hostname>`` line to map to your correct static IP


ClusterUI doesn't show files
----------------------------

* Assuming that PYMEDataServer is running this is likely to be a permissions error on the data directory. It's easiest if
  the PYME user owns the directory in question.

* Check that the computer running the ``clusterUI`` app has an interface on the cluster subnet and an appropriate
  ``dataserver-filter`` entry in its ``config.yaml`` file.


getdents: Bad file descriptor
-----------------------------

* We default to using a low-level directory counting function for a speed improvement. We have run
into issues with it on later kernels (Ubuntu 16, 18), which can present as PYMEDataServer failing 
(and e.g. clusterUI timing out when navigating to `<ip:port>/files`). The offending function call can
be avoided by adding the following to ``.PYME/config.yaml``

.. code-block::
    cluster-listing-no-countdir: True


Poor clusterIO performance
--------------------------
If you are seeing timeout or retry errors on `clusterIO.get_file` calls, consider 
disabling the PYME hybrid nameserver (SQL and zeroconf) and using the PYME 
zeroconf nameserver only by adding the following to ``.PYME/config.yaml``

.. code-block::
    clusterIO-hybridns: False

If you are performing sliding-window background estimation during localization
analysis, you may also want to play with the chunksize used in HTTPSpooler on 
the instrument computer (or wherever you are spooling data from). It
defaults to 50 frames; depending on the window sizes you use in analysis you may
consider increasing this to increase data locality (and decrease network I/O).
This can be done in ``.PYME/config.yaml``. For 100 frame chunks, you would have:

.. code-block::
    httpspooler-chunksize: 100


.. rubric:: Footnotes

.. [#switch] In practice this means an 'enterprise class' switch, not the cheapest 10 port switch you can get

.. [#network] 1GbE is sufficient if there are enough nodes. On new hardware, it might be possible to get enough
  compute power using fewer nodes and 10 GbE connections should be considered if the number of nodes is < 6. It might
  also be worth considering 10GbE for the 'master' node.
