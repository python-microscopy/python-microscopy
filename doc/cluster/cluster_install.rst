.. _cluster_install:

Setting up PYME cluster hardware and software
*********************************************

This document details the steps to set up a linux cluster for high-thoughput analysis of full data rate sCMOS data.
If you want to use the cluster code to analyse data from more conventional PALM/STORM experiments, see the
'Cluster of one' docs as well (to be written).

Recommended Hardware configuration
==================================

For streaming at full frame rate from an sCMOS camera, we recommend the following configuration. A lower spec config
will work if full sCMOS data rate is not needed.

Instrument computer
-------------------

* a modern CPU with at least 4 (ideally 6) physical cores and support for the AVX command set

* >=32GB RAM

* a 10GBE network card for connection to the cluster

* running a 64 bit version of either linux or windows

* it is recommended that this machine not be connected to the institutional network

Network switch
--------------

* a dedicated switch with at least 1 10GB uplink port and sufficient downstream ports for the cluster

* downstream ports may be 1GBE, but backplane bandwidth should be at least 10GB, preferably higher

* In practice this means an 'enterprise class' switch, not the cheapest 10 port switch you can get

Cluster nodes
-------------

Our development cluster has 10 nodes, in general nodes should have:

* a modern CPU

* >= 32 GB RAM

* a 1GBE network connection [#network]_

* enough hard-drives to reach the desired storage capacity. A strong recommendation would be to go with the largest drives
  you can get and to configure them using LVM so that the storage can be easily expanded by adding additional drives

* ideally a (small) SSD boot/OS drive. Python code runs significantly faster off a SSD than a hard drive.

* [optional] a CUDA compatible GPU with compute capacity >=5.2

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

#. Install the python 2.7 version of anaconda

#. Add the ``david_baddeley`` conda channel

#. :code:`conda install python-microscopy`

#. Create ``/etc/PYME/config.yaml`` (or ``~/.PYME/config.yaml``) with the following contents (modified appropriately):

   .. code-block:: yaml

     dataserver-root: "/path/to/directory/to/serve"
     dataserver-filter: ""

   .. note::

     ``dataserver-root`` should point to a directory which will be dedicated to cluster data (not ``home`` or similar)
     and which must be writeable by the PYME user. Anything in this directory will be made visible through the cluster
     file system.

     ``dataserver-filter`` lets you specify a filter that will allow multiple distinct clusters to run on the same network.
     The default value of "" will match all running servers. This is appropriate in the recommended case where the cluster
     is isolated from the general network behind a dedicated switch. If this is not possible, setting ``dataserver-filter``
     is recommended (the typical use case here would be a "Cluster of One" on an acquisition computer for standard low
     throughput analysis).

     **Use with** ``dataserver-filter`` **set to anything other than** ``""`` **is not well tested.**

#. *Optional - enable GPU fitting (PYME will allow CPU fitting in the absence of these steps)*

   #. Install CUDA (PYME will allow CPU based fitting in the absence of CUDA and warpdrive)

   #. Install ``pyme-warp-drive`` following instructions at ``github.com/bewersdorflab/pyme-warp-drive``





On the master/interface node:
-----------------------------

The master node runs 3 extra server processes that do not run on standard cluster nodes - a Web UI to the cluster,
a task scheduler for distributed compute tasks, and, optionally, a webDAV server to permit the cluster to be mapped as
a drive on windows or OSX. It is also reasonable to use the master node as a gateway/proxy into the cluster, in which
case it should have 2 network interfaces. In our installs to date the master node is one of the standard cluster nodes,
just running the extra processes but it could also be a standalone machine.

9. Follow the individual node steps (optionally without configuring the data server if this is not also a storage node)

#. Checkout the PYME source from bitbucket to get the ``clusterUI`` sources. ``clusterUI`` is a Django web app for browsing the cluster.

#. ``conda install`` the ``django`` python module (tested for django=1.8.4, more recent versions might also work)


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

5. *[optional]* Run ``PYMEWebDav`` for the WebDAV server to enable the cluster to be mapped as a network drive on windows
   and mac. The webdav server will bind to port 9090, and has a default **username:password** combo of **test:test**.

   .. warning::

     PYMEWebDav is really buggy, and just barely functional. In order to use it on modern versions of windows you will
     need to set a registry key enabling support for the (insecure) authentication model it uses (googling windows and
     WebDAV turns up the relevant instructions pretty quickly). Look at ``PYME/ParallelTasks/webdav.py`` for info on
     setting custom passwords.

On each node:
-------------
6. Run ``PYMERuleNodeServer`` to launch the distributed analysis clients.

   .. note::

      ``PYMERuleServer`` should be running on the master before the node server is launched. **TODO** - make the nodeserver wait
      for a ruleserver to become available so that startup scripts are more robust.

Spooling data
=============

On the instrument computer
--------------------------

#. Make a development install of PYME following the instructions at http://python-microscopy.org/doc/Installation/InstallationFromSource.html#installationfromsource .

#. Either use the ``PYMEAcquire`` acquisition program, or adapt the code in ``PYME/experimental/dcimgFileChucker.py`` to interface with your acquisition program.



.. rubric:: Footnotes

.. [#network] 1GBE is sufficient if there are enough nodes. On new hardware, it might be possible to get enough
  compute power using fewer nodes and 10 GBE connections should be considered if the number of nodes is < 6. It might
  also be worth considering 10GBE for the 'master' node.