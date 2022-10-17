.. _clusterfs:

PYME Cluster File System
************************

PYME includes support for a very simple clustered "file system", designed to support real time streaming and analysis of
localization data. The main design goals of this file system are:

- high *write* performance to allow streaming from sCMOS cameras
- good read performance for *node local* data access during analysis
- simple to setup and access

For the sake of simplicity and performance, the "filesystem" does not permit random access within files (**reads and writes
are atomic on the whole file level**) and does not perform the type of expensive cluster-wide locking operations that
would be expected in a general purpose distributed filesystem. This lack of locking is facilitated by the atomic file
reads and writes. With some minor exceptions [#aggregate]_, write operations
are strictly *write once* and write in entirety with **no** modification or deletion.

.. warning::

    The lack of cluster-wide locking means that there is the potential for a race condition when performing file creation
    from multiple processes / computers (creation fails if the file already exists, but the existence test relies on a
    locally cached version of the cluster directory tree, with an expiration time of 2-3s).

    **The onus is on the client programmer to ensure that it is *well behaved* and that writes to the same filename do not
    occur from multiple processes simultaneously.**

Implementation
==============

The underlying filesystem protocol is based on HTTP, allowing existing HTTP protocol libraries to be used and clients to
easily be written in a variety of different languages. At it's most basic level, we simply run an HTTP server on each
machine with storage and a client library (:py:mod:`clusterIO <PYME.IO.clusterIO>`) presents a merge of the directory structures on all of the nodes.
In this sense, the file system behaves very much like linux UnionFS. Writes are performed using the HTTP PUT verb, with
load balancing dependent on the data-locality requirements of the analysis. Load balancing in the (:py:mod:`clusterIO <PYME.IO.clusterIO>`)
library is fairly crude - data is written to the server whose prior write time is earliest, but can be configured in more detail
when using the streaming interfaces.

Cluster nodes are automatically discovered using the mDNS (zeroconf) protocol, using the `_pyme-http._tcp.local.` service type.
Programatic usage is described below.

Startup / installation
======================

- install PYME as a python package
- run :py:mod:`PYMEDataServer <PYME.cluster.HTTPDataServer>` (either using :program:`PYMEClusterOfOne`, or launching individually on each
  cluster node after following the :ref:`configuration instructions <clusterinstall>` )

Data streaming
==============

PYME has two data-streaming interfaces which might be useful from 3rd party acquisition programs

High level streaming to a folder of .pzf on the cluster. 
--------------------------------------------------------

Use :py:class:`PYME.IO.acquisition_backends.ClusterBackend`.This is used for streaming localisation data from
`PYMEAcquire`. It is appropriate for time-series data, ensures suitable
data-locality for localisation analysis, and takes care of compression for you. You are, however, tied to PYMEs data
and metadata models.

.. code-block:: python

    from PYME.IO import acquisition_backends

    # create a backend
    # see PYME.IO.PZFFormat for compression_settings docs
    backend = acquisition_backends.ClusterBackend(series_name, compression_settings= ...) 

    # [optional] Populate metadata
    backend.mdh['voxelsize.x']=0.1
    ...

    # initialize (saves metadata)
    backend.initialize()

    
    #put frames
    while(acquiring):
        frame_data = camera.get_frame(...) # your custom way of getting frame data
        
        # tell the backend to store the frame. This is a lightweight call, with the data
        # placed on a queue to be saved asynchonously
        # the data is compressed and packaged in a pzf wrapper before being saved
        backend.store_frame(frame_num : int, frame_data : np.ndarray)


    # finish up
    backend.finalize()


Note: the same programatic framework can be used for HDF or memory backends. In the case of the cluster backend, the
streamer runs one pushing thread for each node on the cluster, uses persistent sessions, and maintains separate channels
for sending and acknowledgement (hiding round-trip latency). Together this allows throughput to be maximised.

Lower-level streaming
---------------------

Use :py:class:`PYME.IO.cluster_streaming.Streamer`. Appropriate where you need more control of where data ends up on
the cluster or if you want to use a custom data format. Useful for, e.g. large volume tiled imaging applications.

.. code-block:: python

    from PYME.IO import cluster_streaming

    streamer = cluster_streaming.Streamer()

    # put a single file to the cluster. The data is written exactly as provided
    # Actual IO, however, is asynchronous with the file being placed on a queue 
    # and the function returns immediately
    streamer.put(filename : str, data : bytes)


A more complete example of the low-level streaming interface, including the uses of a custom distribution function to
enure data-locality when creating an image pyramid can be found in :py:mod:`PYME.Analysis.distributed_pyramid`


Accessing data on the cluster
=============================

Programatic access from python
------------------------------

Programatic access to data stored on the cluster is facilitated by the :py:mod:`PYME.IO.clusterIO`
module. This mimics several of the IO functions found in the python :py:mod:`os` module, such as
:py:func:`listdir <PYME.IO.clusterIO.listdir>`, :py:func:`isdir <PYME.IO.clusterIO.isdir>`,
:py:func:`exists <PYME.IO.clusterIO.exists>`, 
:py:func:`walk <PYME.IO.clusterIO.walk>`, and :py:func:`stat <PYME.IO.clusterIO.stat>` which are useful
for establishing where files are located on the cluster. In addition to cluster versions
of :py:mod:`os` functions, there are two functions :py:func:`PYME.clusterIO.put_file` and
:py:func:`PYME.clusterIO.get_file` for putting and retrieving files. Unlike the streaming functions
discussed above, these functions block until the operation is complete, making the `put_file()`
method unsuitable for high-performance writing operations.

.. code-block:: python

    from PYME.IO import clusterIO

    # list the root directory on the default cluster (as specified in ~/.PYME/config.yaml)
    # by default, this is PYMEClusterOfOne running on the local machine 
    clusterIO.listdir('/')


    # get a file
    data = clusterIO.get_file('test.tif') # returns a bytes object

    # put a file
    clusterIO.put_file('/path/to/location/on/cluster/test2.tif', data) # where data is a bytes object


It is also possible to get data from another storage cluster running on the same network by specifying the cluster name 
(see config instructions) as the ``serverfilter`` keyword argument in any of the above functions.


PYME-CLUSTER:// URIs
--------------------

A file on the cluster may also be specified by using a ``PYME-CLUSTER://`` schema and cluster-relative
path to any of the standard PYME command line programs or image IO functions. A PYME-CLUSTER URI takes
the following form: ``PYME-CLUSTER://<serverfilter>/path/to/file/on/cluster``, or optionally the shortened 
version ``PYME-CLUSTER:///path/to/file/on/cluster``, (**Note the triple /**) to locate the file across **all** [#tripleslash]_
detected clusters. 


Raw, low-level, HTTP access (other programming languages)
---------------------------------------------------------

Because the cluster is implemented on top of a set of HTTP servers, which simply serve
a given directory on their host, it is possible to access the cluster data from other 
programming languages using standard HTTP requests. When accessing the data in this way,
determining what files are in a given directory (the union of the directory listings of
all the individual servers), and conversely which server to query for a particular file
must be performed by the implementation. Files 
may be added to the cluster using an HTTP `PUT` to one of the servers (load distribution - ie 
deciding which server to put to - is left to the implementer). The HTTP servers which make up the cluster can be discovered
using the mDNS protocol and querying/browsing for the `_pyme-http._tcp.local.` service type.

The following is a brief outline of accessing the cluster using command
line tools (note - you'll need to use an mDNS library and programatic HTTP fetches on windows).

.. code-block:: bash

    # find the servers which make up the cluster 
    # [linux]
    >> avahi-browse _pyme-http._tcp --resolve -t
    + wlxd03745363e91 IPv4 PYMEDataServer [DB3]:DB3 - PID:48168          _pyme-http._tcp      local
    = wlxd03745363e91 IPv4 PYMEDataServer [DB3]:DB3 - PID:48168          _pyme-http._tcp      local
    hostname = [PYMEDataServer\032\091DB3\093\058DB3\032-\032PID\05848168._pyme-http._tcp.local]
    address = [127.0.0.1]
    port = [52688]
    txt = []

    # [mac] This unfortunately requires 3 commands vs 1 on linux
    # [mac] find servers
    >> dns-sd -B _pyme-http._tcp. .
    Browsing for _pyme-http._tcp.
    DATE: ---Wed 14 Sep 2022---
     8:35:43.013  ...STARTING...
    Timestamp     A/R    Flags  if Domain               Service Type         Instance Name
    8:35:43.015  Add        3  14 local.               _pyme-http._tcp.     PYMEDataServer [DB3]:DB3 - PID:61575
    8:35:43.015  Add        2   1 local.               _pyme-http._tcp.     PYMEDataServer [DB3]:DB3 - PID:61575
    ^C
    # [mac] - get port number(s) for services advertised above
    >> dns-sd -L "PYMEDataServer [DB3]:DB3 - PID:61575" _pyme-http._tcp. .
    Lookup PYMEDataServer [DB3]:DB3 - PID:61575._pyme-http._tcp..local
    DATE: ---Wed 14 Sep 2022---
    8:38:40.137  ...STARTING...
    8:38:40.208  PYMEDataServer\032[DB3]:DB3\032-\032PID:61575._pyme-http._tcp.local. can be reached at PYMEDataServer\032[DB3]:DB3\032-\032PID:61575._pyme-http._tcp.local.:55003 (interface 14) Flags: 1
    8:38:40.208  PYMEDataServer\032[DB3]:DB3\032-\032PID:61575._pyme-http._tcp.local. can be reached at PYMEDataServer\032[DB3]:DB3\032-\032PID:61575._pyme-http._tcp.local.:55003 (interface 1)
    # [mac] - get ip addresses for advertised services
    >> dns-sd -G v4 "PYMEDataServer [DB3]:DB3 - PID:61575" 
    DATE: ---Wed 14 Sep 2022---
    9:00:42.860  ...STARTING...
    Timestamp     A/R    Flags if Hostname                               Address                                      TTL
    9:00:42.862  Add 40000002  0 PYMEDataServer\032[DB3]:DB3\032-\032PID:61575. 0.0.0.0                                      108002   No Such Record
    ^C


    # get a directory listing
    # an HTTP GET on a directory returns a JSON dictionary of
    # {filename:[flags, size], ...} for each of the files in the directory.
    # where flags is a bitfield containing 2 possible flags - 0x01 : this is a directory, and 0x02 : this is a dataset (a special type of directory which is expected to contain image frames and metadata)
    # if the file is a directory, the size is the number of files in that directory, otherwise the number of bytes.
    >> curl http://0.0.0.0:55003/
    {".DS_Store":[0,14340],"0\/":[1,16],"1\/":[1,9],"2\/":[1,6],"3\/":[1,4],"72\/":[3,9],"73\/":[3,9],
    "75\/":[3,9],"76\/":[3,9],"david\/":[1,34],"LOGS\/":[1,8],"metadata.json":[0,0],"p2.pyr\/":[3,8],
    "RECIPES\/":[1,3],"t28\/":[1,7],"t29\/":[1,6],"t3.pyr\/":[3,8],"t30\/":[1,6],"t31\/":[1,6],
    "t32\/":[1,7],"t33\/":[1,7],"t34\/":[1,7],"t35\/":[1,7],"t37\/":[1,7],"t38\/":[1,7],"t39\/":[3,9],
    "t4.pyr\/":[3,8],"t40\/":[3,8],"t41\/":[3,8],"t44\/":[1,2],"t45\/":[3,8],"t46\/":[3,8],"t47\/":[3,8],
    "t5.pyr\/":[3,10],"t56\/":[3,8],"t6.pyr\/":[3,8],"t61\/":[3,7],"t62\/":[3,7],"t64\/":[3,7],
    "t65\/":[3,3],"t70\/":[3,7],"t77\/":[3,9],"t8.pyr\/":[3,8],"t80\/":[3,9],"t82\/":[1,6],"t83\/":[3,10],
    "t84\/":[3,11],"test\/":[1,3],"Untitled.png":[0,11388],"Users\/":[1,4]}

    # to find all the elements in a directory, you need to perform the listing
    # on each node of the cluster (as identified by the mDNS entries)
    # and combine the entries

    # to download a file, find which node it is on and use a simple http GET:
    >> curl http://0.0.0.0:55003/Untitled.png -o output.png

    # to upload a file, decide which node to save to and use an HTTP PUT.
    # NB: when using low-level access the onus is on the users software to
    # ensure that data is approximately evenly distributed across nodes 
    >> curl -T /path/to/file.png http://0.0.0.0:55003/somefolder/newfile.png

The command line example above is certainly not the easiest way to implement a client. It is 
mainly shown to reinforce the fact that the protocol is just HTTP and is language agnostic.
In practice, you would probably want to reimplement clusterIO in your language of
choice. If performance is important, this reimplementation should include
caching on directory lookups and a "bypass" mechanism to access data stored on the local node
without making a HTTP request.


Read-only access using UnionFS
------------------------------

With a bit of linux-foo, it is possible to set up read-only access to the 
aggregated cluster storage by taking the following steps:

1) share the data directory on each cluster node using NFS (or potentially SMB)
2) mount all the data directories on a single linux machine
3) use unionfs (or one of the many alternative implementations) to merge the 
   single node mounts into a combined file system.
4) [optional] set up an SMB share so that you can access it from windows and mac machines.

Due to the atomic-write and no-delete assumptions made in other parts of the
software, it is unsafe to set this up for write access.


.. rubric:: Footnotes

.. [#aggregate] For .hdf and .txt files, the file system also supports an atomic **append** operation through special
    `_aggregate` urls. Appends made using the `_aggregate` system are not guaranteed to be processed in order, so the
    inclusion of an index key in the records to permit re-ordering in postprocessing is recommended if order is important.

.. [#tripleslash] The behaviour to take the first file it finds across **all** visible clusters when `serverfilter` is ommitted
    from the URI and replaced with a slash has the potential to be confusing if there are indeed multiple clusters accessible 
    (and advertised). As `PYMEClusterOfOne` only advertises locally by default, this is rarely an issue. When running multiple 
    clusters it is nevertheless recommended to use fully specified URIs including the cluster names. This behaviour will likely 
    be changed in the future such that an omitted `serverfilter` defaults to the `PYME.config` setting, rather than all visible
    clusters. 


