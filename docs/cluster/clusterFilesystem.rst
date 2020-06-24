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
machine with storage and a client library (clusterIO) presents a merge of the directory structures on all of the nodes.
In this sense, the file system behaves very much like linux UnionFS. Writes are performed using the HTTP PUT verb, to the
node whose prior write time is earliest (resulting in a crude form of load balancing).

Cluster nodes are automatically discovered using the mDNS (zeroconf) protocol. Nodes announce themselves using the "

.. rubric:: Footnotes

.. [#aggregate] For .hdf and .txt files, the file system also supports an atomic **append** operation through special
    `_aggregate` urls. Appends made using the `_aggregate` system are not guaranteed to be processed in order, so the
    inclusion of an index key in the records to permit re-ordering in postprocessing is recommended if order is important.


