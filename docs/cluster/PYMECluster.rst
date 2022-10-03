The PYME Storage and compute cluster architecture
*************************************************

The PYME cluster architecture facilitates scalable parallelized data analysis and
distributed data storage, and can operate either on a cluster or (when performance 
requirements are more modest) on a single computer. It is designed to allow easy
scaling to larger cluster sizes in the future (i.e. you can start with a single node
and add more as needed).

It consists of two key components:
- a distributed storage component.
- a distributed analysis component which is data-locality aware.

In addition there is a web-based user interface to the components.

Whilst the storage and analysis components can be used separately, the best results are 
obtained when they are used together such that the data-locality features of the analysis
scheduling can be used.


.. toctree::
   :maxdepth: 1

   cluster_install
   clusterFilesystem
   cluster_compute
   cluster_ui