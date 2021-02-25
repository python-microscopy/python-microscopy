.. _pymeacquire:

**PYMEAcquire** - Instrument control and simulation
***************************************************

PYMEAcquire is a general purpose microscope control and data acquisition platform
with several features which make it well suited to localisation microscopy. These
include:

* Support for commonly used hardware (eg Andor EMCCD cameras)
* Streaming to disk of **large** data series
* Acquisition *protocols* which allow various interactions with the hardware (eg
  controlling laser power) to occur at predetermined timepoints within an acquisition
* Tight integration with the real-time analysis tools

The basic usage should (hopefully) be relatively straight forward given some
familiarity with other microscope control packages. Certain aspects, however deserve
further description:

.. toctree::
   :maxdepth: 1

   Configuration <ConfiguringPYMEAcquire>
   Basic Usage <AcquireBasicUsage>
   Simple dSTORM/RPM walkthrough <BasicAcquisition>
   Simulator
   Protocols
   Supported Hardware <supported_hardware>


