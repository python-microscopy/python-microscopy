"""
This is where the single molecule localisation routines live.

Key components:
  - :py:mod:`PYME.localization.FitFactories` : Fit factories. A fit factory is an implementation of a particular method of
                                            fitting - e.g. lateral Gaussian, astigmatic Gaussian, or interpolated 3D.
                                            There are fit factories for a number of different 2D and 3D modes, as well
                                            as preliminary multi-emitter fitting support.
  - :py:mod:`PYME.localization.remFitBuf` : A contraction of *Remote Fit Buffered*, this is the core code which ensures
                                            massages and buffers the image data, and calls the appropriate fit factory.
                                            It contains most of the logic which is shared amongst different fit methods.
                                            This includes noise models, and handling of sCMOS camera maps.
                                            As implied by the *Remote* in it's name, it is typically executed as a task
                                            running under :py:mod:`PYME.ParallelTasks.taskWorkerZC`.
  - :py:mod:`PYME.localization.ofind` : Single molecule identification / finding code. This works by performing matched
                                        filtering and SNR dependent thresholding.
"""