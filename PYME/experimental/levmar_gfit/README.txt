This is an early attempt at using an alternative solver for the weighted-least squares fitting,
moved here to get it out of the way and tidy up the displayed fit factories.

The hope was to improve speed by keeping all computations in complied code, rather than the status 
quo where the solver is (compiled) fortran and the model function is compiled c, but everything goes 
through python on each iteration because we are using the scipy minpack wrappings. In practice there
was a small improvement (around 10-20% if I recall correctly), although not enough to justify the 
extra maintenance hassle - (the code needs to be compiled with gcc, and so doesn't work with a standard
windows build pipeline).

It is retained on the off-chance it might be useful if trying to make a SIMD (ie. GPU) implementation 
of the least-squares fitting but is not really suitable for use in it's current state.