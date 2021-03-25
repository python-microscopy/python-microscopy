# fProfile Thread Profiler


The fProfile profiler is used for monitoring the behaviour of a multi-threaded program. It uses the standard python 
profiling hooks to log the entry and exit times of each function in the program. Unlike the standard profiler, it records
every function call separately (rather than collecting a statistical summary of the average time taken over a number of
function calls), and keeps track of which thread these calls took place in. In essence it records the entire call tree of
the process. This has the potential to be massively overwhelming if, e.g. std. library calls were included in the call 
tree, so we limit our scope using a simple pattern(s). Typically this should recognise the particular module you are
profiling and exclude other modules, although it can be helpful to include additional modules on a case-by-case basis.

fProfile consists of three components: 

- the profiler (`fProfile`)itself which simply records entry and exit times of matching functions
while the target program runs.

- a parser, `convertProfile` which reconstructs the per-thread call-trees from the flat function entry and exit times 
in the raw profile output

- a viewer, `viewProfile` which lets you explore the converted call-trees, zooming in on areas of interest.


## Usage

### Step 1: Profiling


To enable profiling, import the fProfile module within your code, create a `ThreadProfiler` object and enable profiling ...
 
 
 ```python
from PYME.util import fProfile

profiler = fProfile.ThreadProfiler()
profiler.profile_on(subs=['PYME',], outfile='somefile.txt')

#code to be profiled ....

# turn profiling off - this detaches the profiler hook, and ensures that the output file is flushed to disk
profiler.profile_off()
```

Pattern matching occurs on the module name, and will match if any of the `subs`trings given in the `subs` parameter can
be found within the module name. The default pattern of `['PYME',]` will match anything in the PYME package.

> **Note:** pattern matching has recently changed to it's current form, having previously used a regex. This however
> incurred too much profiler overhead.

The raw output will be stored in a text file with the name given by the `outfile` parameter.
  

### Step 2: Conversion

Before it can be viewed, the raw profile output has to be converted to call trees. At a command prompt, run the 
following:

```bash
python -m PYME.util.fProfile.convertProfile somefile.txt calltrees.json
```

The will save the thread-resolved calltree and timings in `calltrees.json`


### Step 3: Viewing the profile

Profiles can be explored using a browser based viewer. Ensure that you have internet access and that javascript is 
enabled and then run:

```bash
python -m PYME.util.fProfile.viewProfile
```

This should launch a lightweight webserver and pop up a window in your webbrowser. Open the profile `.json` file with 
the `Choose File` button. You should
see an overview (top) of the process timeline and a zoomed view below. Dragging within the overview sets the display 
bounds of the zoomed view. Each thread is indicated by a coloured horizontal line, with function calls within that thread
appearing as stacked blue boxes immediately below it. Hovering over a box will show the associated function name and 
timing info.