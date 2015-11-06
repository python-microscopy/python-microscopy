Tricks for the VisGui console
=============================

In `VisGui` you also have access to the console which allows you to manipulate some of the analysis data directly, for example, to make extra plots etc.

### Accessing the analysed data

Primary access is via the `pipeline` variable which exposes the various members of the current analysis pipeline.

	t = pipeline['t']
	x = pipeline['x']
	specgram(x)
	scatter(t,x)

### Getting metadata handles - option 1

	mdh = mdp.mdh
	duration = mdh['EndTime']-mdh['StartTime']
	
### Getting metadata handles - option 2

Instead of using the metadata panel `mdp` (option 1) it may be more straightforward and would also be applicable in scripts to use the `mdh` member of the `pipeline` object:

	mdh = pipeline.mdh
	
