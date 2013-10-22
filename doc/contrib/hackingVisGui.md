## Tricks for the VisGui console

In `VisGui` you also have access to the console which allows you to manipulate some of the analysis data directly, for example, to make extra plots etc.

### Accessing the analysed data

Primary access is via the `pipeline` variable which exposes the various members of the current analysis pipeline.

	t = pipeline['t']
	x = pipeline['x']
	specgram(x)
	scatter(t,x)

### Getting metadata handles

	mdh = mdp.mdh
	duration = mdh['EndTime']-mdh['StartTime']
