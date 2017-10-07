import os
# this is not really tested at all - I suspect the include file parsing code would need
# a full overhaul
# we ignore it for now
def generate_uc480_h(include_uc480_h, uc480_h_name = None):
        if uc480_h_name is None:
            uc480_h_name = include_uc480_h # default name
	assert os.path.isfile(include_uc480_h), repr(include_uc480_h)
	d = {}
	l = ['# This file is auto-generated. Do not edit!']
	error_map = {}
	f = open (include_uc480_h, 'r')

	def is_number(s):
		try:
			float(s)
			return True
		except ValueError:
			return False
			
	for line in f.readlines():
		if not line.startswith('#define'): continue
		i = line.find('//')
		words = line[7:i].strip().split(None, 2)
		if len (words)!=2: continue
		name, value = words
		if value.startswith('0x'):
			exec '%s = %s' % (name, value) in globals(), locals()
			d[name] = eval(value)
			l.append('%s = %s' % (name, value))
            # elif name.startswith('DAQmxError') or name.startswith('DAQmxWarning'):
			# assert value[0]=='(' and value[-1]==')', `name, value`
			# value = int(value[1:-1])
			# error_map[value] = name[10:]
            # elif name.startswith('DAQmx_Val') or name[5:] in ['Success','_ReadWaitMode']:
			# d[name] = eval(value)
			# l.append('%s = %s' % (name, value))
		elif is_number(value):
			d[name] = eval(value)
			l.append('%s = %s' % (name, value))
		elif value.startswith('UC'):
			print(value)
			d[name] = u'' + value[3:-1]
			l.append('%s = u"%s"' % (name, value[3:-1]))
		elif d.has_key(value):
			d[name] = d[value]
			l.append('%s = %s' % (name, d[value]))
		else:
			d[name] = value
			l.append('%s = %s' % (name, value))
			pass
	l.append('error_map = %r' % (error_map))
	fn = os.path.join(uc480_h_name+'.py')
	print(('Generating %r' % (fn)))
	f = open(fn, 'w')
	f.write ('\n'.join(l) + '\n')
	f.close()
	# print(('Please upload generated file %r to http://code.google.com/p/pylibuc480/issues' % (fn)))
#    else:
#        pass
        #d = uc480_h.__dict__

