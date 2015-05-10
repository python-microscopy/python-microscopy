#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################


from taskServerMP import *
taskQueueName = 'PrivateTaskQueues.%s' % compName
def main():
    print('foo')
    profile = False
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        print('profiling')
        profile = True
        from PYME.mProfile import mProfile
        mProfile.profileOn(['taskServerMP.py', 'HDFTaskQueue.py', 'TaskQueue.py'])

    Pyro.config.PYRO_MOBILE_CODE = 0
    Pyro.core.initServer()
    ns=Pyro.naming.NameServerLocator().getNS()
    daemon=Pyro.core.Daemon()
    daemon.useNameServer(ns)

    #check to see if we've got the TaskQueues group
    if not 'PrivateTaskQueues' in [n[0] for n in ns.list('')]:
        ns.createGroup('PrivateTaskQueues')

    #get rid of any previous queue
    try:
        ns.unregister(taskQueueName)
    except Pyro.errors.NamingError:
        pass

    tq = TaskQueueSet()
    uri=daemon.connect(tq,taskQueueName)

    tw = TaskWatcher(tq)
    tw.start()
    try:
        daemon.requestLoop(tq.isAlive)
    finally:
        daemon.shutdown(True)
        tw.alive = False
        
        if profile:
            mProfile.report()
            
#print __name__
if __name__ == '__main__':
    main()