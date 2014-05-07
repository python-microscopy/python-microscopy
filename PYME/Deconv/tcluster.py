#!/usr/bin/python

##################
# tcluster.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import threading
import scipy.cow
import scipy.cow.sync_cluster
import Queue 
import time

class tTask:
    pass

class ThreadedCluster(scipy.cow.machine_cluster):
    def __init__(self,server_list):        
        """ machine_cluster(slave_list) --> cluster_object
        
            Description
            
                Create a cluster object from a list of slave machines.
                slave_list is a list of 2-tuples of the form (address, port)
                that specify the network address and port where the slave
                interpreters will live and listen.  The address is always
                a string and can be either the machine name or its IP address.
                The port should be an unused port on the slave machine.
                Always use a port number higher than 1024.
            
            Example::
                
                # example 1 using IP addresses
                >>> slave_list = [ ('127.0.0.1',10000), ('127.0.0.1',10001)]
                >>> cluster = scipy.cow.machine_cluster(slave_list)                
                # example 2 using machine names                
                >>> slave_list = [ ('node0',11500), ('node1',11500)]
                >>> cluster = scipy.cow.machine_cluster(slave_list)
        """
        self.workers=[]
        self.worker_by_name={}
        worker_id = 1
        for host,port in server_list:
            # Add the uid here can help with port conflicts, but only works
            # on Unix clusters.  We really need to work out a daemon service
            # model that makes the port mess transparent.
            port = port #+ os.getuid()
            new_worker = tMachine(host,port,worker_id) ### use overridden worker class
            self.workers.append(new_worker)
            self.worker_by_name[host] = new_worker
            worker_id = worker_id + 1
    def loop_send_recv(self,package,loop_data,loop_var):        
        #----------------------------------------------------
        # Now split the loop data evenly among the workers,
        # pack them up as addendums to the original package,
        # and send them off for processing.
        #----------------------------------------------------
        #job_groups = equal_balance(loop_data,len(self.workers))
        #addendums = []
        results = {}
        
        taskQueue = Queue.Queue()
        
        tThreads = []
        Tasks = []
        
        i = 0
        for jobs in loop_data:
            #addendums.append({loop_var:jobs})
            t = tTask()
            t.package = package
            t.addendum = {loop_var:[jobs]}
            #print t.addendum
            t.name = i
            i = i + 1
            
            Tasks.append(t)
            taskQueue.put(t)
            
        for worker in self.workers:
            tH = TaskThread(worker, taskQueue, results, len(Tasks))
            tThreads.append(tH)
            tH.start()
        
        #wThread = WatcherThread(tThreads)
        #wThread.start()
        #wThread.join()
            
        #Wait for all the threads    
        for tH in tThreads:
            tH.join()
            
        #results = self._send_recv(package,addendums)
        # Nothing done here to figure out the output format.
        # It is always returned as a tuple
        # Probably will be handier to have it as a 
        # Numeric array sometimes.
        
        #print results
        #reorder results to reflect task order
        out = []
        for task in Tasks:
            out.append(results[task.name])
        return out
    
    
    

class tMachine(scipy.cow.sync_cluster.standard_sync_client):
    def executeTask(self, task):
        start_time = time.time()
        print(('Starting task: %s on host: %s:%s' % (task.name, self.host, self.port)))
        self.send(task.package, task.addendum)
        ret = self.recv()
        end_time = time.time()
        print(('Task: %s completed on host: %s:%s, Elapsed time: %s seconds' % (task.name, self.host, self.port, end_time - start_time)))
        return ret
    
    def is_running(self):
        try:
            # Send a simple command to all workers
            # and wait till they handle it successfully
            self.exec_code("1==1") 
        except ClusterError:
                return 0

class TaskThread(threading.Thread):
    def __init__(self, machine, task_queue, results, num_tasks, timeout = 0):
        self.machine = machine
        self.task_queue = task_queue
        self.results = results
        self.num_tasks = num_tasks
        self.timeout = timeout
        threading.Thread.__init__(self)
        self.killme = False
        
    def run(self):
        while len(self.results) < self.num_tasks:
           try: 
                self.task = self.task_queue.get_nowait()
                print(('%s tasks remaining ' % (self.task_queue.qsize() + 1,))) 
                #if there are no items left then the exception thrown
                #should get us out of the while loop
                try:
                    res = self.machine.executeTask(self.task)
                    self.results[self.task.name] = res
                except Exception, ex: # if something went wrong put task back
                    #pass
                    self.task_queue.put(self.task)
                    self.task = None
                    raise ex
                
           except Queue.Empty:
                time.sleep(1)
                #pass
        #except Timeout:
        #        pass
      
        
class WatcherThread(threading.Thread):
    def __init__(self, task_threads):
        self.task_threads = task_threads
        threading.Thread.__init__(self)
        
    def run(self):
        cond = True
        while cond:
           cond = False
           for t in self.task_threads:
               if t.isAlive():
                   cond = True
                   if not t.machine.is_running():
                       t.exit()
                       if not t.task == None:
                           t.task_queue.put(t.task)
                
           time.sleep(1)
                   