from django.shortcuts import render


# Create your views here.

def status(request):
    from PYME.cluster.status import get_polled_status
    nodes = get_polled_status()

    total_storage = 0
    free_storage = 0

    for i, node in enumerate(nodes):
        nodes[i]['percent_free'] = int(100*float(node['Disk']['free'])/node['Disk']['total'])
        total_storage += node['Disk']['total']
        free_storage += node['Disk']['free']

        nodes[i]['percent_mem_free'] = 100 - node['MemUsage']['percent']

    context = {'storage_nodes' : nodes, 'total_storage' : total_storage,
               'free_storage' : free_storage, 'used_storage' : total_storage-free_storage,
               'percent_total_free' : int(100*float(free_storage+1)/(total_storage+1))}

    if '_ajax' in request.GET.keys():
        return render(request, 'clusterstatus/status_dash_content.html', context)
    else:
        return render(request, 'clusterstatus/status_dash.html', context)


def load(request):
    from PYME.cluster.status import get_polled_status
    
    nodes = get_polled_status()

    total_storage = 0
    free_storage = 0

    context = {'storage_nodes' : nodes}


    return render(request, 'clusterstatus/load_content.html', context)

_numCompleted = {}
_lastTime = 0

def queues(request):
    from PYME.cluster import distribution
    #from PYME.IO import clusterIO
    distributors = distribution.getDistributorInfo()
    distNodes = distribution.getNodeInfo()

    #nodes = clusterIO.getStatus()
    
    queueInfo = []
    
    for distName, distURL in distributors.items():
        queues = distribution.getQueueInfo(distURL)
        keys = sorted(queues.keys())
        queueInfo.append({'name':distName, 'queues': [(k, queues[k]) for k in keys]})


    #print queueInfo

    context = {'distributors': distributors, 'distNodes': distNodes, 'queueInfo':queueInfo}

    if '_ajax' in request.GET.keys():
        return render(request, 'clusterstatus/queue_info_content.html', context)
    else:
        return render(request, 'clusterstatus/queue_info.html', context)