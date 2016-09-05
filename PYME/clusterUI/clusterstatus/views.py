from django.shortcuts import render

# Create your views here.

def status(request):
    from PYME.IO import clusterIO
    nodes = clusterIO.getStatus()

    total_storage = 0
    free_storage = 0

    for i, node in enumerate(nodes):
        nodes[i]['percent_free'] = int(100*float(node['Disk']['free'])/node['Disk']['total'])
        total_storage += node['Disk']['total']
        free_storage += node['Disk']['free']

    context = {'storage_nodes' : nodes, 'total_storage' : total_storage,
               'free_storage' : free_storage, 'used_storage' : total_storage-free_storage,
               'percent_total_free' : int(100*float(free_storage)/total_storage)}

    return render(request, 'clusterstatus/status_dash.html', context)