from . import clusterIO
import os


def upload_directory(in_dir, target_dir, serverfilter=clusterIO.local_serverfilter):
    if not os.path.isdir(in_dir):
        raise RuntimeError('Expected to be given the name of a directory to upload')
    
    target_dir = target_dir.lstrip('/')
    
    def _visit(arg, dirname, names):
        rel_path = dirname[len(in_dir):]
        
        for name in names:
            full_name = os.path.join(dirname, name)
            if os.path.isfile(full_name):
                print('uploading %s' % full_name)
                with open(full_name, 'rb') as f:
                    clusterIO.put_file('/'.join([target_dir, rel_path, name]), f.read(), serverfilter)
        
        #print dirname[len(in_dir):] #, names
    
    os.path.walk(in_dir, _visit, None)
    
    
def upload_glob(in_glob, target_dir, serverfilter=clusterIO.local_serverfilter):
    import glob
    inputs = glob.glob(in_glob)

    target_dir = target_dir.lstrip('/')
    
    for name in inputs:
        if os.path.isfile(name):
            print('uploading %s' % name)
            with open(name, 'rb') as f:
                clusterIO.put_file('/'.join([target_dir, os.path.basename(name)]), f.read(), serverfilter)
        elif os.path.isdir(name):
            upload_directory(name, '/'.join([target_dir, os.path.basename(name)]), serverfilter)


if __name__ == '__main__':
    import sys
    
    try:
        input_glob, target_dir = sys.argv[1:]
        
        upload_glob(input_glob, target_dir)
    except IndexError:
        print('Usage: clusterUpload input_pattern target_dir_on_cluster')