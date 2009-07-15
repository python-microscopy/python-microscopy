import Image
import subprocess

def writeTiff(im, outfile):
    command = ["tiffcp"]
    # add options here, if any (e.g. for compression)

    im = im.astype('uint16')

    for i in range(im.shape[2]):
        framefile = "/tmp/frame%d.tif" % i

        Image.fromarray(im[:,:,i].squeeze(), 'I;16').save(framefile)
        command.append(framefile)

    command.append(outfile)
    subprocess.call(command)

    # remove frame files here
    subprocess.call('rm /tmp/frame*.tif', shell=True)
