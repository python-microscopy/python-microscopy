# convert our nested setup.py files to nested meson.build files suitable for use with subdir() in the root meson.build file.
import os
import sys
import re
import glob


def create_meson_build_file(dir_path, base_path):
    submodule_name = os.path.basename(dir_path)
    subdir = os.path.relpath(dir_path, base_path).replace(os.path.sep, '/')
    if subdir == '.':
        subdir = ''  # If the directory is the base path, we don't need a subdir

    setup_py_path = os.path.join(dir_path, 'setup.py')
    if not os.path.exists(setup_py_path):
        return  # No setup.py file, nothing to convert
    meson_build_path = os.path.join(dir_path, 'meson.build')

    with open(setup_py_path, 'r') as setup_file:
        setup_content = setup_file.read()

    out = f'''
# Boilerplate  to make sure things go in the right place - TODO can we do some of this in the top-level meson.build?
py = import('python').find_installation(pure: false)
np_include_dir = run_command(py,  ['-c',  '"import numpy; print(numpy.get_include())"'], check: true).stdout().strip()
install_dir = py.get_install_dir() / '{subdir}'

'''
    # find python sources and add them to the file
    py_sources = glob.glob(os.path.join(dir_path, '*.py'))
    if py_sources:
        out += 'py_sources = files(\n'
        for src in py_sources:
            out += f'    \'{os.path.relpath(src, dir_path)}\',\n'
        out += ')\n\n'

        out += f'py.install_sources(py_sources, subdir:\'{subdir}\')\n\n'
    
    
    # match all add_subpackage calls, extract the package names, and add them to the m file as subdir() calls, each on a new line
    subpackages = re.findall(r'^\s*config\.add_subpackage\([\'"]([^\'"]+)[\'"]\)', setup_content, re.MULTILINE)

    for subpackage in subpackages:
        if os.path.exists(os.path.join(dir_path, subpackage, 'setup.py')):
            out += f'subdir(\'{subpackage}\')\n'
        else:
            spdir = subpackage.replace('.', '/')

            py_sources = glob.glob(os.path.join(dir_path, spdir, '*.py'))
            if py_sources:
                out += 'py_sources = files(\n'
                for src in py_sources:
                    out += f'    \'{os.path.relpath(src, dir_path)}\',\n'
                out += ')\n\n'

                out += f'py.install_sources(py_sources, subdir:\'{subdir}\')\n\n'


    
    # match all add_data_dir calls, extract the directory names, and add them to the m file as as install_subdir, each on a new line
    data_dirs = re.findall(r'^\s*config\.add_data_dir\([\'"]([^\'"]+)[\'"]\)', setup_content, re.MULTILINE)
    for data_dir in data_dirs:
        
        out += f'install_subdir(\'{data_dir}\', install_dir: \'{subdir}/{data_dir}\')\n'
        

    # add .pxd and .pyx files as data files
    pxd_files = glob.glob(os.path.join(dir_path, '*.pxd'))
    pyx_files = glob.glob(os.path.join(dir_path, '*.pyx'))
    if pxd_files or pyx_files:
        out += 'data_files = files(\n'
        for pxd_file in pxd_files:
            out += f'    \'{os.path.relpath(pxd_file, dir_path)}\',\n'
        for pyx_file in pyx_files:
            out += f'    \'{os.path.relpath(pyx_file, dir_path)}\',\n'
        out += ')\n\n'

        out += f'install_data(data_files, install_dir: \'{subdir}\')\n'

    # find all c extensions (add_extension in setup.py) and convert to py.extension_module
    # FIXME - this looks hard, just save a flag which we can search and manually fix
    extensions = re.findall(r'^\s*config\.add_extension\([\'"]([^\'"]+)[\'"]', setup_content, re.MULTILINE)
    for e in extensions:
        out += 'FIXME - add_extension\n'

    # do the same for Extension() calls
    extensions = re.findall(r'^\s*(?!#).*Extension\((.*)$', setup_content, re.MULTILINE)
    for e in extensions:
        out += 'FIXME - Extension()'

    # write output to meson.build
    with open(meson_build_path,'w') as f:
        print(f'Writing to {meson_build_path}')
        f.write(out)




def find_and_write_meson(PYMEBaseDir=None):
    if PYMEBaseDir is None:
        PYMEBaseDir = os.path.join(os.path.dirname(__file__))


    setup_files = glob.glob(f'{PYMEBaseDir}/PYME/**/setup.py', recursive=True)

    for f in setup_files:
        print(f'creating meson.build for {f}')
        create_meson_build_file(os.path.dirname(f), PYMEBaseDir)


if __name__ == '__main__':
    find_and_write_meson()






