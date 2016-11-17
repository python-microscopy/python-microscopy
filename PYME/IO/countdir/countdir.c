#include <Python.h>
#include <stdio.h>
#include <dirent.h>

static PyObject * dirsize(PyObject *self, PyObject *args)
{
    DIR *dir;
    struct dirent *ent;
    const char *path;
    long count=0;

    if (!PyArg_ParseTuple(args, "s", &path))
        return NULL;

    Py_BEGIN_ALLOW_THREADS

    dir = opendir(path);

    while((ent = readdir(dir)))
            ++count;

    closedir(dir);

    Py_END_ALLOW_THREADS

    return Py_BuildValue("i", count);
}

static PyMethodDef countdirMethods[] = {
    {"dirsize",  dirsize, METH_VARARGS,
     "count the number of files in a directory."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initcountdir(void)
{
    (void) Py_InitModule("countdir", countdirMethods);
}