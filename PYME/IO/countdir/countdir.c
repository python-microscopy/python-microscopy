#include <Python.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>



#ifdef __linux__
    #include <sys/syscall.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <stdlib.h>

    #define handle_error(msg) \
        do { perror(msg); } while (0)

    struct linux_dirent64 {
               ino64_t        d_ino;    /* 64-bit inode number */
               off64_t        d_off;    /* 64-bit offset to next structure */
               unsigned short d_reclen; /* Size of this dirent */
               unsigned char  d_type;   /* File type */
               char           d_name[]; /* Filename (null-terminated) */
           };

    #define DIR_BUF_SIZE 5242880 //5MB
#endif

#define FILETYPE_NORMAL             0
#define FILETYPE_DIRECTORY          1 << 0
#define FILETYPE_SERIES             1 << 1 // a directory which is actually a series - treat specially
#define FILETYPE_SERIES_COMPLETE    1 << 2 // a series which has finished spooling



#ifdef __linux__
    /// See http://be-n.com/spw/you-can-list-a-million-files-in-a-directory-but-not-with-ls.html
    int count_files(const char *path)
    {
       int fd, nread;
       int n_files = 0;
       char buf[DIR_BUF_SIZE];
       struct linux_dirent64 *d;
       int bpos;
       char d_type;

       fd = open(path, O_RDONLY | O_DIRECTORY);
       if (fd == -1)
          handle_error("open");

       for ( ; ; ) {
            nread = syscall(SYS_getdents64, fd, buf, DIR_BUF_SIZE);
            if (nread == -1)
                //close(fd);
                handle_error("getdents");

           if (nread == 0)
                break;

            for (bpos = 0; bpos < nread;) {
                d = (struct linux_dirent64 *) (buf + bpos);
                if (d->d_ino!=0)
                {
                    n_files +=1;
                }
                bpos += d->d_reclen;
            }
        }

        close(fd);

        return n_files;
    }

#else

int count_files(const char *path)
{
    DIR *dir;
    struct dirent *ent;
    long count=0;

    dir = opendir(path);

    while((ent = readdir(dir)))
            ++count;

    closedir(dir);

    return count;
}

#endif

static PyObject * dirsize(PyObject *self, PyObject *args)
{
    const char *path;
    long count=0;

    if (!PyArg_ParseTuple(args, "s", &path))
        return NULL;

    Py_BEGIN_ALLOW_THREADS

    count = count_files(path);

    Py_END_ALLOW_THREADS

    return Py_BuildValue("i", count);
}


#define BUF_SIZE 256
static PyObject * file_info(PyObject *self, PyObject *args)
{
    struct stat stat_info;
    struct stat test_stat;
    int status;

    int type=FILETYPE_NORMAL;
    long size;

    const char *path;
    char test_path[BUF_SIZE];

    if (!PyArg_ParseTuple(args, "s", &path))
        return NULL;

    Py_BEGIN_ALLOW_THREADS

    status = stat(path, &stat_info);

    if (status == 0)
    {
        if (S_ISDIR(stat_info.st_mode))
        {
            type |= FILETYPE_DIRECTORY;

            //test for presence of metadata file
            strncpy(test_path, path,BUF_SIZE - 1);
            strncat(test_path, "/metadata.json", BUF_SIZE - strlen(test_path) -1);

            if (stat(test_path, &test_stat) == 0) type |= FILETYPE_SERIES;

            //test for presence of events file
            strncpy(test_path, path, BUF_SIZE - 1);
            strncat(test_path, "/events.json", BUF_SIZE - strlen(test_path) -1);

            if (stat(test_path, &test_stat) == 0) type |= FILETYPE_SERIES_COMPLETE;

            size = count_files(path);
        } else
        {
            size = stat_info.st_size;
        }
    }

    Py_END_ALLOW_THREADS

    if (status != 0) return NULL;

    return Py_BuildValue("i, i", type, size);
}

static PyMethodDef countdirMethods[] = {
    {"dirsize",  dirsize, METH_VARARGS,
     "count the number of files in a directory."},
     {"file_info",  file_info, METH_VARARGS,
     "get info for a specific directory."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "countdir",     /* m_name */
        "fast count of nuber of files in directory",  /* m_doc */
        -1,                  /* m_size */
        countdirMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_countdir(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);

    return m;
}
#else

PyMODINIT_FUNC initcountdir(void)
{
    (void) Py_InitModule("countdir", countdirMethods);
}

#endif