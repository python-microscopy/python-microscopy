#include "../Common/axodefn.h"
#include <string.h>

#undef _WINDOWS

int cdecl AXODBG_printf( char *lpsz, ... ) {printf(lpsz);return 0;}
/*********************************************************************
 *           CloseW32Handle (KERNEL.474)
 *           CloseHandle    (KERNEL32.@)
 *
 * Closes a handle.
 *
 * PARAMS
 *  handle [I] Handle to close.
 *
 * RETURNS
 *  Success: TRUE.
 *  Failure: FALSE, check GetLastError().
 */
BOOL WINAPI c_CloseHandle( FILEHANDLE handle )
{
	// returns the opposite of the Windows-function
#ifndef _WINDOWS
	return (!fclose(handle));
#else
	return CloseHandle( handle );
#endif
}

/***********************************************************************
 *           GetFileSize   (KERNEL32.@)
 *
 * Retrieve the size of a file.
 *
 * PARAMS
 *  hFile        [I] File to retrieve size of.
 *  filesizehigh [O] On return, the high bits of the file size.
 *
 * RETURNS
 *  Success: The low bits of the file size.
 *  Failure: INVALID_FILE_SIZE. As this is could also be a success value,
 *           check GetLastError() for values other than ERROR_SUCCESS.
 */
DWORD WINAPI c_GetFileSize( FILEHANDLE hFile, LPDWORD filesizehigh )
{
#ifndef _WINDOWS
    long lSize;
	fpos_t cur;
	if (fgetpos(hFile,&cur)!=0)
		return -1;
	if (fseek (hFile, 0, SEEK_END)!=0)
		return -1;
   	lSize=ftell (hFile);
	if (fsetpos(hFile,&cur)!=0)
		return -1;
    return lSize;
#else
	return GetFileSize( hFile, filesizehigh );
#endif
}

/***********************************************************************
 *              ReadFile                (KERNEL32.@)
 */
BOOL WINAPI c_ReadFile( FILEHANDLE hFile, LPVOID buffer, DWORD bytesToRead,
                        LPDWORD bytesRead, LPOVERLAPPED overlapped )
{
#ifndef _WINDOWS
        *bytesRead=(DWORD)fread(buffer,1,bytesToRead,hFile);
		if ( *bytesRead != bytesToRead)
            return FALSE;
        else
            return TRUE;
#else
	return ReadFile( hFile, buffer, bytesToRead, bytesRead, overlapped );
#endif
}

/***********************************************************************
 *           SetFilePointer   (KERNEL32.@)
 */

DWORD WINAPI c_SetFilePointer( FILEHANDLE hFile, LONG distance, LONG *highword, DWORD method )
{
#ifndef _WINDOWS
    long     res;
    short    origin = 0;

    switch (method)
    {
            case FILE_BEGIN : origin = SEEK_SET;                    /* start of file */
                     break;
            case FILE_CURRENT : origin = SEEK_CUR; /* current position of file pointer */
                     break;
            case FILE_END : origin = SEEK_END;                      /* end of file */
                     break;
    }
    res = fseek (hFile, distance, origin);                 /* stdio read */
    return (DWORD) ftell(hFile);
#else
	return SetFilePointer( hFile, distance, highword, method );
#endif
}
#ifndef _WINDOWS
/*********************************************************************
 *		_splitpath (NTDLL.@)
 *
 * Split a path into its component pieces.
 *
 * PARAMS
 *  inpath [I] Path to split
 *  drv    [O] Destination for drive component (e.g. "A:"). Must be at least 3 characters.
 *  dir    [O] Destination for directory component. Should be at least MAX_PATH characters.
 *  fname  [O] Destination for File name component. Should be at least MAX_PATH characters.
 *  ext    [O] Destination for file extension component. Should be at least MAX_PATH characters.
 *
 * RETURNS
 *  Nothing.
 */
void /*CSH __cdecl */ _splitpath(const char* inpath, char * drv, char * dir,
                        char* fname, char * ext )
{
    const char *p, *end;

    if (inpath[0] && inpath[1] == ':')
    {
        if (drv)
        {
            drv[0] = inpath[0];
            drv[1] = inpath[1];
            drv[2] = 0;
        }
        inpath += 2;
    }
    else if (drv) drv[0] = 0;

    /* look for end of directory part */
    end = NULL;
    for (p = inpath; *p; p++) if (*p == '/' || *p == '\\') end = p + 1;

    if (end)  /* got a directory */
    {
        if (dir)
        {
            memcpy( dir, inpath, end - inpath );
            dir[end - inpath] = 0;
        }
        inpath = end;
    }
    else if (dir) dir[0] = 0;

    /* look for extension: what's after the last dot */
    end = NULL;
    for (p = inpath; *p; p++) if (*p == '.') end = p;

    if (!end) end = p; /* there's no extension */

    if (fname)
    {
        memcpy( fname, inpath, end - inpath );
        fname[end - inpath] = 0;
    }
    if (ext) strcpy( ext, end );
}

/*********************************************************************
 *                  _strnicmp   (NTDLL.@)
 */
int /*CSH __cdecl*/ _strnicmp( LPCSTR str1, LPCSTR str2, size_t n )
{
    return strncasecmp( str1, str2, n );
}

/*********************************************************************
 *		_makepath (MSVCRT.@)
 *
 * Create a pathname.
 *
 * PARAMS
 *  path      [O] Destination for created pathname
 *  drive     [I] Drive letter (e.g. "A:")
 *  directory [I] Directory
 *  filename  [I] Name of the file, excluding extension
 *  extension [I] File extension (e.g. ".TXT")
 *
 * RETURNS
 *  Nothing. If path is not large enough to hold the resulting pathname,
 *  random process memory will be overwritten.
 */
void cdecl _makepath(char * path, const char * drive,
                     const char *directory, const char * filename,
                     const char * extension)
{
    char ch;

//    TRACE("(%s %s %s %s)\n", debugstr_a(drive), debugstr_a(directory),
//          debugstr_a(filename), debugstr_a(extension) );

    if ( !path )
        return;

    path[0] = '\0';
    if (drive && drive[0])
    {
        path[0] = drive[0];
        path[1] = ':';
        path[2] = 0;
    }
    if (directory && directory[0])
    {
        strcat(path, directory);
        ch = path[strlen(path)-1];
        if (ch != '/' && ch != '\\')
            strcat(path,"\\");
    }
    if (filename && filename[0])
    {
        strcat(path, filename);
        if (extension && extension[0])
        {
            if ( extension[0] != '.' )
                strcat(path,".");
            strcat(path,extension);
        }
    }
 //   TRACE("returning %s\n",path);
}
#endif

/***********************************************************************
 *             WriteFile               (KERNEL32.@)
 */

BOOL WINAPI c_WriteFile( FILEHANDLE hFile, LPCVOID buffer, DWORD bytesToWrite,
                         LPDWORD bytesWritten, LPOVERLAPPED overlapped )
{
#ifndef _WINDOWS
	*bytesWritten=(DWORD)fwrite(buffer, 1, bytesToWrite, hFile);
	return (*bytesWritten==bytesToWrite);
#else
	return WriteFile( hFile, buffer, bytesToWrite, bytesWritten, overlapped );
#endif
}

/*************************************************************************
 *              CreateFileA              (KERNEL32.@)
 *
 * See CreateFileW.
 */

FILEHANDLE WINAPI c_CreateFileA( LPCSTR filename, DWORD access, DWORD sharing,
                           LPSECURITY_ATTRIBUTES sa, DWORD creation,
                           DWORD attributes, HANDLE templ)
{
#ifndef _WINDOWS
	char    fname[70];          /* To get near variable holding string */
    char*     omode;

    switch (access)                /* use C library constants to set mode */
    {
        case GENERIC_WRITE: omode = "w";
                 break;
        case GENERIC_READ | GENERIC_WRITE: omode = "w+";
                 break;
        default: omode = "r";
                 break;
     }

     strcpy(fname, filename);              /* Get filename in near var */
     return fopen(fname,omode);
#else
	return CreateFileA( filename, access, sharing, sa, creation, attributes, templ);
#endif
}


