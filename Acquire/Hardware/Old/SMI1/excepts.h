#ifndef excepts_h
#define excepts_h

#include <string>
using namespace std;

class SMIException
{
public:
	string desc;
	SMIException(string Description){desc = Description;}
};

class MemoryAllocError : public SMIException
{
public:
	MemoryAllocError(string Desc):SMIException(Desc){}
};
class FileIOError : public SMIException
{
public:
	FileIOError(string Desc):SMIException(Desc){}
};

class IndexOutOfBounds : public SMIException
{
public:
	IndexOutOfBounds(string Desc):SMIException(Desc){}
};

#endif