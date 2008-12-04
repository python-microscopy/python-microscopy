//***********************************************************************************************
//
//    Copyright (c) 1996-1997 Axon Instruments.
//    All rights reserved.
//
//***********************************************************************************************
// HEADER:  FileReadCache.HPP
// PURPOSE: Contains class definition for CFileReadCache for creating and maintaining a BufferedArray array.
// AUTHOR:  BHI  Dec 1996
//

#ifndef INC_FILEREADCACHE_HPP
#define INC_FILEREADCACHE_HPP

#include "./../Common/FileIO.hpp"
#include <boost/shared_array.hpp>
//-----------------------------------------------------------------------------------------------
// CFileReadCache class definition

class CFileReadCache
{
private:
   UINT     m_uItemSize;       // Size of all data items (structures)
   CFileIO  m_File;            // DOS file handle to the read-only data.
   UINT     m_uItemCount;      // Number of items available in the file
   LONGLONG m_llFileOffset;    // Start offset in the file.
   UINT     m_uCacheSize;
   UINT     m_uCacheStart;
   UINT     m_uCacheCount;
   boost::shared_array<BYTE>    m_pItemCache;

private:    // Unimplemented default member functions.
   // Declare but don't define copy constructors to prevent use of defaults.
   CFileReadCache(const CFileReadCache &CS);
   const CFileReadCache &operator=(const CFileReadCache &CS);

private:
   BOOL LoadCache(UINT uEntry);

public:     // Public member functions
   CFileReadCache();
   ~CFileReadCache();

   // Call this function to initialize the DTB and the temp file.
   BOOL Initialize(UINT uItemSize, UINT uCacheSize, FILEHANDLE hFile, LONGLONG llOffset, UINT uItems );

   // Get one or more items from the array.
   BOOL Get( UINT uFirstEntry, void *pItem, UINT uEntries );

   // Get a pointer to one entry in the cache.
   void *Get( UINT uEntry );

   // Get the count of items in the file.
   UINT GetCount() const;
};

//===============================================================================================
// PROCEDURE: GetCount
// PURPOSE:   Returns the current count of ITEMs, both cached and written to file.
//
inline UINT CFileReadCache::GetCount() const
{
//   MEMBERASSERT();
   return m_uItemCount;
}

#endif      // INC_FILEREADCACHE_HPP
