//***********************************************************************************************
//
//    Copyright (c) 1996-1997 Axon Instruments.
//    All rights reserved.
//
//***********************************************************************************************
// MODULE:  FileReadCache.cpp
// PURPOSE: Contains class implementation for CFileReadCache.
// AUTHOR:  BHI  Apr 1998
//

#include "wincpp.hpp"
#include "FileReadCache.hpp"

#if defined(__UNIX__) || defined (__STF__)
	#define max(a,b)   (((a) > (b)) ? (a) : (b))
	#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

//===============================================================================================
// PROCEDURE: CFileReadCache
// PURPOSE:   Constructor. 
//
CFileReadCache::CFileReadCache()
{
   //MEMBERASSERT();

   // Initialize the internal variables.
   m_uItemSize    = 0;                    // Size of all data items (structures)
   m_uItemCount   = 0;                    // Number of items available
   m_llFileOffset = 0;                    // Start offset in the file.
   m_uCacheSize   = 0;
   m_uCacheStart  = 0;
   m_uCacheCount  = 0;
   m_pItemCache.reset(NULL);
}

//===============================================================================================
// PROCEDURE: ~CFileReadCache
// PURPOSE:   Destructor. Closes the temporary file and deletes it.
//
CFileReadCache::~CFileReadCache()
{
   //MEMBERASSERT();
/* #ifdef __UNIX__
   // need to close explicitly:
   m_File.Close();
   remove(m_File.GetFileName());
#else
*/   m_File.Release();
//#endif
//   delete[] m_pItemCache;
//   m_pItemCache = NULL;
}

//===============================================================================================
// PROCEDURE: Initialize
// PURPOSE:   Gets a unique filename and opens it as a temporary file.
//
BOOL CFileReadCache::Initialize(UINT uItemSize, UINT uCacheSize, FILEHANDLE hFile, 
                                LONGLONG llOffset, UINT uItems )
{
   //MEMBERASSERT();
   //ASSERT(uItems > 0);
   m_uItemSize    = uItemSize;
   m_uCacheSize   = min(uCacheSize, uItems);
   m_uItemCount   = uItems;
   m_llFileOffset = llOffset;
   m_File.SetFileHandle(hFile);
   m_uCacheSize   = uCacheSize;
   m_uCacheStart  = 0;
   m_uCacheCount  = 0;
   m_pItemCache.reset(new BYTE[uItemSize * uCacheSize]);
   return (m_pItemCache.get() != NULL);
}

//===============================================================================================
// PROCEDURE: Get
// PURPOSE:   Retrieves one or more ITEMs from the virtualized array.
//
BOOL CFileReadCache::Get( UINT uFirstEntry, void *pvItems, UINT uEntries )
{
   //MEMBERASSERT();
   //ASSERT(m_File.IsOpen());
   //ASSERT(uFirstEntry+uEntries <= GetCount());
   //ASSERT(uEntries > 0);
   BYTE *pItems = (BYTE *)pvItems;
   //ARRAYASSERT(pItems, uEntries * m_uItemSize);

   while (uEntries)
   {
      // Make sure that the first item is in the cache.
      if (!LoadCache(uFirstEntry))
         return FALSE;

      // Calculate which portion of the cache we want.
      UINT uStart = uFirstEntry - m_uCacheStart;
      UINT uCopy  = m_uCacheStart + m_uCacheCount - uFirstEntry;
      if (uCopy > uEntries)
         uCopy = uEntries;

      // Copy the data out.
      UINT uBytes = uCopy * m_uItemSize;
      memcpy(pItems, m_pItemCache.get() + uStart * m_uItemSize, uBytes);

      // Update the pointers.
      pItems      += uBytes;
      uEntries    -= uCopy;
      uFirstEntry += uCopy;
   }
   return TRUE;
}

//===============================================================================================
// PROCEDURE: Get
// PURPOSE:   Get a pointer to one entry in the cache.
//
void *CFileReadCache::Get( UINT uEntry )
{
   //MEMBERASSERT();
   //ASSERT(m_File.IsOpen());
   //ASSERT(uEntry < GetCount());

   // Make sure that the item is in the cache.
   if (!LoadCache(uEntry))
      return NULL;

   // return a pointer to the item.
   return m_pItemCache.get() + (uEntry - m_uCacheStart) * m_uItemSize;
}

//===============================================================================================
// PROCEDURE: LoadCache
// PURPOSE:   If an entry is not in the cache, the cache is reloaded from disk.
//
BOOL CFileReadCache::LoadCache(UINT uEntry)
{
   //MEMBERASSERT();
   //ASSERT(m_File.IsOpen());

   if ((uEntry >= m_uCacheStart) && (uEntry < m_uCacheStart+m_uCacheCount))
      return TRUE;

   // Set the cache at the start of the cache size block that includes the requested item
   m_uCacheStart = uEntry - (uEntry % m_uCacheSize);
   m_uCacheCount = min(m_uItemCount-m_uCacheStart, m_uCacheSize);

   // seek to the start point.
   if (!m_File.Seek(m_uCacheStart * m_uItemSize + m_llFileOffset, FILE_BEGIN))
      return FALSE;

   // Read the items from the file.
   return m_File.Read(m_pItemCache.get(), m_uCacheCount * m_uItemSize);
}

