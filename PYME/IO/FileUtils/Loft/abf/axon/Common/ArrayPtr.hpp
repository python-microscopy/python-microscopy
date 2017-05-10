//***********************************************************************************************
//
//    Copyright (c) 1996-1997 Axon Instruments.
//    All rights reserved.
//
//***********************************************************************************************
// HEADER:  ARRAYPTR.HPP
// PURPOSE: Define the template class CArrayPtr<ITEM>
// AUTHOR:  BHI  Sep 1996
//
//
#ifndef INC_ARRAYPTR_HPP
#define INC_ARRAYPTR_HPP

#pragma once
#include <stdlib.h> 
#include <boost/shared_array.hpp>

#if defined(__UNIX__) || defined(__STF__)
	#define max(a,b)   (((a) > (b)) ? (a) : (b))
	#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif
 
//***********************************************************************************************
// CLASS:   CArrayPtr
// PURPOSE: A smart pointer class for arrays of objects or primitive ITEMs.
//
template<class ITEM>
class CArrayPtr
{
private:    // Private data.
   boost::shared_array<ITEM> m_pArray;

private:    // Prevent copy constructors and operator=().
   CArrayPtr(const CArrayPtr &);
   const CArrayPtr &operator=(const CArrayPtr &);

public:     // Public member functions.

   // Constructors & destructor. See notes below.
   CArrayPtr(ITEM *pItem = NULL);
   CArrayPtr(UINT uCount);
   ~CArrayPtr();
   
   // Allocation and destruction of memory pointed to by this object.
   BOOL Alloc(UINT uCount);
/*   BOOL Realloc(UINT uOldCount, UINT uNewCount, BOOL bZero=FALSE);
   BOOL Clone(const ITEM *pItem, UINT uCount);
*/   void Free();

   // Accessor functions to get at the wrapped array.
   operator ITEM *() const;
   ITEM *Get() const;
/*
   // Destroys the held array and replaces it with another.
   void Set(ITEM *);

   // Non-destructive release of the held pointer.
   ITEM *Release();

   // Zero the held array from 0 for uCount items.
   void Zero(UINT uCount);

   // Sorts the data held in the pointer according to a user-supplied callback.
   void Sort(UINT uCount, int (__cdecl *FnCompare )(const void *elem1, const void *elem2 ));
*/
};

//================================================================================================
// FUNCTION: Constructor
// PURPOSE:  Create a new object that wraps a passed pointer.
// NOTES:    If the passed pointer is non-NULL, it *MUST* have been created on the 
//           heap using the array allocater new[]. 
//           Unfortunately there is no reliable way to ASSERT this.
//
//           The definition of this constructor contains a default parameter of NULL 
//           so that CArrayPtr objects can be created that do not hold anything.
//

template <class ITEM>
inline CArrayPtr<ITEM>::CArrayPtr/*CSH<ITEM>*/(ITEM *pItem)
{
//   MEMBERASSERT();
   ASSERT_NOTONSTACK(pItem);
   m_pArray = pItem;
}

//================================================================================================
// FUNCTION: Constructor
// PURPOSE:  Create a new object and allocate a buffer of the passed size.
//
template <class ITEM>
inline CArrayPtr<ITEM>::CArrayPtr/*CSH<ITEM>*/(UINT uCount)
{
//   MEMBERASSERT();
   m_pArray.reset(NULL);
   Alloc(uCount);
}

//================================================================================================
// FUNCTION: Destructor
// PURPOSE:  Frees a held pointer if non-NULL.
//
template <class ITEM>
inline CArrayPtr<ITEM>::~CArrayPtr/*CSH<ITEM>*/()
{
//   MEMBERASSERT();
//   delete[] m_pArray;
//   m_pArray = NULL;
}

//================================================================================================
// FUNCTION: Alloc
// PURPOSE:  Frees any held pointer and allocates a new array of the wrapped ITEM.
//
template <class ITEM>
inline BOOL CArrayPtr<ITEM>::Alloc(UINT uCount)
{
//   MEMBERASSERT();

   // Free any existing array.
   Free();

   // Return now if nothing is to be allocated.
   if (uCount == 0)
      return TRUE;

   // Allocate the new array.
   m_pArray.reset(new ITEM[uCount]);
   return (m_pArray!=NULL);
}
/*
//================================================================================================
// FUNCTION: Realloc
// PURPOSE:  Reallocates the held pointer keeping the held data.
//
template <class ITEM>
inline BOOL CArrayPtr<ITEM>::Realloc(UINT uOldCount, UINT uNewCount, BOOL bZero)
{
   MEMBERASSERT();
   ARRAYASSERT(m_pArray, uOldCount);

   CArrayPtr<ITEM> pNewArray;
   if (uNewCount && !pNewArray.Alloc(uNewCount))
      return FALSE;

   for (UINT i=0; i<min(uOldCount, uNewCount); i++)
      pNewArray[i] = m_pArray[i];

   // Free any existing array.
   Free();

   // Return now if nothing is to be allocated.
   if (uNewCount == 0)
      return TRUE;

   // Allocate the new array.
   m_pArray = pNewArray.Release();
   return TRUE;
}

//================================================================================================
// FUNCTION: Clone
// PURPOSE:  Frees any held pointer and allocates a new array of the wrapped ITEM.
//           The passed array is then copied into the new buffer.
//
template <class ITEM>
inline BOOL CArrayPtr<ITEM>::Clone(const ITEM *pItem, UINT uCount)
{
   MEMBERASSERT();

   // Reallocate the held pointer.
   if (!Alloc(uCount))
      return FALSE;

   // Check that we have valid parameters.
   ARRAYASSERT(pItem, uCount);

   // If this object was only for wrapping primitive types, a memcpy call
   // would be most efficient for cloning, but this is inappropriate for
   // arrays of objects.
   // memcpy(m_pArray, pItem, uCount*sizeof(ITEM));

   // Use a for loop to copy the array into the new buffer.
   for (UINT i=0; i<uCount; i++)
      m_pArray[i] = pItem[i];

   return TRUE;
}
*/
//================================================================================================
// FUNCTION: Free
// PURPOSE:  De-allocates the held pointer. Benign NOP if no pointer held.
//
template <class ITEM>
inline void CArrayPtr<ITEM>::Free()
{
//   MEMBERASSERT();
   m_pArray.reset(NULL);
}

//================================================================================================
// FUNCTION: Get
// PURPOSE:  Returns the held pointer without giving up ownership of it.
//
template <class ITEM>
inline ITEM *CArrayPtr<ITEM>::Get() const
{
//   MEMBERASSERT();
   return m_pArray.get(); 
}
/*
//================================================================================================
// FUNCTION: Set
// PURPOSE:  Destroys the held array and replaces it with another.
//
template <class ITEM>
inline void CArrayPtr<ITEM>::Set(ITEM *pItem)
{
   ASSERT_NOTONSTACK(pItem);
   Free();
   m_pArray = pItem;
}
*/
//================================================================================================
// FUNCTION: Overloaded cast operator
// PURPOSE:  Returns the held pointer without giving up ownership of it.
//
template <class ITEM>
inline CArrayPtr<ITEM>::operator ITEM *() const
{
//   MEMBERASSERT();
   return Get();
}
/*
//================================================================================================
// FUNCTION: Release
// PURPOSE:  Returns the held pointer, giving up ownership of it.
//
template <class ITEM>
inline ITEM *CArrayPtr<ITEM>::Release()
{
   MEMBERASSERT();
   ITEM *rval = m_pArray;
   m_pArray   = NULL;
   return rval; 
}

//================================================================================================
// FUNCTION: Zero
// PURPOSE:  Zero's out the held pointer from item 0 for uCount items.
//
template <class ITEM>
inline void CArrayPtr<ITEM>::Zero(UINT uCount)
{
   MEMBERASSERT();
   ARRAYASSERT(m_pArray, uCount);
   memset(m_pArray, 0, uCount*sizeof(ITEM));
}

//================================================================================================
// FUNCTION: Sort
// PURPOSE:  Sorts the data held in the pointer according to a user-supplied callback.
//
template <class ITEM>
inline void CArrayPtr<ITEM>::Sort(UINT uCount, int (__cdecl *FnCompare )(const void *elem1, const void *elem2 ))
{
   MEMBERASSERT();
   ARRAYASSERT(m_pArray, uCount);
   qsort(m_pArray, uCount, sizeof(ITEM), FnCompare);
}
*/

/*
//################################################################################################
//################################################################################################
//###
//###  CLASS:   CArrayPtrEx
//###  PURPOSE: Extended smart pointer that also holds the count of objects held.
//###
//################################################################################################
//################################################################################################

// ***********************************************************************************************
// CLASS:   CArrayPtrEx
// PURPOSE: A vector class for arrays of objects or primitive ITEMs.
//
template<class ITEM>
class CArrayPtrEx
{
private:    // Private data.
   ITEM *m_pArray;
   UINT  m_uCount;

private:    // Prevent copy constructors and operator=().
   CArrayPtrEx(const CArrayPtrEx &);
   const CArrayPtrEx &operator=(const CArrayPtrEx &);

public:     // Public member functions.

   // Constructors & destructor. See notes below.
   CArrayPtrEx();
   CArrayPtrEx(UINT uCount);
   ~CArrayPtrEx();
   
   // Allocation and destruction of memory pointed to by this object.
   BOOL Alloc(UINT uCount);
   BOOL Realloc(UINT uNewCount, BOOL bZero=FALSE);
   BOOL Clone(const ITEM *pItem, UINT uCount);
   void Free();

   // Accessor functions to get at the wrapped array.
   operator ITEM *() const;
   ITEM *Get() const;

   // Destroys the held array and replaces it with another.
   void Set(ITEM *, UINT uCount);

   // Non-destructive release of the held pointer.
   ITEM *Release();

   // Returns number of items held.
   UINT GetCount() const;

   // Zero the held array.
   void Zero();

   // Sorts the data held in the pointer according to a user-supplied callback.
   void Sort(int (__cdecl *FnCompare )(const void *elem1, const void *elem2 ));
};

//================================================================================================
// FUNCTION: Constructor
// PURPOSE:  Create a new object that wraps a passed pointer.
// NOTES:    If the passed pointer is non-NULL, it *MUST* have been created on the 
//           heap using the array allocater new[]. 
//           Unfortunately there is no reliable way to ASSERT this.
//
//           The definition of this constructor contains a default parameter of NULL 
//           so that CArrayPtrEx objects can be created that do not hold anything.
//
template <class ITEM>
inline CArrayPtrEx<ITEM>::CArrayPtrEx<ITEM>()
{
   MEMBERASSERT();
   m_pArray = NULL;
   m_uCount = 0;
}

//================================================================================================
// FUNCTION: Constructor
// PURPOSE:  Create a new object and allocate a buffer of the passed size.
//
template <class ITEM>
inline CArrayPtrEx<ITEM>::CArrayPtrEx<ITEM>(UINT uCount)
{
   MEMBERASSERT();
   m_pArray = NULL;
   m_uCount = 0;
   Alloc(uCount);
}

//================================================================================================
// FUNCTION: Destructor
// PURPOSE:  Frees a held pointer if non-NULL.
//
template <class ITEM>
inline CArrayPtrEx<ITEM>::~CArrayPtrEx<ITEM>()
{
   MEMBERASSERT();
   Free();
}

//================================================================================================
// FUNCTION: Alloc
// PURPOSE:  Frees any held pointer and allocates a new array of the wrapped ITEM.
//
template <class ITEM>
inline BOOL CArrayPtrEx<ITEM>::Alloc(UINT uCount)
{
   MEMBERASSERT();

   // Check if we already have the right size.
   if (m_uCount == uCount)
      return TRUE;

   // Free any existing array.
   Free();

   // Return now if nothing is to be allocated.
   if (uCount == 0)
      return TRUE;

   // Allocate the new array.
   m_pArray = new ITEM[uCount];
   if (m_pArray)
      m_uCount = uCount;
   return (m_pArray!=NULL);
}

//================================================================================================
// FUNCTION: Realloc
// PURPOSE:  Reallocates the held pointer keeping the held data.
//
template <class ITEM>
inline BOOL CArrayPtrEx<ITEM>::Realloc(UINT uNewCount, BOOL bZero)
{
   MEMBERASSERT();
   ARRAYASSERT(m_pArray, m_uCount);

   if (m_uCount == uNewCount)
      return TRUE;

   UINT uOldCount = m_uCount;

   CArrayPtrEx<ITEM> pNewArray;
   if (uNewCount && !pNewArray.Alloc(uNewCount))
      return FALSE;

   for (UINT i=0; i<min(m_uCount, uNewCount); i++)
      pNewArray[i] = m_pArray[i];

   // Free any existing array.
   Free();

   // Return now if nothing is to be allocated.
   if (uNewCount == 0)
      return TRUE;

   // Keep the new array.
   m_pArray = pNewArray.Release();
   m_uCount = uNewCount;

   if (bZero && (uOldCount < m_uCount))
      memset(m_pArray+uOldCount, 0, (m_uCount-uOldCount)*sizeof(ITEM));

   return TRUE;
}

//================================================================================================
// FUNCTION: Clone
// PURPOSE:  Frees any held pointer and allocates a new array of the wrapped ITEM.
//           The passed array is then copied into the new buffer.
//
template <class ITEM>
inline BOOL CArrayPtrEx<ITEM>::Clone(const ITEM *pItem, UINT uCount)
{
   MEMBERASSERT();

   // Reallocate the held pointer.
   if (!Alloc(uCount))
      return FALSE;

   // Check that we have valid parameters.
   ARRAYASSERT(pItem, uCount);

   // If this object was only for wrapping primitive types, a memcpy call
   // would be most efficient for cloning, but this is inappropriate for
   // arrays of objects.
   // memcpy(m_pArray, pItem, uCount*sizeof(ITEM));

   // Use a for loop to copy the array into the new buffer.
   for (UINT i=0; i<uCount; i++)
      m_pArray[i] = pItem[i];

   m_uCount = uCount;
   return TRUE;
}

//================================================================================================
// FUNCTION: Free
// PURPOSE:  De-allocates the held pointer. Benign NOP if no pointer held.
//
template <class ITEM>
inline void CArrayPtrEx<ITEM>::Free()
{
   MEMBERASSERT();
   delete[] m_pArray;
   m_pArray = NULL;
   m_uCount = 0;
}

//================================================================================================
// FUNCTION: Get
// PURPOSE:  Returns the held pointer without giving up ownership of it.
//
template <class ITEM>
inline ITEM *CArrayPtrEx<ITEM>::Get() const
{
   MEMBERASSERT();
   return m_pArray; 
}

//================================================================================================
// FUNCTION: GetCount
// PURPOSE:  Returns the count of items held.
//
template <class ITEM>
inline UINT CArrayPtrEx<ITEM>::GetCount() const
{
   MEMBERASSERT();
   return m_uCount; 
}

//================================================================================================
// FUNCTION: Set
// PURPOSE:  Destroys the held array and replaces it with another.
//
template <class ITEM>
inline void CArrayPtrEx<ITEM>::Set(ITEM *pItem, UINT uCount)
{
   ASSERT_NOTONSTACK(pItem);
   Free();
   m_pArray = pItem;
   m_uCount = uCount;
}

//================================================================================================
// FUNCTION: Overloaded cast operator
// PURPOSE:  Returns the held pointer without giving up ownership of it.
//
template <class ITEM>
inline CArrayPtrEx<ITEM>::operator ITEM *() const
{
   MEMBERASSERT();
   return Get();
}

//================================================================================================
// FUNCTION: Release
// PURPOSE:  Returns the held pointer, giving up ownership of it.
//
template <class ITEM>
inline ITEM *CArrayPtrEx<ITEM>::Release()
{
   MEMBERASSERT();
   ITEM *rval = m_pArray;
   m_pArray   = NULL;
   m_uCount   = 0;
   return rval; 
}

//================================================================================================
// FUNCTION: Zero
// PURPOSE:  Zero's out the held pointer from item 0 for uCount items.
//
template <class ITEM>
inline void CArrayPtrEx<ITEM>::Zero()
{
   MEMBERASSERT();
   ARRAYASSERT(m_pArray, m_uCount);
   memset(m_pArray, 0, m_uCount*sizeof(ITEM));
}

//================================================================================================
// FUNCTION: Sort
// PURPOSE:  Sorts the data held in the pointer according to a user-supplied callback.
//
template <class ITEM>
inline void CArrayPtrEx<ITEM>::Sort(int (__cdecl *FnCompare )(const void *elem1, const void *elem2 ))
{
   MEMBERASSERT();
   ARRAYASSERT(m_pArray, m_uCount);
   qsort(m_pArray, m_uCount, sizeof(ITEM), FnCompare);
}
*/
#endif          // INC_ARRAYPTR_HPP
