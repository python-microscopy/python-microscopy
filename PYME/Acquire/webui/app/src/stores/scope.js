import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { scopeAPI } from '@/api'

export const useScopeStore = defineStore('scope', () => {
  const state = ref({})
  const isPolling = ref(false)

  const integrationTimeMs = computed(() => {
    return (state.value['Camera.IntegrationTime'] || 0) * 1000
  })

  const laserNames = computed(() => {
    const laserKeys = Object.keys(state.value).filter(key => 
      key.startsWith('Lasers') && key.endsWith('On')
    )
    return laserKeys.map(key => key.split('.')[1])
  })

  async function loadState() {
    try {
      const data = await scopeAPI.getState()
      state.value = data
    } catch (error) {
      console.error('Error loading scope state:', error)
    }
  }

  async function updateState(updates) {
    try {
      await scopeAPI.updateState(updates)
      // Update local state optimistically
      Object.assign(state.value, updates)
    } catch (error) {
      console.error('Error updating scope state:', error)
    }
  }

  async function startPolling() {
    if (isPolling.value) return
    
    isPolling.value = true
    
    const poll = async () => {
      if (!isPolling.value) return
      
      try {
        const data = await scopeAPI.pollState()
        state.value = data
        poll() // Continue polling
      } catch (error) {
        console.error('Error polling scope state:', error)
        // Retry after delay
        setTimeout(poll, 5000)
      }
    }
    
    poll()
  }

  function stopPolling() {
    isPolling.value = false
  }

  return {
    state,
    integrationTimeMs,
    laserNames,
    loadState,
    updateState,
    startPolling,
    stopPolling,
  }
})
