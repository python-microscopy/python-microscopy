import { defineStore } from 'pinia'
import { ref } from 'vue'
import { stackAPI } from '@/api'

export const useStackStore = defineStore('stack', () => {
  const settings = ref({
    ScanPiezo: 'z',
    ScanMode: 'Middle and Number',
    StartPos: 0,
    EndPos: 10,
    StepSize: 0.2,
    NumSlices: 50,
    DwellFrames: 1
  })
  const isPolling = ref(false)

  async function loadSettings() {
    try {
      const data = await stackAPI.getSettings()
      settings.value = data
    } catch (error) {
      console.error('Error loading stack settings:', error)
    }
  }

  async function updateSettings(updates) {
    try {
      await stackAPI.updateSettings(updates)
      // Update local state optimistically
      Object.assign(settings.value, updates)
    } catch (error) {
      console.error('Error updating stack settings:', error)
    }
  }

  async function startPolling() {
    if (isPolling.value) return
    
    isPolling.value = true
    
    const poll = async () => {
      if (!isPolling.value) return
      
      try {
        const data = await stackAPI.pollSettings()
        settings.value = data
        poll() // Continue polling
      } catch (error) {
        console.error('Error polling stack settings:', error)
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
    settings,
    loadSettings,
    updateSettings,
    startPolling,
    stopPolling,
  }
})
