import { defineStore } from 'pinia'
import { ref } from 'vue'
import { spoolerAPI } from '@/api'

export const useSpoolerStore = defineStore('spooler', () => {
  const info = ref({
    status: { spooling: false, frames_spooled: 0 },
    settings: { z_stepped: false, method: 'Queue', series_name: '' },
    available_spool_methods: []
  })
  const isPolling = ref(false)

  async function loadInfo() {
    try {
      const data = await spoolerAPI.getInfo()
      info.value = data
    } catch (error) {
      console.error('Error loading spooler info:', error)
    }
  }

  async function updateSettings(settings) {
    try {
      await spoolerAPI.updateSettings(settings)
      // Update local state optimistically
      Object.assign(info.value.settings, settings)
    } catch (error) {
      console.error('Error updating spooler settings:', error)
    }
  }

  async function startSpooling() {
    try {
      await spoolerAPI.startSpooling()
      await loadInfo() // Reload to get updated status
    } catch (error) {
      console.error('Error starting spooling:', error)
    }
  }

  async function stopSpooling() {
    try {
      await spoolerAPI.stopSpooling()
      await loadInfo() // Reload to get updated status
    } catch (error) {
      console.error('Error stopping spooling:', error)
    }
  }

  async function startPolling() {
    if (isPolling.value) return
    
    isPolling.value = true
    
    const poll = async () => {
      if (!isPolling.value) return
      
      try {
        const data = await spoolerAPI.pollInfo()
        info.value = data
        poll() // Continue polling
      } catch (error) {
        console.error('Error polling spooler info:', error)
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
    info,
    loadInfo,
    updateSettings,
    startSpooling,
    stopSpooling,
    startPolling,
    stopPolling,
  }
})
