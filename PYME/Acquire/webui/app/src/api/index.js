// Helper function to get cookie value
export function getCookie(name) {
  const value = `; ${document.cookie}`
  const parts = value.split(`; ${name}=`)
  if (parts.length === 2) return parts.pop().split(';').shift()
  return null
}

// Helper function to set cookie
export function setCookie(name, value, days = 7) {
  const expires = new Date()
  expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000)
  document.cookie = `${name}=${value}; path=/; expires=${expires.toUTCString()}; SameSite=Strict`
}

// Helper function to delete cookie
export function deleteCookie(name) {
  document.cookie = `${name}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT`
}

// Authentication API
export const authAPI = {
  async login(email, password) {
    const response = await fetch('/do_login?' + new URLSearchParams({
      email,
      password
    }))
    return response.json()
  },

  async logout() {
    const response = await fetch('/logout')
    return response.json()
  },

  async getUserInfo() {
    const response = await fetch('/api/user', {
      credentials: 'same-origin'
    })
    return response.json()
  }
}

// Scope state API
export const scopeAPI = {
  async getState() {
    const response = await fetch('/get_scope_state')
    return response.json()
  },

  async updateState(state) {
    const response = await fetch('/update_scope_state', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(state),
    })
    return response.json()
  },

  // Long polling for state updates
  async pollState() {
    const response = await fetch('/scope_state_longpoll')
    return response.json()
  }
}

// Camera API
export const cameraAPI = {
  async getFramePZF() {
    const response = await fetch('/get_frame_pzf', {
      credentials: 'same-origin'
    })
    return response.arrayBuffer()
  },

  async getFramePNG(min, max) {
    const params = new URLSearchParams()
    if (min !== null) params.append('min', min)
    if (max !== null) params.append('max', max)
    
    const response = await fetch(`/get_frame_png?${params}`)
    return response.blob()
  }
}

// Spooler API
export const spoolerAPI = {
  async getInfo() {
    const response = await fetch('/spool_controller/info')
    return response.json()
  },

  async getSettings() {
    const response = await fetch('/spool_controller/settings')
    return response.json()
  },

  async updateSettings(settings) {
    const response = await fetch('/spool_controller/settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    })
    return response.json()
  },

  async startSpooling() {
    const response = await fetch('/spool_controller/start_spooling')
    return response.json()
  },

  async stopSpooling() {
    const response = await fetch('/spool_controller/stop_spooling')
    return response.json()
  },

  // Long polling for spooler updates
  async pollInfo() {
    const response = await fetch('/spool_controller/info_longpoll')
    return response.json()
  }
}

// Stack settings API
export const stackAPI = {
  async getSettings() {
    const response = await fetch('/stack_settings/settings')
    return response.json()
  },

  async updateSettings(settings) {
    const response = await fetch('/stack_settings/update', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    })
    return response.json()
  },

  // Long polling for stack settings updates
  async pollSettings() {
    const response = await fetch('/stack_settings/settings_longpoll')
    return response.json()
  }
}
