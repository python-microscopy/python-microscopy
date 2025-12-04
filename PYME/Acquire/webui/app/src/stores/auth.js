import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { authAPI, getCookie, setCookie, deleteCookie } from '@/api'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const isAuthenticated = computed(() => user.value !== null)

  async function login(email, password) {
    try {
      const data = await authAPI.login(email, password)
      
      if (data.success) {
        // Set the auth cookie
        setCookie('auth', data.token)
        user.value = data.user
        return { success: true }
      } else {
        return { success: false, error: data.error }
      }
    } catch (error) {
      console.error('Login error:', error)
      return { success: false, error: 'An error occurred during login' }
    }
  }

  async function logout() {
    try {
      await authAPI.logout()
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      // Clear cookie and user state regardless of API result
      deleteCookie('auth')
      user.value = null
    }
  }

  async function checkAuth() {
    try {
      const data = await authAPI.getUserInfo()
      
      if (data.authenticated) {
        user.value = data.user
      } else {
        user.value = null
      }
    } catch (error) {
      console.error('Auth check error:', error)
      user.value = null
    }
  }

  return {
    user,
    isAuthenticated,
    login,
    logout,
    checkAuth,
  }
})
