<template>
  <div id="app">
    <router-view />
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()

onMounted(() => {
  // Check authentication status on app load
  authStore.checkAuth()
})
</script>

<style>
#app {
  width: 100%;
  height: 100vh;
}
</style>
