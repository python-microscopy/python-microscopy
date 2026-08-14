<template>
  <div class="main-view">
    <nav class="navbar">
      <div class="navbar-brand">PYME Acquire - now with more web</div>
      <div class="navbar-user">
        <span v-if="authStore.user" class="user-name">
          Signed in as {{ authStore.user }}
        </span>
        <button @click="handleLogout" class="btn-logout">Sign out</button>
      </div>
    </nav>

    <div class="container">
      <div class="sidebar">
        
        <DisplayControls ref="displayControlsRef" />
        <HardwareControls />
        <AcquisitionControls />
      </div>

      <main class="main-content">
        <CameraDisplay ref="cameraDisplayRef" />
        <div class="tabs">
          <button
            v-for="tab in tabs"
            :key="tab.id"
            :class="['tab', { active: activeTab === tab.id }]"
            @click="activeTab = tab.id"
          >
            {{ tab.label }}
          </button>
        </div>

        <div class="tab-content">
          <component :is="activeTabComponent" />
        </div>
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useScopeStore } from '@/stores/scope'
import { useSpoolerStore } from '@/stores/spooler'
import { useStackStore } from '@/stores/stack'

import CameraDisplay from '@/components/CameraDisplay.vue'
import DisplayControls from '@/components/DisplayControls.vue'
import HardwareControls from '@/components/HardwareControls.vue'
import AcquisitionControls from '@/components/AcquisitionControls.vue'
import AcquisitionTab from '@/components/tabs/AcquisitionTab.vue'
import StateTab from '@/components/tabs/StateTab.vue'
import ConsoleTab from '@/components/tabs/ConsoleTab.vue'

const router = useRouter()
const authStore = useAuthStore()
const scopeStore = useScopeStore()
const spoolerStore = useSpoolerStore()
const stackStore = useStackStore()

const cameraDisplayRef = ref(null)
const displayControlsRef = ref(null)
const activeTab = ref('acquisition')

const tabs = [
  { id: 'acquisition', label: 'Acquisition' },
  { id: 'state', label: 'State' },
  { id: 'console', label: 'Console' },
]

const activeTabComponent = computed(() => {
  switch (activeTab.value) {
    case 'acquisition':
      return AcquisitionTab
    case 'state':
      return StateTab
    case 'console':
      return ConsoleTab
    default:
      return AcquisitionTab
  }
})

async function handleLogout() {
  await authStore.logout()
  router.push('/login')
}

onMounted(async () => {
  // Load initial data
  await Promise.all([
    scopeStore.loadState(),
    spoolerStore.loadInfo(),
    stackStore.loadSettings(),
  ])
  
  // Start long polling
  scopeStore.startPolling()
  spoolerStore.startPolling()
  stackStore.startPolling()
  
  // Sync display controls with camera display
  if (cameraDisplayRef.value && displayControlsRef.value) {
    // Watch display control changes and update camera
    watch(() => displayControlsRef.value.min, (newVal) => {
      if (cameraDisplayRef.value) {
        cameraDisplayRef.value.displayMin = newVal
      }
    })
    
    watch(() => displayControlsRef.value.max, (newVal) => {
      if (cameraDisplayRef.value) {
        cameraDisplayRef.value.displayMax = newVal
      }
    })
    
    watch(() => displayControlsRef.value.autoscale, (newVal) => {
      if (cameraDisplayRef.value) {
        cameraDisplayRef.value.autoScale = newVal
      }
    })
    
    watch(() => displayControlsRef.value.zoom, (newVal) => {
      if (cameraDisplayRef.value) {
        cameraDisplayRef.value.zoom = newVal
      }
    })
  }
})

onUnmounted(() => {
  // Stop long polling when component unmounts
  scopeStore.stopPolling()
  spoolerStore.stopPolling()
  stackStore.stopPolling()
})
</script>

<style scoped lang="scss">
.main-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #1a1a1a;
  color: #fff;
}

.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1.5rem;
  background-color: #343a40;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.navbar-brand {
  font-size: 1.25rem;
  font-weight: 500;
}

.navbar-user {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.user-name {
  color: #adb5bd;
}

.btn-logout {
  padding: 0.5rem 1rem;
  background-color: #6c757d;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.btn-logout:hover {
  background-color: #5a6268;
}

.container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 320px;
  padding: 1rem;
  background-color: #212529;
  overflow-y: auto;
  border-right: 1px solid #343a40;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.tabs {
  display: flex;
  background-color: #212529;
  border-bottom: 1px solid #343a40;
}

.tab {
  padding: 0.75rem 1.5rem;
  background-color: transparent;
  color: #adb5bd;
  border: none;
  border-bottom: 3px solid transparent;
  cursor: pointer;
  transition: all 0.2s;
}

.tab:hover {
  background-color: #2c3136;
  color: #fff;
}

.tab.active {
  color: #fff;
  border-bottom-color: #007bff;
  background-color: #2c3136;
}

.tab-content {
  flex: 1;
  overflow: auto;
  background-color: #1a1a1a;
}
</style>
