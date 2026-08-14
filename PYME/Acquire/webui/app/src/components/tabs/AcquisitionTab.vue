<template>
  <div class="acquisition-tab">
    <h3>PALM/STORM/PAINT Settings</h3>
    
    <div class="content-grid">
      <div class="settings-section">
        <h6>Protocol</h6>
        <div class="form-group">
          <label>Protocol File:</label>
          <select class="form-control">
            <option>None chosen...</option>
          </select>
        </div>

        <div class="radio-group">
          <label class="radio-label">
            <input
              type="radio"
              :value="false"
              v-model="spoolerStore.info.settings.z_stepped"
              @change="updateZStepped"
            />
            <span>Standard</span>
          </label>
          <label class="radio-label">
            <input
              type="radio"
              :value="true"
              v-model="spoolerStore.info.settings.z_stepped"
              @change="updateZStepped"
            />
            <span>Z stepped</span>
          </label>
        </div>

        <StackSettings
          v-if="spoolerStore.info.settings.z_stepped"
          :show-dwell-time="true"
        />
      </div>

      <div class="settings-section">
        <h6>Spooling</h6>
        <div class="form-group">
          <label>Spool to:</label>
          <div class="spool-methods">
            <label
              v-for="method in spoolerStore.info.available_spool_methods"
              :key="method"
              class="radio-label"
            >
              <input
                type="radio"
                :value="method"
                v-model="spoolerStore.info.settings.method"
                @change="updateSpoolMethod"
              />
              <span>{{ method }}</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useSpoolerStore } from '@/stores/spooler'
import StackSettings from '@/components/StackSettings.vue'

const spoolerStore = useSpoolerStore()

function updateZStepped(event) {
  spoolerStore.updateSettings({ z_stepped: event.target.value === 'true' })
}

function updateSpoolMethod(event) {
  spoolerStore.updateSettings({ method: event.target.value })
}
</script>

<style scoped lang="scss">
.acquisition-tab {
  padding: 2rem;
}

.acquisition-tab h3 {
  margin-bottom: 1.5rem;
  color: #fff;
}

.content-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
}

.settings-section h6 {
  margin-bottom: 1rem;
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
  color: #adb5bd;
}

.spool-methods {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}
</style>
