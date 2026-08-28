<template>
  <div class="acquisition-controls section">
    <h4 class="section-heading">Acquisition</h4>

    <h6 class="subsection-heading">Type</h6>
    <div class="radio-group">
      <label class="radio-label">
        <input type="radio" value="stack" v-model="acquisitionType" disabled />
        <span>Z Stack</span>
      </label>
      <label class="radio-label">
        <input type="radio" value="series" v-model="acquisitionType" checked />
        <span>PALM/STORM/PAINT</span>
      </label>
      <label class="radio-label">
        <input type="radio" value="tiles" v-model="acquisitionType" disabled />
        <span>Tiled</span>
      </label>
    </div>

    <div v-if="spoolerStore.info.settings" class="spool-info">
      <p class="spool-target">
        Spool to: {{ spoolerStore.info.settings.series_name }}
      </p>

      <div v-if="spoolerStore.info.status.spooling" class="spool-status">
        <p>Currently spooling, {{ spoolerStore.info.status.frames_spooled }} frames spooled</p>
        <button @click="spoolerStore.stopSpooling()" class="btn btn-danger">
          Stop
        </button>
      </div>
      <div v-else class="spool-status">
        <button @click="spoolerStore.startSpooling()" class="btn btn-primary">
          Start
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useSpoolerStore } from '@/stores/spooler'

const spoolerStore = useSpoolerStore()
const acquisitionType = ref('series')
</script>

<style scoped lang="scss">
.section-heading {
  font-size: 1rem;
  color: #fff;
}

.spool-info {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #343a40;
}

.spool-target {
  margin-bottom: 0.75rem;
  text-align: center;
  font-size: 0.875rem;
  color: #adb5bd;
}

.spool-status {
  text-align: center;
}

.spool-status p {
  margin-bottom: 0.75rem;
  font-size: 0.875rem;
  color: #dee2e6;
}
</style>
