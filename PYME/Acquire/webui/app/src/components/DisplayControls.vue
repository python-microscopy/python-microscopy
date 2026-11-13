<template>
  <div class="display-controls section">
    <h6 class="section-heading">Display</h6>
    
    <div class="control-group inline-control">
      <label>Range:</label>
      <div class="range-inputs">
        <input
          type="number"
          v-model.number="min"
          class="input-sm"
        />
        <input
          type="number"
          v-model.number="max"
          class="input-sm"
        />
      </div>
      <label class="checkbox-label">
        <input type="checkbox" v-model="autoscale" />
        <span>auto</span>
      </label>
    </div>

    <div class="control-group inline-control">
      <label>Zoom:</label>
      <select v-model.number="zoom" class="select-sm">
        <option :value="25">25%</option>
        <option :value="50">50%</option>
        <option :value="100">100%</option>
        <option :value="200">200%</option>
        <option :value="400">400%</option>
      </select>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const min = ref(950)
const max = ref(1050)
const autoscale = ref(false)
const zoom = ref(100)

defineExpose({
  min,
  max,
  autoscale,
  zoom,
})
</script>

<style scoped lang="scss">
.inline-control {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  > label:first-child {
    min-width: 60px;
    margin: 0;
  }
}

.range-inputs {
  display: flex;
  gap: 0.5rem;
  flex: 1;
}

.input-sm {
  width: 80px;
}
</style>

