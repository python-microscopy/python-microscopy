<template>
  <div class="hardware-controls">
    <div class="section">
      <h6 class="section-heading">Camera</h6>
      <div class="control-group">
        <label>
          Integration time [ms]:
          <input
            type="number"
            :value="integrationTimeMs"
            @change="updateIntegrationTime"
            class="input-sm"
          />
        </label>
      </div>
    </div>

    <div class="section">
      <h6 class="section-heading">Lasers</h6>
      <LaserControl
        v-for="lname in scopeStore.laserNames"
        :key="lname"
        :name="lname"
        :power="scopeStore.state['Lasers.' + lname + '.Power']"
        :on="scopeStore.state['Lasers.' + lname + '.On']"
        :max-power="scopeStore.state['Lasers.' + lname + '.MaxPower']"
      />
    </div>

    <div class="section">
      <h6 class="section-heading">Positioning</h6>
      <PositionControl
        v-model="xPosition"
        axis="x"
        :delta="1"
      />
      <PositionControl
        v-model="yPosition"
        axis="y"
        :delta="1"
      />
      <PositionControl
        v-model="zPosition"
        axis="z"
        :delta="0.2"
      />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useScopeStore } from '@/stores/scope'
import LaserControl from './LaserControl.vue'
import PositionControl from './PositionControl.vue'

const scopeStore = useScopeStore()

const integrationTimeMs = computed(() => scopeStore.integrationTimeMs)

const xPosition = computed({
  get: () => scopeStore.state['Positioning.x'] || 0,
  set: (value) => scopeStore.updateState({ 'Positioning.x': value })
})

const yPosition = computed({
  get: () => scopeStore.state['Positioning.y'] || 0,
  set: (value) => scopeStore.updateState({ 'Positioning.y': value })
})

const zPosition = computed({
  get: () => scopeStore.state['Positioning.z'] || 0,
  set: (value) => scopeStore.updateState({ 'Positioning.z': value })
})

function updateIntegrationTime(event) {
  const value = parseFloat(event.target.value) / 1000.0
  scopeStore.updateState({ 'Camera.IntegrationTime': value })
}
</script>

<style scoped lang="scss">
.input-sm {
  width: 100px;
  margin-left: 0.5rem;
}
</style>
