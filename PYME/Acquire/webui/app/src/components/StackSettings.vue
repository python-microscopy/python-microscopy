<template>
  <div class="stack-settings">
    <div class="form-group">
      <label>Axis:</label>
      <select
        :value="stackStore.settings.ScanPiezo"
        @change="updateSetting('ScanPiezo', $event.target.value)"
        class="form-control"
      >
        <option>z</option>
        <option>...</option>
      </select>
    </div>

    <div class="form-group">
      <label>Mode:</label>
      <select
        :value="stackStore.settings.ScanMode"
        @change="updateSetting('ScanMode', $event.target.value)"
        class="form-control"
      >
        <option>Middle and Number</option>
        <option>Start and End</option>
      </select>
    </div>

    <div class="form-row">
      <div class="form-col">
        <label>Start:</label>
        <div class="input-with-button">
          <input
            type="number"
            step="0.01"
            :value="stackStore.settings.StartPos"
            @change="updateSetting('StartPos', parseFloat($event.target.value))"
            class="form-control"
          />
          <button class="btn-secondary">Set</button>
        </div>
      </div>
      <div class="form-col">
        <label>End:</label>
        <div class="input-with-button">
          <input
            type="number"
            step="0.01"
            :value="stackStore.settings.EndPos"
            @change="updateSetting('EndPos', parseFloat($event.target.value))"
            class="form-control"
          />
          <button class="btn-secondary">Set</button>
        </div>
      </div>
    </div>

    <div class="form-row">
      <div class="form-col">
        <label>Step size [µm]:</label>
        <input
          type="number"
          :value="stackStore.settings.StepSize"
          @change="updateSetting('StepSize', parseFloat($event.target.value))"
          class="form-control"
        />
      </div>
      <div class="form-col">
        <label>Num slices:</label>
        <input
          type="number"
          :value="stackStore.settings.NumSlices"
          @change="updateSetting('NumSlices', parseInt($event.target.value))"
          class="form-control"
        />
      </div>
      <div v-if="showDwellTime" class="form-col">
        <label>Dwell time:</label>
        <input
          type="number"
          step="0.01"
          :value="stackStore.settings.DwellFrames"
          @change="updateSetting('DwellFrames', parseFloat($event.target.value))"
          class="form-control"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { useStackStore } from '@/stores/stack'

const props = defineProps({
  showDwellTime: {
    type: Boolean,
    default: false
  }
})

const stackStore = useStackStore()

function updateSetting(propName, value) {
  stackStore.updateSettings({ [propName]: value })
}
</script>

<style scoped lang="scss">
.stack-settings {
  padding: 1rem;
  background-color: #2c3136;
  border-radius: 4px;
}

.input-with-button {
  display: flex;
  gap: 0.25rem;
}

.input-with-button .form-control {
  flex: 1;
}
</style>
