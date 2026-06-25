<template>
  <div class="laser-control">
    <label class="laser-label">{{ name }}:</label>
    <div class="laser-inputs">
      <input
        type="range"
        :value="power"
        :max="maxPower"
        @input="updatePower"
        class="range-slider"
      />
      <input
        type="number"
        :value="power"
        @change="updatePower"
        class="input-sm"
      />
      <label class="checkbox-label">
        <input
          type="checkbox"
          :checked="on"
          @change="updateOn"
        />
        <span>On</span>
      </label>
    </div>
  </div>
</template>

<script setup>
import { useScopeStore } from '@/stores/scope'

const props = defineProps({
  name: String,
  power: Number,
  on: Boolean,
  maxPower: Number,
})

const scopeStore = useScopeStore()

function updatePower(event) {
  const value = parseFloat(event.target.value)
  scopeStore.updateState({
    [`Lasers.${props.name}.Power`]: value
  })
}

function updateOn(event) {
  scopeStore.updateState({
    [`Lasers.${props.name}.On`]: event.target.checked
  })
}
</script>

<style scoped lang="scss">
.laser-control {
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.laser-label {
  font-size: 0.875rem;
  color: #dee2e6;
  margin: 0;
  min-width: 50px;
}

.laser-inputs {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
}

.range-slider {
  flex: 1;
  height: 6px;
  background: #495057;
  outline: none;
  border-radius: 3px;
}

.range-slider::-webkit-slider-thumb {
  width: 16px;
  height: 16px;
  background: #007bff;
  cursor: pointer;
  border-radius: 50%;
}

.range-slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  background: #007bff;
  cursor: pointer;
  border-radius: 50%;
}

.input-sm {
  width: 60px;
}

.checkbox-label {
  white-space: nowrap;
}

.checkbox-label span {
  gap: 0.25rem;
}
</style>
