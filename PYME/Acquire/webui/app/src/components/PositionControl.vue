<template>
  <div class="position-control">
    <label>{{ axis }} [µm]:</label>
    <div class="position-inputs">
      <button @click="decrement" class="btn-sm">&lt;</button>
      <input
        type="number"
        :value="modelValue"
        @change="updateValue"
        class="input-sm"
      />
      <button @click="increment" class="btn-sm">&gt;</button>
    </div>
  </div>
</template>

<script setup>
import { useScopeStore } from '@/stores/scope'

const props = defineProps({
  modelValue: Number,
  axis: String,
  delta: {
    type: Number,
    default: 1.0
  }
})

const emit = defineEmits(['update:modelValue'])

const scopeStore = useScopeStore()

function updateValue(event) {
  const value = parseFloat(event.target.value)
  setPosition(value)
}

function increment() {
  setPosition(props.modelValue + props.delta)
}

function decrement() {
  setPosition(props.modelValue - props.delta)
}

function setPosition(value) {
  emit('update:modelValue', value)
}
</script>

<style scoped lang="scss">
.position-control {
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.position-control label {
  font-size: 0.875rem;
  color: #dee2e6;
  margin: 0;
  min-width: 60px;
}

.position-inputs {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  flex: 1;
}

.input-sm {
  width: 100px;
  text-align: center;
}
</style>
