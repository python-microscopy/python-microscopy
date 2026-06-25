<template>
  <div class="state-tab">
    <table class="state-table">
      <thead>
        <tr>
          <th>Property</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(value, key) in scopeStore.state" :key="key">
          <td>{{ key }}</td>
          <td>{{ value }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { useScopeStore } from '@/stores/scope'

const scopeStore = useScopeStore()
</script>

<style scoped lang="scss">
.state-tab {
  padding: 2rem;
}

.state-table {
  width: 100%;
  border-collapse: collapse;
  background-color: #212529;
  border-radius: 4px;
  overflow: hidden;
}

.state-table th {
  padding: 0.75rem 1rem;
  background-color: #343a40;
  color: #adb5bd;
  text-align: left;
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
}

.state-table td {
  padding: 0.75rem 1rem;
  border-top: 1px solid #343a40;
  color: #dee2e6;
  font-size: 0.875rem;
}

.state-table tbody tr:hover {
  background-color: #2c3136;
}
</style>
