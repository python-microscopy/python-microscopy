<template>
  <div class="console-tab">
    <iframe
      src="http://localhost:8765/"
      scrolling="no"
      class="console-frame"
    ></iframe>
  </div>
</template>

<script setup>
</script>

<style scoped lang="scss">
.console-tab {
  height: 100%;
}

.console-frame {
  width: 100%;
  height: 100%;
  border: none;
  background-color: #000;
}
</style>
