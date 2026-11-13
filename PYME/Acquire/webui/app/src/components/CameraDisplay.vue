<template>
  <div class="camera-display">
    <canvas ref="canvas" id="cam_canvas"></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { cameraAPI } from '@/api'
import { decodePZF, mapArrayToRGBA } from '@/utils/pzf'

const canvas = ref(null)
const isPolling = ref(false)
const displayMin = ref(950)
const displayMax = ref(1050)
const autoScale = ref(false)
const zoom = ref(100)

let _actualMin = 0
let _actualMax = 1e9

async function pollFrame() {
  if (!isPolling.value) return
  
  try {
    const arrayBuffer = await cameraAPI.getFramePZF()
    
    // Decode PZF data
    const decoded = decodePZF(arrayBuffer)
    
    // Map to RGBA with intensity scaling
    const mapped = mapArrayToRGBA(
      decoded.data, 
      displayMin.value, 
      displayMax.value
    )
    
    // Update actual min/max for autoscaling
    _actualMin = mapped.min
    _actualMax = mapped.max
    
    if (autoScale.value) {
      displayMin.value = _actualMin
      displayMax.value = _actualMax
    }
    
    // Create ImageData and render to canvas
    const imageData = new ImageData(mapped.imageData, decoded.width, decoded.height)
    
    if (canvas.value) {
      const zoomScale = zoom.value / 100
      const canvasWidth = decoded.width * zoomScale
      const canvasHeight = decoded.height * zoomScale
      
      canvas.value.width = canvasWidth
      canvas.value.height = canvasHeight
      
      const ctx = canvas.value.getContext('2d')
      
      // Use createImageBitmap for better performance with resizing
      createImageBitmap(imageData, {
        resizeWidth: canvasWidth,
        resizeHeight: canvasHeight,
        resizeQuality: 'pixelated'
      }).then(bitmap => {
        ctx.drawImage(bitmap, 0, 0)
      })
    }
    
    // Continue polling
    setTimeout(pollFrame, 0)
  } catch (error) {
    console.error('Error polling camera frame:', error)
    setTimeout(pollFrame, 5000)
  }
}

onMounted(() => {
  isPolling.value = true
  pollFrame()
})

onUnmounted(() => {
  isPolling.value = false
})

defineExpose({
  displayMin,
  displayMax,
  autoScale,
  zoom,
})
</script>

<style scoped lang="scss">
.camera-display {
  margin-bottom: 1rem;
  background-color: #000;
  border-radius: 4px;
  overflow: hidden;
}

#cam_canvas {
  width: 100%;
  display: block;
  image-rendering: pixelated;
}
</style>
