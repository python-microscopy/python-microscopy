import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      // Proxy API requests to the Python server (running on port 8999)
      '/api': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
      '/get_frame_pzf': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
      '/get_scope_state': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
      '/update_scope_state': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
      '/spool_controller': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
      '/stack_settings': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
      '/scope_state_longpoll': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
      '/do_login': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
      '/logout': {
        target: 'http://localhost:8999',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
})
