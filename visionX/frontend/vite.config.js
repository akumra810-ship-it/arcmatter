import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload-video': 'http://localhost:8000',
      '/video-job': 'http://localhost:8000',
      '/videos': 'http://localhost:8000',
      '/interpolate': 'http://localhost:8000'
    }
  }
})