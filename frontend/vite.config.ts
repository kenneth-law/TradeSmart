import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api':                        { target: 'http://localhost:5001', changeOrigin: true },
      '/analysis_progress':          { target: 'http://localhost:5001', changeOrigin: true },
      '/integrated_progress_stream': { target: 'http://localhost:5001', changeOrigin: true },
      '/backtest_progress_stream':   { target: 'http://localhost:5001', changeOrigin: true },
      '/healthcheck':                { target: 'http://localhost:5001', changeOrigin: true },
    },
  },
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
})
