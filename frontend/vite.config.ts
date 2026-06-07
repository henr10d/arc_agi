import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    watch: {
      ignored: [
        '**/node_modules/**',
        '**/dist/**',
        '../.git/**',
        '../__pycache__/**',
        '../all/**',
        '../arc_synth_experiment/**',
        '../attention_maps/**',
        '../checkpoints/**',
        '../data/**',
        '../onnx/**',
        '../out/**',
        '../submission/**',
        '../tests/**'
      ]
    },
    proxy: {
      '/tasks': 'http://127.0.0.1:8000',
      '/projects': 'http://127.0.0.1:8000',
      '/run': 'http://127.0.0.1:8000',
      '/compile': 'http://127.0.0.1:8000',
      '/export': 'http://127.0.0.1:8000'
    }
  }
});
