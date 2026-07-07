import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5173,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ['recharts'],
          icons: ['lucide-react'],
          motion: ['framer-motion'],
          router: ['react-router', 'react-router-dom'],
          query: ['@tanstack/react-query', '@tanstack/react-query-persist-client', '@tanstack/query-sync-storage-persister'],
        },
      },
    },
    chunkSizeWarningLimit: 600,
  },
});
