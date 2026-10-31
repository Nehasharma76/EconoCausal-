import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, requests to /api/... are forwarded to the Python API (uvicorn on port 8000).
// If the API is not running, the dashboard falls back to the saved results in /public.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, rewrite: (path) => path.replace(/^\/api/, "") },
    },
  },
});
