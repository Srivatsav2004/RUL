// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// replace target with your actual backend URL on Render
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "https://rul-3tso.onrender.com",
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, "")
      }
    }
  },
  build: {
    outDir: "dist"
  }
});
