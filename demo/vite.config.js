import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

const localModelDir = path.resolve(import.meta.dirname, '../runs/qwen35-countdown-100k-optimized');

export default defineConfig({
  base: '/countdown/',
  server: { port: 5173, strictPort: true },
  plugins: [{
    name: 'serve-local-model-in-development',
    configureServer(server) {
      server.middlewares.use('/countdown/shards/', (request, response, next) => {
        const name = path.basename(decodeURIComponent((request.url || '').split('?')[0]));
        if (!/^countdown-Q4_K_M-\d{5}-of-\d{5}\.gguf$/.test(name)) return next();
        const localModel = path.join(localModelDir, name);
        if (!fs.existsSync(localModel)) return next();
        const stat = fs.statSync(localModel);
        response.setHeader('Content-Type', 'application/octet-stream');
        response.setHeader('Content-Length', stat.size);
        fs.createReadStream(localModel).pipe(response);
      });
    },
  }],
});
