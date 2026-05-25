# services/frontend

React 19 + TypeScript SPA for uploading black hole images and viewing enhanced results.

**Owner:** Stajyer 7.

## Bootstrap

```bash
cd services/frontend
npm create vite@latest . -- --template react-ts
npm install
npm install zustand @tanstack/react-query three d3 tailwindcss
```

## Layout (target)

| Path | Purpose |
|:---|:---|
| `src/pages/` | Route components (upload, gallery, model details) |
| `src/components/` | Reusable UI |
| `src/api/` | Typed client for the Go gateway |
| `src/state/` | Zustand stores |
| `public/` | Static assets |
