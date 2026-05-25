# services/frontend

Next.js 15 (App Router) + TypeScript + Tailwind CSS 3 frontend for DeepHorizon. Talks to the Go API gateway in `services/api/`.

**Owner:** Stajyer 7.

## Why Next.js (and not Vite)

- File-based routing and layouts beat hand-rolled React Router for an intern team.
- Familiar conventions — most fresh React developers in TR have seen Next.
- We do **not** use SSR data fetching, ISR, or API routes — the Go gateway is authoritative. Treat Next as a build tool + router.
- Production deploy uses `next start` in a Node container. Yes, that's one more pod than Vite would need; we accept the trade.

## Why Tailwind 3 (and not 4)

- v4 still has plugin/PostCSS churn; v3 is stable with mature docs and tutorials.
- Bump to 4 only after a Tailwind-4 retrospective from a project that already shipped it.

## Bootstrap

```bash
cd services/frontend
npx create-next-app@latest . \
  --typescript --tailwind --app --src-dir=false \
  --eslint --no-import-alias --no-turbopack

# pin Tailwind to 3 explicitly (create-next-app default is 4 in Next 15)
npm uninstall tailwindcss @tailwindcss/postcss postcss
npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p

# project deps
npm install zustand @tanstack/react-query three d3
npm install -D @types/three @types/d3
```

> The flags pick the **App Router**, **no `src/`**, **no `~` import alias**, and **no Turbopack** (Turbopack still has SSR edge cases as of Next 15). Adjust if you have strong reasons.

## Layout (target)

```
app/
  layout.tsx              # root layout (Tailwind globals, providers)
  page.tsx                # home — upload form
  enhance/[jobId]/page.tsx  # async job status + result viewer
  models/page.tsx         # model browser
  api/                    # DO NOT put backend logic here — Go gateway owns it
components/
  upload/                 # dropzone, progress, validation
  viewer/                 # before/after image viewer (Three.js)
  metrics/                # PSNR/SSIM/LPIPS chart components (D3)
  ui/                     # primitives (Button, Card, Dialog)
lib/
  api.ts                  # typed fetch client for the Go gateway
  store.ts                # Zustand stores
  query.ts                # TanStack Query config
  proto.ts                # optional: TS types mirrored from proto/
public/
  favicon.ico
  octapull-logo.svg
tailwind.config.ts
next.config.ts
```

## Dev workflow

```bash
npm run dev               # http://localhost:3000
npm run build && npm start  # production build sanity check
npm run lint
```

The Go gateway expects to live at `NEXT_PUBLIC_API_BASE_URL` — default `http://localhost:8080`. Put dev overrides in `.env.local` (gitignored).

## Production container

Multi-stage Dockerfile (`infra/docker/frontend.Dockerfile`):

1. Stage 1: `node:20-alpine` — `npm ci && npm run build`
2. Stage 2: `node:20-alpine` slim — copy `.next/standalone` + `public/` + `static/`, run `node server.js`

Use Next 15's `output: "standalone"` option in `next.config.ts` so the runtime image is ~80 MB instead of ~400 MB.

## What NOT to do

- Don't add `pages/`. App Router only.
- Don't call MLflow or the inference gRPC service directly from the browser. Always go through the Go gateway.
- Don't introduce SSR data fetching to "speed up" the upload page. There's nothing to render server-side; the user always uploads, then polls.
- Don't upgrade to Tailwind 4 mid-project.
