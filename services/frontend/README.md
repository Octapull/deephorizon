# DeepHorizon Frontend

Next.js 15, TypeScript and Tailwind CSS frontend for the DeepHorizon image enhancement platform.

## Development

```bash
npm install
npm run dev
```

The application runs at `http://localhost:3000`. The Go gateway URL is configured with `NEXT_PUBLIC_API_BASE_URL` and defaults to `http://localhost:8080`.

## Commands

```bash
npm run dev
npm run build
npm run lint
npm run typecheck
```

## Structure

- `app/` contains routes and layouts.
- `components/` contains reusable UI and workspace components.
- `lib/` contains API contracts, mock data and shared utilities.
- `public/` contains the sample observation assets.

The current workspace uses a local mock inference flow so UI development can continue while the Go gateway and inference service contracts are finalized.
