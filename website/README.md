# parry documentation site

The landing page and docs for parry, built with [Astro](https://astro.build)
and [Starlight](https://starlight.astro.build).

```bash
cd website
npm install
npm run dev      # http://localhost:4321/parry/
npm run build    # static output in dist/
```

## Layout

| Path | What it is |
|---|---|
| `src/pages/index.astro` | the landing page (custom, outside Starlight) |
| `src/styles/landing.css` | landing page styles and animations |
| `src/content/docs/docs/**` | documentation pages (MDX), served under `/docs/` |
| `src/components/docs/*` | interactive widgets: cache-key checker, token-bucket simulator, retry timeline, cost calculator, pricing table, request trace |
| `src/styles/docs.css` | the Starlight theme (colours, typography, widget styles) |
| `src/data/site.ts` | site helpers. Reads the pricing table from `../src/cost.rs` and benchmark numbers from `../README.md` at build time |
| `astro.config.mjs` | site config and the sidebar |

The pricing table and the benchmark figures are **generated from the library
sources**, so they cannot drift from the code. After updating prices in
`src/cost.rs`, or re-running `benches/overhead_bench.py --update-readme`,
rebuild the site.

## Deploying

`.github/workflows/docs.yml` builds the site on every PR that touches it and
deploys it to GitHub Pages on pushes to `main`. Enable it once under
**Settings → Pages → Source: GitHub Actions**.

The site is served from `/parry/` by default. For a custom domain, set
`DOCS_SITE` and `DOCS_BASE` when building:

```bash
DOCS_SITE=https://parry.dev DOCS_BASE=/ npm run build
```

## Writing docs

- Link between pages with root-relative paths, such as `[Budgets](/docs/features/budgets/)`.
  A remark plugin adds the base path. In component props, which the plugin
  doesn't see, use `siteUrl('docs/...')` from `src/data/site.ts`.
- Widgets are plain Astro components with a small inline script and no
  framework runtime. Each one honours `prefers-reduced-motion`.
