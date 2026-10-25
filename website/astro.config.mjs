// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { remarkBaseLinks } from './src/plugins/remark-base-links.mjs';

// The site is served from https://www.anshumanbuilds.dev/parry/ (the portfolio
// proxies /parry to this project). Override with DOCS_SITE / DOCS_BASE for
// another host (e.g. DOCS_SITE=https://parry.dev DOCS_BASE=/).
const site = process.env.DOCS_SITE ?? 'https://www.anshumanbuilds.dev';
const base = process.env.DOCS_BASE ?? '/parry';

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  markdown: {
    remarkPlugins: [remarkBaseLinks(base)],
  },
  vite: {
    // File watching across a Docker bind mount on Windows needs polling.
    server: process.env.POLL ? { watch: { usePolling: true, interval: 400 } } : {},
  },
  integrations: [
    starlight({
      title: 'parry',
      description:
        'Zero-config rate limiting, caching, and cost control for LLM calls. One line to install, no rewrites.',
      logo: { src: './src/assets/logo.svg', replacesTitle: false },
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/IND-Anshuman/parry' }],
      editLink: { baseUrl: 'https://github.com/IND-Anshuman/parry/edit/main/website/' },
      lastUpdated: false,
      customCss: [
        '@fontsource-variable/inter',
        '@fontsource-variable/jetbrains-mono',
        './src/styles/docs.css',
      ],
      expressiveCode: {
        themes: ['github-dark-default', 'github-light'],
        styleOverrides: {
          borderRadius: '0.75rem',
          codeFontFamily: "'JetBrains Mono Variable', ui-monospace, monospace",
          frames: { shadowColor: 'transparent' },
        },
      },
      components: {
        Footer: './src/components/docs/Footer.astro',
      },
      sidebar: [
        {
          label: 'Get started',
          items: [
            { label: 'Introduction', slug: 'docs' },
            { label: 'Quickstart', slug: 'docs/quickstart' },
            { label: 'Installation', slug: 'docs/installation' },
            { label: 'How it works', slug: 'docs/how-it-works' },
          ],
        },
        {
          label: 'Features',
          items: [
            { label: 'Response caching', slug: 'docs/features/caching' },
            { label: 'Rate limiting', slug: 'docs/features/rate-limiting' },
            { label: 'Cost tracking', slug: 'docs/features/cost-tracking' },
            { label: 'Budgets', slug: 'docs/features/budgets' },
            { label: 'Retries & backoff', slug: 'docs/features/retries' },
            { label: 'Streaming', slug: 'docs/features/streaming' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Scoping a guard', slug: 'docs/guides/scoping' },
            { label: 'Persistence & CLI', slug: 'docs/guides/persistence' },
            { label: 'Stats & logging', slug: 'docs/guides/observability' },
            { label: 'LangChain', slug: 'docs/guides/langchain' },
            { label: 'LlamaIndex', slug: 'docs/guides/llamaindex' },
            { label: 'Custom clients & primitives', slug: 'docs/guides/custom-clients' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'Configuration', slug: 'docs/reference/configuration' },
            { label: 'Python API', slug: 'docs/reference/api' },
            { label: 'Pricing table', slug: 'docs/reference/pricing' },
            { label: 'Failure modes', slug: 'docs/reference/failure-modes' },
            { label: 'CLI', slug: 'docs/reference/cli' },
          ],
        },
        {
          label: 'Project',
          items: [
            { label: 'Benchmarks', slug: 'docs/benchmarks' },
            { label: 'FAQ', slug: 'docs/faq' },
            { label: 'Changelog', slug: 'docs/changelog' },
          ],
        },
      ],
    }),
  ],
});
