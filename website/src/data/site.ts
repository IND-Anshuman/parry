import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

/** Join a site-relative path onto Astro's base (`/parry/` on GitHub Pages). */
export function url(path = ''): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const clean = path.replace(/^\//, '');
  return `${base}/${clean}`;
}

export const GITHUB = 'https://github.com/IND-Anshuman/parry';
export const PYPI_NAME = 'llm-parry';

/** Read the package version from pyproject.toml so the site never drifts from
 * the published package. */
function loadVersion(): string {
  const toml = readFileSync(resolve(process.cwd(), '../pyproject.toml'), 'utf8');
  return toml.match(/^version\s*=\s*"([^"]+)"/m)?.[1] ?? '0.0.0';
}

export const VERSION = loadVersion();

export interface PriceRow {
  provider: 'openai' | 'anthropic';
  model: string;
  input: number;
  output: number;
  longThreshold?: number;
  longInput?: number;
  longOutput?: number;
  /** Cache-read (hit) input rate, per million tokens. */
  cachedInput?: number;
  /** Anthropic 5-minute / 1-hour cache-write rates, per million input tokens. */
  cacheWrite5m?: number;
  cacheWrite1h?: number;
  /** Whole-call tier multipliers (batch is cheaper, fast is dearer). */
  batchMultiplier?: number;
  fastMultiplier?: number;
}

/**
 * Derive the tier rates the library computes in `Pricing::with_provider_tiers`
 * (src/cost.rs): cache read is 10% of input for both providers; Anthropic adds
 * 5-minute (1.25×) and 1-hour (2×) cache writes; batch is 50% off; fast is
 * 1.75× (OpenAI) or 2× (Anthropic). Keep these ratios in sync with the Rust.
 */
function withTiers(row: PriceRow): PriceRow {
  row.cachedInput = row.input * 0.1;
  row.batchMultiplier = 0.5;
  if (row.provider === 'anthropic') {
    row.cacheWrite5m = row.input * 1.25;
    row.cacheWrite1h = row.input * 2.0;
    row.fastMultiplier = 2.0;
  } else {
    row.fastMultiplier = 1.75;
  }
  return row;
}

/**
 * The pricing table is parsed straight out of the Rust source (src/cost.rs)
 * at build time, so the website can never disagree with the library.
 */
function loadPricing(): { rows: PriceRow[]; verified: string } {
  const source = readFileSync(resolve(process.cwd(), '../src/cost.rs'), 'utf8');
  const rows: PriceRow[] = [];
  const re =
    /p\("(openai|anthropic)",\s*"([^"]+)",\s*Pricing::(flat|tiered)\(\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d_]+),\s*([\d.]+),\s*([\d.]+))?\s*\)\)/g;
  for (const m of source.matchAll(re)) {
    const row: PriceRow = {
      provider: m[1] as PriceRow['provider'],
      model: m[2],
      input: Number(m[4]),
      output: Number(m[5]),
    };
    if (m[3] === 'tiered') {
      row.longThreshold = Number(m[6].replace(/_/g, ''));
      row.longInput = Number(m[7]);
      row.longOutput = Number(m[8]);
    }
    rows.push(withTiers(row));
  }
  const verified = source.match(/Pricing last verified (\d{4}-\d{2}-\d{2})/)?.[1] ?? 'unknown';
  if (rows.length === 0) throw new Error('could not parse any pricing rows from src/cost.rs');
  return { rows, verified };
}

export const pricing = loadPricing();

/** Numbers from benches/overhead_bench.py (see README "Benchmarks"). */
function loadBench(): { baseline: number; miss: number; hit: number; env: string } {
  const readme = readFileSync(resolve(process.cwd(), '../README.md'), 'utf8');
  const block = readme.match(/<!-- BENCH:START -->([\s\S]*?)<!-- BENCH:END -->/)?.[1] ?? '';
  const num = (label: RegExp) => Number(block.match(label)?.[1] ?? NaN);
  const baseline = num(/Unguarded[^|]*\|\s*([\d.]+)\s*µs/);
  const miss = num(/cache miss\s*\|\s*([\d.]+)\s*µs/);
  const hit = num(/cache hit[^|]*\|\s*([\d.]+)\s*µs/);
  const env = block.match(/^_(.+)_\s*$/m)?.[1] ?? '';
  return { baseline, miss, hit, env };
}

export const bench = loadBench();
