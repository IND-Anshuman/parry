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
export const VERSION = '0.1.0';

export interface PriceRow {
  provider: 'openai' | 'anthropic';
  model: string;
  input: number;
  output: number;
  longThreshold?: number;
  longInput?: number;
  longOutput?: number;
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
    rows.push(row);
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
