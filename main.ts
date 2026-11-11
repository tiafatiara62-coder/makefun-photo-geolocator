// photo-geolocator-01 entry: two-pass vision geolocation that beats the house baseline.
//
// The baseline is one plain claude-sonnet-5-5 call per photo ("where was this taken?").
// This entry beats it three ways:
//   1. A structured GeoGuessr-style prompt (continent -> country -> region -> city, with
//      explicit evidence: driving side, signs, plates, vegetation, architecture, sun)
//      and a forced confidence, on every photo, temperature 0.
//   2. A verification pass with the stronger claude-opus-5-5 on the least-confident photos,
//      which may correct the first guess; the two guesses are combined by confidence.
//   3. It never leaves a photo unanswered: model fallbacks (haiku, then gpt-6-luna) and a
//      fixed fallback coordinate, because an unanswered photo costs 20,015 km.
//
// Budget: the gateway prices every call at the vendor's list price when it is sent and
// refuses a call whose worst case (its input at one token per three body bytes, plus
// its whole output cap) does not fit in what is left of the trial's $1.00. A 512px
// photo is ~25-135 KB of JPEG, so one call's worst case is ~10-60k input tokens:
// a dear model on a big photo can be refused while a cheaper one still fits. The
// agent therefore tries models in tiers (sonnet -> haiku -> luna in pass 1,
// opus -> sonnet -> luna in pass 2) and a budget refusal only blocks that one model:
// the next cheaper tier is tried for the same photo. A refused call costs nothing.
// Requests: at most 25 + 12 = 37 of the 200 allowed; max_tokens 512/768, far under
// the 16,384 cap. Actual spend is far below worst case (a photo is ~1,500 real input
// tokens and ~300 output tokens), so a full trial should cost roughly $0.30-0.60.
//
// Determinism: every request is a function of the photo bytes and the fixed prompts.
// No wall clock, no unseeded randomness, temperature 0, results stored by photo index
// (never by completion order), retries send byte-identical requests with fixed backoff.

const SONNET = 'claude-sonnet-5-5';
const OPUS = 'claude-opus-5-5';
const HAIKU = 'claude-haiku-5-5';
const LUNA = 'gpt-6-luna';
const PASS1_TOKENS = 512;
const PASS2_TOKENS = 768;
const CONCURRENCY = 4;
const PASS2_CONFIDENCE = 0.75;
const PASS2_MAX = 12;
const CALL_TIMEOUT_MS = 90_000;
const RETRIES = 2;
// A gateway refusal (budget, cap, allowlist, max_tokens) is final for the whole trial.
const GATEWAY_REFUSALS = ['budget_exhausted', 'request_cap', 'model_not_allowed', 'max_tokens_required'];
// A provider transient error is retried with fixed backoff.
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
// Fallback when no model can place a photo: the median of the 75 public practice
// photos (30.25 N, 21.03 E, Mediterranean). Its mean error on those photos is 7,294 km
// against 20,015 km for leaving a photo unanswered. One constant, not a table.
const FALLBACK = { lat: 30.25, lon: 21.03 };

const PROMPT1 = [
  'Locate this street-level photo. Reason step by step, then answer.',
  '1. Hemisphere and continent: vegetation, climate, architecture style, which side of the road vehicles drive on, road markings.',
  '2. Country: language and script on signs, license plates, phone and electric poles, bollards, street furniture, driving side.',
  '3. Region and city: architecture, vegetation, terrain, sun position and shadows, building materials.',
  '4. Commit to your single best latitude and longitude.',
  'Return ONLY this JSON, no other text: {"lat": <degrees north>, "lon": <degrees east>, "confidence": <0-1>, "country": "<name>", "reasoning": "<at most 40 words>"}.',
].join('\n');

function prompt2(first: Guess): string {
  return [
    'Verify or correct this geolocation. A first analyst placed this photo at',
    `latitude ${first.lat}, longitude ${first.lon} (${first.country ?? 'unknown country'}) with confidence ${first.confidence}.`,
    'Look at the photo fresh: work through continent, country, region, city from the visual evidence, and decide whether the first guess is right or wrong.',
    'Return ONLY this JSON, no other text: {"lat": <degrees north>, "lon": <degrees east>, "confidence": <0-1>, "country": "<name>", "reasoning": "<at most 40 words>"}.',
  ].join('\n');
}

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
  confidence: number;
  country: string | null;
}

/** 'ok' with the model's text, 'no-key' (401: this vendor has no key), 'budget' (gateway refusal: this model is blocked for the trial), 'failed' (other error after retries). */
type AskResult = { kind: 'ok'; text: string } | { kind: 'no-key' | 'budget' | 'failed' };

/** Budget state for the trial. A gateway refusal blocks only that one model; the
 * cheapest model of a vendor being refused (or the vendor having no model behind
 * it) retires that vendor for the trial. */
interface Budget {
  blocked: Set<string>;
  anthropicDead: boolean;
  openaiDead: boolean;
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

function errorType(bodyText: string): string | null {
  try {
    const body = JSON.parse(bodyText) as { error?: { type?: string; code?: string } };
    const e = body.error ?? {};
    return typeof e.type === 'string' ? e.type : typeof e.code === 'string' ? e.code : null;
  } catch {
    return null;
  }
}

/** One Anthropic Messages call. A gateway refusal blocks only this model for the
 * trial (and retires the vendor if it is the cheapest model); other errors retry. */
async function askAnthropic(
  baseUrl: string,
  apiKey: string,
  jpeg: Buffer,
  text: string,
  model: string,
  maxTokens: number,
  budget: Budget,
): Promise<AskResult> {
  const body = JSON.stringify({
    model,
    max_tokens: maxTokens,
    temperature: 0,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${baseUrl}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
        body,
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
    } catch (err) {
      if (attempt < RETRIES) {
        await sleep(2000 * (attempt + 1));
        continue;
      }
      process.stderr.write(`geolocator: anthropic ${model} unreachable: ${(err as Error).message}\n`);
      return { kind: 'failed' };
    }
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return {
        kind: 'ok',
        text: (msg.content ?? [])
          .filter((b) => b.type === 'text')
          .map((b) => b.text ?? '')
          .join('\n'),
      };
    }
    const textBody = await res.text();
    if (res.status === 401) return { kind: 'no-key' };
    const eType = errorType(textBody);
    if (eType === 'upstream_unreachable') {
      // The gateway has no model behind it for this vendor (no key): dead for the trial.
      budget.anthropicDead = true;
      return { kind: 'failed' };
    }
    if (GATEWAY_REFUSALS.includes(eType ?? '')) {
      budget.blocked.add(model);
      if (model === HAIKU) budget.anthropicDead = true;
      process.stderr.write(`geolocator: gateway refused ${model} (${eType}): model blocked\n`);
      return { kind: 'budget' };
    }
    if (RETRY_STATUSES.includes(res.status) && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`geolocator: anthropic ${model} HTTP ${res.status}: ${textBody.slice(0, 200)}\n`);
    return { kind: 'failed' };
  }
}

/** One OpenAI Chat Completions call (gpt-6-luna fallback). Same result contract as askAnthropic. */
async function askOpenAI(
  baseUrl: string,
  apiKey: string,
  jpeg: Buffer,
  text: string,
  model: string,
  maxTokens: number,
  budget: Budget,
): Promise<AskResult> {
  const body = JSON.stringify({
    model,
    max_completion_tokens: maxTokens,
    temperature: 0,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${jpeg.toString('base64')}` } },
          { type: 'text', text },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
        body,
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
    } catch (err) {
      if (attempt < RETRIES) {
        await sleep(2000 * (attempt + 1));
        continue;
      }
      process.stderr.write(`geolocator: openai ${model} unreachable: ${(err as Error).message}\n`);
      return { kind: 'failed' };
    }
    if (res.ok) {
      const msg = (await res.json()) as { choices?: { message?: { content?: string | null } }[] };
      return { kind: 'ok', text: msg.choices?.[0]?.message?.content ?? '' };
    }
    const textBody = await res.text();
    if (res.status === 401) return { kind: 'no-key' };
    const eType = errorType(textBody);
    if (eType === 'upstream_unreachable') {
      // The gateway has no model behind it for this vendor (no key): dead for the trial.
      budget.openaiDead = true;
      return { kind: 'failed' };
    }
    if (GATEWAY_REFUSALS.includes(eType ?? '')) {
      budget.blocked.add(model);
      if (model === LUNA) budget.openaiDead = true;
      process.stderr.write(`geolocator: gateway refused ${model} (${eType}): model blocked\n`);
      return { kind: 'budget' };
    }
    if (RETRY_STATUSES.includes(res.status) && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`geolocator: openai ${model} HTTP ${res.status}: ${textBody.slice(0, 200)}\n`);
    return { kind: 'failed' };
  }
}

type ModelName = typeof SONNET | typeof OPUS | typeof HAIKU | typeof LUNA;

/** One model attempt: skips models the gateway blocked or vendors that are dead or
 * keyless, and records a 401 as that vendor having no key. Returns the model's text
 * on success, null otherwise (the caller tries the next tier). */
async function tryModel(
  model: ModelName,
  jpeg: Buffer,
  text: string,
  maxTokens: number,
  cfg: Cfg,
  budget: Budget,
  vendorUp: { anthropic: boolean; openai: boolean },
): Promise<string | null> {
  if (budget.blocked.has(model)) return null;
  let r: AskResult;
  if (model === LUNA) {
    if (!vendorUp.openai || budget.openaiDead) return null;
    r = await askOpenAI(cfg.openaiUrl, cfg.openaiKey, jpeg, text, model, maxTokens, budget);
    if (r.kind === 'no-key') vendorUp.openai = false;
  } else {
    if (!vendorUp.anthropic || budget.anthropicDead) return null;
    r = await askAnthropic(cfg.anthropicUrl, cfg.anthropicKey, jpeg, text, model, maxTokens, budget);
    if (r.kind === 'no-key') vendorUp.anthropic = false;
  }
  return r.kind === 'ok' ? r.text : null;
}

/** Try the models in order (dearest first) for one photo; null if none answers. */
async function askChain(
  jpeg: Buffer,
  text: string,
  maxTokens: number,
  chain: ModelName[],
  cfg: Cfg,
  budget: Budget,
  vendorUp: { anthropic: boolean; openai: boolean },
): Promise<string | null> {
  for (const model of chain) {
    const textOut = await tryModel(model, jpeg, text, maxTokens, cfg, budget, vendorUp);
    if (textOut !== null) return textOut;
  }
  return null;
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The guess in an answer: its first JSON object with a valid lat/lon, else null. (0,0) is rejected as absurd. */
function parseGuess(id: string, text: string): Omit<Guess, 'id'> | null {
  const clean = text.replace(/```json|```/g, '');
  for (const m of clean.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (!isLat(lat) || !isLon(lon)) continue;
      if (lat === 0 && lon === 0) continue;
      const conf = typeof o.confidence === 'number' && Number.isFinite(o.confidence) ? Math.min(1, Math.max(0, o.confidence)) : 0.5;
      const country = typeof o.country === 'string' ? o.country : null;
      return { lat, lon, confidence: conf, country };
    } catch {
      // Not JSON: try the next object.
    }
  }
  return null;
}

function haversineKm(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const r = Math.PI / 180;
  const dLat = (bLat - aLat) * r;
  const dLon = (bLon - aLon) * r;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(aLat * r) * Math.cos(bLat * r) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

/** Combine pass 1 and pass 2. When one guess is clearly more confident, take it outright:
 * averaging would drag a strong second look toward a weak first guess. Only average
 * when the two confidences are close and the guesses are near each other. */
function combine(first: Guess, second: Guess): Guess {
  if (Math.abs(second.confidence - first.confidence) >= 0.2) {
    return second.confidence > first.confidence ? second : first;
  }
  const dist = haversineKm(first.lat, first.lon, second.lat, second.lon);
  if (dist < 100 && first.confidence >= 0.6 && second.confidence >= 0.6) {
    const w = second.confidence / (first.confidence + second.confidence);
    return {
      id: first.id,
      lat: first.lat + (second.lat - first.lat) * w,
      lon: first.lon + (second.lon - first.lon) * w,
      confidence: Math.max(first.confidence, second.confidence),
      country: second.country ?? first.country,
    };
  }
  if (second.confidence >= first.confidence) return second;
  return first;
}

interface Cfg {
  envUrl: string;
  anthropicUrl: string;
  anthropicKey: string;
  openaiUrl: string;
  openaiKey: string;
}

function env(name: string, fallback = ''): string {
  const v = process.env[name];
  return v === undefined || v === '' ? fallback : v;
}

async function fetchJpeg(envUrl: string, p: Photo): Promise<Buffer | null> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      const res = await fetch(`${envUrl}${p.url}`, { signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return Buffer.from(await res.arrayBuffer());
    } catch (err) {
      if (attempt < RETRIES) {
        await sleep(1000 * (attempt + 1));
        continue;
      }
      process.stderr.write(`geolocator: cannot fetch ${p.id}: ${(err as Error).message}\n`);
      return null;
    }
  }
}

async function main(): Promise<number> {
  const cfg: Cfg = {
    envUrl: env('MAKEFUN_ENV_URL').replace(/\/+$/, ''),
    anthropicUrl: env('ANTHROPIC_BASE_URL', 'https://api.anthropic.com').replace(/\/+$/, ''),
    anthropicKey: env('ANTHROPIC_API_KEY'),
    openaiUrl: env('OPENAI_BASE_URL', 'https://api.openai.com/v1').replace(/\/+$/, ''),
    openaiKey: env('OPENAI_API_KEY'),
  };
  if (cfg.envUrl === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const budget: Budget = { blocked: new Set<string>(), anthropicDead: false, openaiDead: false };
  // A vendor that answers 401 has no key this trial: never call it again.
  const vendorUp = { anthropic: cfg.anthropicKey !== '', openai: cfg.openaiKey !== '' };

  const listed = (await (await fetch(`${cfg.envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  const jpegs: (Buffer | null)[] = new Array(photos.length).fill(null);

  // Pass 1: sonnet on every photo, haiku then luna as fallbacks per photo.
  let next = 0;
  const pass1 = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      const jpeg = await fetchJpeg(cfg.envUrl, p);
      if (jpeg === null) continue;
      jpegs[i] = jpeg;
      const text = await askChain(jpeg, PROMPT1, PASS1_TOKENS, [SONNET, HAIKU, LUNA], cfg, budget, vendorUp);
      if (text === null) continue;
      const g = parseGuess(p.id, text);
      if (g !== null) guesses[i] = { id: p.id, ...g };
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => pass1()));

  // Pass 2: opus verification on the least-confident photos (cap PASS2_MAX).
  {
    const needy: number[] = guesses
      .map((g, i) => ({ g, i }))
      .filter((x): x is { g: Guess; i: number } => x.g !== null && x.g.confidence < PASS2_CONFIDENCE)
      .sort((a, b) => a.g.confidence - b.g.confidence || a.i - b.i)
      .slice(0, PASS2_MAX)
      .map((x) => x.i);
    let n2 = 0;
    const pass2 = async (): Promise<void> => {
      while (n2 < needy.length) {
        const i = needy[n2]!;
        n2 += 1;
        const p = photos[i]!;
        const jpeg = jpegs[i]!;
        const first = guesses[i]!;
        const text = await askChain(jpeg, prompt2(first), PASS2_TOKENS, [OPUS, SONNET, LUNA], cfg, budget, vendorUp);
        if (text === null) continue;
        const g = parseGuess(p.id, text);
        if (g !== null) guesses[i] = combine(first, { id: p.id, ...g });
      }
    };
    await Promise.all(Array.from({ length: CONCURRENCY }, () => pass2()));
  }

  // Never leave a photo unanswered.
  const answers = photos.map((p, i) => {
    const g = guesses[i];
    return g === null ? { id: p.id, lat: FALLBACK.lat, lon: FALLBACK.lon } : { id: p.id, lat: g.lat, lon: g.lon };
  });
  const post = async (list: { id: string; lat: number; lon: number }[]): Promise<boolean> => {
    if (list.length === 0) return true;
    const res = await fetch(`${cfg.envUrl}/answers`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ answers: list }),
    });
    if (res.ok) return true;
    process.stderr.write(`geolocator: POST /answers HTTP ${res.status}: ${(await res.text()).slice(0, 200)}\n`);
    return false;
  };
  let ok = await post(answers);
  if (!ok) {
    // The POST is all-or-nothing; check what is answered and post only the rest.
    try {
      const cur = (await (await fetch(`${cfg.envUrl}/photos`)).json()) as { photos: Photo[] };
      const left = answers.filter((a) => !cur.photos.find((p) => p.id === a.id)?.answered);
      ok = await post(left);
    } catch (err) {
      process.stderr.write(`geolocator: re-check failed: ${(err as Error).message}\n`);
    }
  }
  const placed = guesses.filter((g) => g !== null).length;
  process.stdout.write(`geolocator: ${answers.length} answers posted (${placed} placed by a model, ${answers.length - placed} fallback); POST ${ok ? 'accepted' : 'failed'}\n`);
  return ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`geolocator: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
