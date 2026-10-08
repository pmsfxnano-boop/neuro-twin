import type { CapabilitySet, RuntimeSnapshot } from '../types';

const rawApiBase = import.meta.env.VITE_NEURO_TWIN_API_URL;
const DEFAULT_API_BASE = typeof rawApiBase === 'string' ? rawApiBase.replace(/\/$/, '') : '';

export class RuntimeApi {
  constructor(public readonly base = DEFAULT_API_BASE) {}
  async status(): Promise<RuntimeSnapshot> { return this.get('/v1/runtime/status'); }
  async capabilities(): Promise<CapabilitySet> { return this.get('/v1/capabilities'); }
  async runRuntime(payload: unknown): Promise<{status:string;runtime_id:string;summary:RuntimeSnapshot}> {
    const r = await fetch(`${this.base}/v1/runtime/run`, {
      method: 'POST',
      headers: {'content-type':'application/json', accept:'application/json'},
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      let detail = `runtime HTTP ${r.status}`;
      try { const body = await r.json(); detail = body.detail ?? detail; } catch { /* keep status */ }
      throw new Error(detail);
    }
    return r.json();
  }
  async refresh(): Promise<{status:string;summary:RuntimeSnapshot}> {
    const r = await fetch(`${this.base}/v1/runtime/refresh`, {method:'POST', headers:{accept:'application/json'}});
    if (!r.ok) throw new Error(`refresh HTTP ${r.status}`);
    return r.json();
  }
  wsUrl(path='/ws'): string {
    if (this.base.startsWith('http://')) return this.base.replace(/^http:/,'ws:') + path;
    if (this.base.startsWith('https://')) return this.base.replace(/^https:/,'wss:') + path;
    return `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${path}`;
  }
  private async get<T>(path:string): Promise<T> {
    const r = await fetch(`${this.base}${path}`, {headers:{accept:'application/json'}});
    if (!r.ok) throw new Error(`GET ${path} HTTP ${r.status}`);
    return r.json();
  }
}
