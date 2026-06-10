"use client";
import { useEffect, useState } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────
interface EtfSignal {
  ticker: string; sector: string;
  current_rv: number; baseline_rv: number; z_score: number;
  vrp: number; beta_30d: number; consec_vol: number; excess_10d: number;
  level: number; label: string; reason: string;
}
interface OilPrice { label: string; current: number; baseline: number; z_score: number; bw_spread: number; }
interface OilInv   { latest_mbbl: number; wow_chg_pct: number; surprise: number; }
interface OilData  { wti?: OilPrice; brent?: OilPrice; inv?: OilInv; level: number; reasons: string[]; }
interface FredItem { sid: string; label: string; desc: string; etfs: string[]; current: number; baseline: number; z_score: number; last_date: string; }
interface ChinaItem { sym: string; label: string; desc: string; etfs: string[]; current: number; baseline: number; z_score: number; rv_20d: number | null; last_date: string; }
interface MonitorData {
  scan_date: string;
  etf_signals: EtfSignal[];
  oil: OilData;
  fred: FredItem[];
  china: ChinaItem[];
  flagged_count: number; alert_count: number; high_count: number;
}

// ── Category map ──────────────────────────────────────────────────────────────
const CATEGORY: Record<string, string> = {
  XLE:"US Sector", XLF:"US Sector", SOXX:"US Sector", XTN:"US Sector",
  XLK:"US Sector", XLV:"US Sector", XAR:"US Sector", XLB:"US Sector",
  XLI:"US Sector", XLC:"US Sector", XLY:"US Sector", GLD:"US Sector", TLT:"US Sector",
  "CL=F":"Futures", "ES=F":"Futures", "GC=F":"Futures", "ZN=F":"Futures",
  EWY:"Intl ETF", EWJ:"Intl ETF", FXI:"Intl ETF", MCHI:"Intl ETF", EWU:"Intl ETF", EEM:"Intl ETF",
  "^KS11":"Global Index", "^HSI":"Global Index", "^FTSE":"Global Index",
  "^FTMC":"Global Index", "^N225":"Global Index", "^STOXX50E":"Global Index", "000001.SS":"Global Index",
};

const CAT_ORDER = ["US Sector", "Futures", "Intl ETF", "Global Index"];
const CAT_COLOR: Record<string, string> = {
  "US Sector":   "text-blue-400",
  "Futures":     "text-purple-400",
  "Intl ETF":    "text-emerald-400",
  "Global Index":"text-amber-400",
};

// ── Helpers ───────────────────────────────────────────────────────────────────
const LEVEL_COLOR = ["text-gray-500", "text-yellow-400", "text-orange-400", "text-red-400"];
const LEVEL_BG    = ["", "bg-yellow-400/10", "bg-orange-400/10", "bg-red-400/10"];
const LEVEL_BADGE = ["---", "WATCH", "ALERT", "HIGH"];

function zColor(z: number) {
  if (z >= 2)    return "text-red-400";
  if (z >= 1.5)  return "text-orange-400";
  if (z >= 1)    return "text-yellow-400";
  if (z <= -1.5) return "text-emerald-400";
  return "text-gray-300";
}

function Stat({ label, value, sub, accent }: { label: string; value: string | number; sub?: string; accent?: string }) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-2xl font-bold ${accent ?? "text-gray-100"}`}>{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  );
}

function Badge({ level }: { level: number }) {
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded ${LEVEL_BG[level]} ${LEVEL_COLOR[level]}`}>
      {LEVEL_BADGE[level]}
    </span>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function MonitorPage() {
  const [data, setData] = useState<MonitorData | null>(null);
  const [err,  setErr]  = useState("");

  useEffect(() => {
    fetch("/data/vol_monitor_data.json")
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(setData)
      .catch(e => setErr(String(e)));
  }, []);

  if (err)   return <div className="text-red-400 p-8">{err}</div>;
  if (!data) return <div className="text-gray-500 p-8 animate-pulse">Loading monitor data...</div>;

  const scanTime = new Date(data.scan_date).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" });
  const flagged  = data.etf_signals.filter(r => r.level >= 1);
  const hasOil   = data.oil && (data.oil.wti || data.oil.brent);
  const hasFred  = data.fred && data.fred.length > 0;
  const hasChina = data.china && data.china.length > 0;

  // Group ETF signals by category, preserving pipeline sort order within group
  const grouped: Record<string, EtfSignal[]> = {};
  for (const cat of CAT_ORDER) grouped[cat] = [];
  for (const sig of data.etf_signals) {
    const cat = CATEGORY[sig.ticker] ?? "US Sector";
    grouped[cat].push(sig);
  }

  return (
    <div className="space-y-8">

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Vol Anomaly Monitor</h1>
          <p className="text-gray-400 text-sm mt-1">
            US 섹터 · 선물 · 국제 ETF · 글로벌 지수 변동성 이상 탐지 + EIA + FRED + 중국 지수
          </p>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500">Last scan</div>
          <div className="text-sm text-gray-300">{scanTime}</div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat label="Flagged" value={data.flagged_count}
              accent={data.flagged_count > 0 ? "text-yellow-400" : "text-gray-100"}
              sub="WATCH or above" />
        <Stat label="Alert" value={data.alert_count}
              accent={data.alert_count > 0 ? "text-orange-400" : "text-gray-100"}
              sub="ALERT level" />
        <Stat label="High" value={data.high_count}
              accent={data.high_count > 0 ? "text-red-400" : "text-gray-100"}
              sub="HIGH level" />
        <Stat label="Markets" value={data.etf_signals.length}
              sub={`US ${grouped["US Sector"].length} · Futures ${grouped["Futures"].length} · Intl ${grouped["Intl ETF"].length + grouped["Global Index"].length}`} />
      </div>

      {/* ETF Signal Table — grouped by category */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-800 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">Volatility Signals</h2>
          <span className="text-xs text-gray-500">vol window 10d vs 1yr baseline  |  z &ge; 1.8 = WATCH</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500">
                <th className="text-left px-4 py-2 font-medium w-24">Ticker</th>
                <th className="text-left px-4 py-2 font-medium">Sector / Name</th>
                <th className="text-right px-4 py-2 font-medium">RV</th>
                <th className="text-right px-4 py-2 font-medium">Base</th>
                <th className="text-right px-4 py-2 font-medium">Z</th>
                <th className="text-right px-4 py-2 font-medium">VRP</th>
                <th className="text-right px-4 py-2 font-medium">Beta</th>
                <th className="text-right px-4 py-2 font-medium">Excess 10d</th>
                <th className="text-center px-4 py-2 font-medium">Status</th>
                <th className="text-left px-4 py-2 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody>
              {CAT_ORDER.map(cat => {
                const rows = grouped[cat];
                if (rows.length === 0) return null;
                return (
                  <>
                    {/* Category separator */}
                    <tr key={`sep-${cat}`} className="border-b border-gray-800/60 bg-gray-900/60">
                      <td colSpan={10} className={`px-4 py-1.5 text-xs font-semibold tracking-wider uppercase ${CAT_COLOR[cat]}`}>
                        {cat}
                      </td>
                    </tr>
                    {rows.map(r => (
                      <tr key={r.ticker}
                          className={`border-b border-gray-800/40 hover:bg-gray-800/30 transition-colors ${LEVEL_BG[r.level]}`}>
                        <td className="px-4 py-2 font-semibold text-gray-100">{r.ticker}</td>
                        <td className="px-4 py-2 text-gray-400">{r.sector}</td>
                        <td className={`px-4 py-2 text-right font-mono ${zColor(r.z_score)}`}>{r.current_rv}%</td>
                        <td className="px-4 py-2 text-right font-mono text-gray-500">{r.baseline_rv}%</td>
                        <td className={`px-4 py-2 text-right font-mono font-semibold ${zColor(r.z_score)}`}>
                          {r.z_score > 0 ? "+" : ""}{r.z_score.toFixed(2)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono ${r.vrp < -0.05 ? "text-red-400" : "text-gray-400"}`}>
                          {r.vrp.toFixed(3)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono ${Math.abs(r.beta_30d - 1) > 0.3 ? "text-orange-400" : "text-gray-400"}`}>
                          {r.beta_30d.toFixed(2)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono ${r.excess_10d > 0 ? "text-emerald-400" : "text-red-400"}`}>
                          {r.excess_10d > 0 ? "+" : ""}{r.excess_10d.toFixed(1)}%
                        </td>
                        <td className="px-4 py-2 text-center"><Badge level={r.level} /></td>
                        <td className="px-4 py-2 text-gray-500">{r.reason || "-"}</td>
                      </tr>
                    ))}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Oil + FRED + China */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* EIA Oil */}
        {hasOil ? (
          <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-200">Oil Fundamentals (EIA)</h2>
              <Badge level={data.oil.level ?? 0} />
            </div>
            {(["wti","brent"] as const).map(key => {
              const d = data.oil[key];
              if (!d) return null;
              return (
                <div key={key} className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-gray-300">{d.label} Spot</span>
                    <span className={`text-xs font-mono ${zColor(d.z_score)}`}>z = {d.z_score > 0 ? "+" : ""}{d.z_score.toFixed(2)}</span>
                  </div>
                  <div className="flex items-end gap-3">
                    <span className="text-xl font-bold text-gray-100">${d.current.toFixed(2)}</span>
                    <span className="text-xs text-gray-500 mb-0.5">/ bbl  (base ${d.baseline.toFixed(0)})</span>
                  </div>
                  <div className={`text-xs mt-1 ${d.bw_spread >= 5 ? "text-orange-400" : "text-gray-500"}`}>
                    vs 60d MA: {d.bw_spread > 0 ? "+" : ""}{d.bw_spread.toFixed(1)}%
                    {d.bw_spread >= 5 && " ← backwardation"}
                  </div>
                </div>
              );
            })}
            {data.oil.inv && (
              <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
                <div className="text-xs font-semibold text-gray-300 mb-2">US Crude Inventory</div>
                <div className="flex gap-6 text-xs">
                  <div>
                    <div className="text-gray-500">Latest</div>
                    <div className="text-gray-100 font-mono">{data.oil.inv.latest_mbbl.toLocaleString()} Mbbl</div>
                  </div>
                  <div>
                    <div className="text-gray-500">WoW</div>
                    <div className={`font-mono ${data.oil.inv.wow_chg_pct < 0 ? "text-red-400" : "text-emerald-400"}`}>
                      {data.oil.inv.wow_chg_pct > 0 ? "+" : ""}{data.oil.inv.wow_chg_pct.toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">vs Avg</div>
                    <div className={`font-mono ${Math.abs(data.oil.inv.surprise) >= 3 ? "text-orange-400" : "text-gray-300"}`}>
                      {data.oil.inv.surprise > 0 ? "+" : ""}{data.oil.inv.surprise.toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            )}
            {data.oil.reasons && data.oil.reasons.length > 0 && (
              <div className="text-xs text-orange-400">{data.oil.reasons.join("  |  ")}</div>
            )}
          </div>
        ) : (
          <div className="rounded-xl border border-gray-800 bg-gray-900/20 p-5 flex items-center justify-center text-gray-600 text-sm">
            EIA data unavailable
          </div>
        )}

        {/* FRED Macro */}
        {hasFred ? (
          <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-3">
            <h2 className="text-sm font-semibold text-gray-200">Macro Stress (FRED)</h2>
            {data.fred.map(d => {
              const isAlert = Math.abs(d.z_score) >= 1.8;
              return (
                <div key={d.sid} className={`rounded-lg border p-3 ${isAlert ? "border-orange-800 bg-orange-400/5" : "border-gray-800 bg-gray-900/40"}`}>
                  <div className="flex items-center justify-between mb-1">
                    <div>
                      <span className="text-xs font-semibold text-gray-200">{d.label}</span>
                      <span className="text-xs text-gray-500 ml-2">{d.desc}</span>
                    </div>
                    <span className={`text-xs font-mono font-semibold ${zColor(d.z_score)}`}>
                      z = {d.z_score > 0 ? "+" : ""}{d.z_score.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-3 text-xs">
                    <span className="text-gray-100 font-mono text-sm">{d.current.toFixed(2)}</span>
                    <span className="text-gray-500">base {d.baseline.toFixed(2)}</span>
                    <span className="text-gray-600 ml-auto">{d.etfs.join(", ")}  ·  {d.last_date}</span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="rounded-xl border border-gray-800 bg-gray-900/20 p-5 flex items-center justify-center text-gray-600 text-sm">
            FRED data unavailable
          </div>
        )}

        {/* China Indices */}
        {hasChina ? (
          <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-3">
            <h2 className="text-sm font-semibold text-gray-200">China Indices (akshare)</h2>
            {data.china.map(d => {
              const isAlert = Math.abs(d.z_score) >= 1.8;
              return (
                <div key={d.sym} className={`rounded-lg border p-3 ${isAlert ? "border-orange-800 bg-orange-400/5" : "border-gray-800 bg-gray-900/40"}`}>
                  <div className="flex items-center justify-between mb-1">
                    <div>
                      <span className="text-xs font-semibold text-gray-200">{d.label}</span>
                      <span className="text-xs text-gray-500 ml-2">{d.desc}</span>
                    </div>
                    <span className={`text-xs font-mono font-semibold ${zColor(d.z_score)}`}>
                      z = {d.z_score > 0 ? "+" : ""}{d.z_score.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-3 text-xs">
                    <span className="text-gray-100 font-mono text-sm">{d.current.toFixed(0)}</span>
                    <span className="text-gray-500">base {d.baseline.toFixed(0)}</span>
                    {d.rv_20d !== null && (
                      <span className={`font-mono ml-auto ${d.rv_20d > 25 ? "text-orange-400" : "text-gray-400"}`}>
                        RV {d.rv_20d.toFixed(1)}%
                      </span>
                    )}
                  </div>
                  <div className="flex items-center justify-between mt-1 text-xs text-gray-600">
                    <span>{d.etfs.join(", ")}</span>
                    <span>{d.last_date}</span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="rounded-xl border border-gray-800 bg-gray-900/20 p-5 flex items-center justify-center text-gray-600 text-sm">
            China data unavailable
          </div>
        )}
      </div>

      {/* Flagged Detail */}
      {flagged.length > 0 && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-4">
          <h2 className="text-sm font-semibold text-gray-200">
            Flagged ({flagged.length}) — investigate for potential events
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {flagged.map(r => {
              const fredForTicker = (data.fred ?? []).filter(
                f => f.etfs.includes(r.ticker) || f.etfs.includes("ALL")
              ).filter(f => Math.abs(f.z_score) >= 1.3);
              const oilForTicker = r.ticker === "XLE" && hasOil ? data.oil : null;
              const cat = CATEGORY[r.ticker] ?? "US Sector";
              return (
                <div key={r.ticker}
                     className={`rounded-lg border p-4 space-y-2 ${LEVEL_BG[r.level]} ${
                       r.level >= 2 ? "border-orange-800" : "border-yellow-900"
                     }`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-gray-100">{r.ticker}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded bg-gray-800 ${CAT_COLOR[cat]}`}>{cat}</span>
                      <span className="text-xs text-gray-500">{r.sector}</span>
                    </div>
                    <Badge level={r.level} />
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <div className="text-gray-500">RV / Base</div>
                      <div className={`font-mono ${zColor(r.z_score)}`}>{r.current_rv}% / {r.baseline_rv}%</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Z-score</div>
                      <div className={`font-mono font-semibold ${zColor(r.z_score)}`}>
                        {r.z_score > 0 ? "+" : ""}{r.z_score.toFixed(2)}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-500">Beta 30d</div>
                      <div className={`font-mono ${Math.abs(r.beta_30d - 1) > 0.3 ? "text-orange-400" : "text-gray-300"}`}>
                        {r.beta_30d.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <div className="text-xs">
                    <span className="text-gray-500">10d vs SPY: </span>
                    <span className={r.excess_10d > 0 ? "text-emerald-400" : "text-red-400"}>
                      {r.excess_10d > 0 ? "+" : ""}{r.excess_10d.toFixed(1)}%
                    </span>
                  </div>
                  {r.reason && <div className="text-xs text-gray-500">{r.reason}</div>}
                  {oilForTicker?.wti && (
                    <div className="text-xs text-orange-300">
                      WTI ${oilForTicker.wti.current.toFixed(0)}/bbl  z={oilForTicker.wti.z_score > 0 ? "+" : ""}{oilForTicker.wti.z_score.toFixed(1)}
                      {oilForTicker.wti.bw_spread >= 5 ? "  backwardation" : ""}
                    </div>
                  )}
                  {fredForTicker.map(f => (
                    <div key={f.sid} className="text-xs text-yellow-300">
                      {f.label}: {f.current.toFixed(2)} (z={f.z_score > 0 ? "+" : ""}{f.z_score.toFixed(2)})  {f.desc}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <p className="text-xs text-gray-600">
        Parameters: vol window 10d vs 252d baseline  |  ALERT z &ge; 1.8
        {hasOil ? "  |  EIA: WTI/Brent spot + weekly inventory" : ""}
        {hasFred ? "  |  FRED: EPU, HY spread, yield curve, GPR" : ""}
        {hasChina ? "  |  China: CSI300/500, ChiNext, Shanghai via akshare" : ""}
      </p>
    </div>
  );
}
