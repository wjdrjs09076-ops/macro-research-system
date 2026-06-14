"use client";

import { useEffect, useState } from "react";
import {
  ComposedChart, LineChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ReferenceArea, ResponsiveContainer, Legend,
} from "recharts";

type TimingNull = {
  n_active: number; gross_return: number;
  null_pctile: number | null; null_median: number | null;
};
type StrategyData = {
  oos_cagr: number; oos_sharpe: number; oos_mdd: number | null;
  oos_active_days: number; alpha_capture: number;
  oos_raw_return?: number;
  timing_null?: TimingNull;
  cumulative: { dates: string[]; values: number[] };
};
type EvaluationMeta = {
  alpha_capture_def: string;
  bnh_raw_return: number;
  random_null: { sims: number; method: string };
  baselines: Record<string, string>;
};
type GateData = {
  generated_at: string;
  is_period: { start: string; end: string };
  oos_period: { start: string; end: string };
  gate_params: { window: number; consec: number; threshold: number; weights: Record<string, number> };
  is_gate_open_pct: number;
  oos_gate_open_pct: number;
  timeseries: {
    dates: string[]; gate_score: number[];
    beta_component: number[]; vrp_component: number[];
    dir_component: number[]; gate_open: boolean[];
  };
  strategies: Record<string, StrategyData>;
  evaluation_meta?: EvaluationMeta;
  gate_ml_concordance?: {
    crosstab: { both: number; gate_only: number; ml_only: number; neither: number };
    phi: number; observed_both: number; expected_both_if_independent: number;
    ml_active_total: number; gate_active_total: number; note: string;
  };
  threshold_sensitivity?: { threshold: number; oos_cagr: number; oos_sharpe: number; oos_active_days: number }[];
  param_freeze?: {
    params: { threshold: number; window: number; consec: number; weights: Record<string, number> };
    first_commit: string; oos_window: { start: string; end: string };
    verdict: string; note: string;
  };
  ml_shadow?: {
    in_decision_path: boolean; n_fire: number; reeval_n: number;
    reeval_ready: boolean; note: string; verdict: string;
  };
  action_comparison_hormuz?: Record<string, ActionMetric>;
  straddle_iv_hormuz?: Record<string, StraddleIV>;
  crisis_windows?: Record<string, {
    label: string; etf: string; direction: string;
    oos_start: string; oos_end: string;
    oos_gate_open_pct: number; bnh_raw_return: number;
    strategies: Record<string, StrategyData>;
    action_comparison?: Record<string, ActionMetric>;
    straddle_iv?: Record<string, StraddleIV>;
  }>;
};

type ActionMetric = { oos_cagr: number; oos_sharpe: number | null; oos_mdd: number; oos_active_days: number } | null;
type StraddleIV = { iv_mult: number; n_trades: number; wins: number; total_return: number; oos_sharpe: number | null; oos_mdd: number };

const STRATEGY_COLORS: Record<string, string> = {
  "Pure ML" : "#ef4444",
  "Gate Only": "#10b981",
  "Gate+ML"  : "#8b5cf6",
  "Gate+S2"  : "#f59e0b",
  "VolTgt XLE": "#38bdf8",
  "200DMA XLE": "#facc15",
  "BnH XLE"  : "#6b7280",
};

const STRATEGY_ORDER = [
  "Gate Only", "Gate+S2", "Pure ML", "Gate+ML",
  "VolTgt XLE", "200DMA XLE", "BnH XLE",
];

// 베이스라인(한 줄짜리 경쟁자) — 게이트와 같은 일(노출 조절)을 하는 대조군
const BASELINE_NAMES = new Set(["VolTgt XLE", "200DMA XLE"]);

function MetricCard({
  name, cagr, sharpe, mdd, days, capture, rawReturn, bnhRaw, timingNull, isBaseline, shadowNote, color,
}: {
  name: string; cagr: number; sharpe: number; mdd: number | null;
  days: number; capture: number; rawReturn?: number; bnhRaw?: number;
  timingNull?: TimingNull; isBaseline?: boolean; shadowNote?: string; color: string;
}) {
  const pct = timingNull?.null_pctile ?? null;
  // 백분위 색상: ≥0.95 유의(녹) / 0.5~0.95 약함(노랑) / <0.5 평균 이하(빨강)
  const pctColor = pct == null ? "text-gray-500"
    : pct >= 0.95 ? "text-emerald-400" : pct >= 0.5 ? "text-amber-400" : "text-red-400";
  return (
    <div className={`rounded-xl border p-4 ${isBaseline
      ? "border-sky-800/50 bg-sky-950/10" : "border-gray-800 bg-gray-900/50"}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-200">
          {name}{isBaseline && <span className="ml-1 text-[10px] text-sky-500">베이스라인</span>}
        </span>
        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <div className="text-gray-500">OOS CAGR</div>
          <div className={`font-mono mt-0.5 ${cagr >= 0 ? "text-emerald-400" : "text-red-400"}`}>
            {cagr >= 0 ? "+" : ""}{cagr.toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-gray-500">Sharpe</div>
          <div className="font-mono text-gray-200 mt-0.5">{sharpe.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-gray-500">MDD</div>
          <div className="font-mono text-gray-400 mt-0.5">
            {mdd != null ? `${mdd.toFixed(1)}%` : "—"}
          </div>
        </div>
        <div>
          <div className="text-gray-500">활성일</div>
          <div className="font-mono text-gray-400 mt-0.5">{days}일</div>
        </div>
        <div className="col-span-2">
          <div className="text-gray-500" title="OOS 누적수익(raw) ÷ B&H 누적수익(raw) × 100">
            Alpha Capture
            {rawReturn != null && bnhRaw != null && (
              <span className="ml-1 text-[10px] text-gray-600 font-mono">
                = {rawReturn.toFixed(1)}/{bnhRaw.toFixed(1)}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-1">
            <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.min(capture, 100)}%`, backgroundColor: color }}
              />
            </div>
            <span className="font-mono text-gray-300">{capture.toFixed(1)}%</span>
          </div>
        </div>
        {timingNull && (
          <div className="col-span-2 border-t border-gray-800 pt-1.5 mt-0.5">
            <div className="text-gray-500" title="활성일 수를 고정한 무작위 노출 분포에서의 백분위. 0.5≈타이밍 무정보, ≥0.95=유의">
              랜덤 타이밍 백분위
            </div>
            <div className={`font-mono mt-0.5 ${pctColor}`}>
              {pct == null ? "N/A (상시 노출)" : `p=${pct.toFixed(2)} ${pct >= 0.95 ? "(유의)" : pct >= 0.5 ? "(무정보)" : "(평균 이하)"}`}
            </div>
          </div>
        )}
        {shadowNote && (
          <div className="col-span-2 border-t border-violet-900/40 pt-1.5 mt-0.5 text-[10px] text-violet-300/80">
            {shadowNote}
          </div>
        )}
      </div>
    </div>
  );
}

export default function GatePage() {
  const [data, setData] = useState<GateData | null>(null);
  const [showComponents, setShowComponents] = useState(true);

  useEffect(() => {
    fetch("/data/gate_scores.json")
      .then((r) => r.json())
      .then(setData);
  }, []);

  if (!data) return <div className="text-gray-500 text-sm">Loading...</div>;

  const ts = data.timeseries;
  const gatePoints = ts.dates.map((d, i) => ({
    date: d,
    gate_score:    ts.gate_score[i],
    beta:  +(ts.beta_component[i] * 0.40).toFixed(4),
    vrp:   +(ts.vrp_component[i]  * 0.35).toFixed(4),
    dir:   +(ts.dir_component[i]  * 0.25).toFixed(4),
    open:  ts.gate_open[i],
  }));

  const oosStart = data.oos_period.start;

  // Build cumulative performance chart
  const allDates = new Set<string>();
  STRATEGY_ORDER.forEach((name) => {
    const s = data.strategies[name];
    if (s) s.cumulative.dates.forEach((d) => allDates.add(d));
  });
  const sortedDates = Array.from(allDates).sort();
  const perfPoints = sortedDates.map((date) => {
    const pt: Record<string, number | string> = { date };
    STRATEGY_ORDER.forEach((name) => {
      const s = data.strategies[name];
      if (!s) return;
      const idx = s.cumulative.dates.indexOf(date);
      if (idx >= 0) pt[name] = +s.cumulative.values[idx].toFixed(6);
    });
    return pt;
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Gate Score Timeline</h1>
        <p className="text-gray-400 text-sm mt-1">
          XLE / 호르무즈 봉쇄 — 게이트 점수 시계열 + 4-전략 OOS 성과 비교
        </p>
      </div>

      {/* Gate summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">IS 게이트 열림</div>
          <div className="text-2xl font-mono text-gray-300">{data.is_gate_open_pct.toFixed(1)}%</div>
          <div className="text-xs text-gray-600 mt-1">{data.is_period.start} ~ {data.is_period.end}</div>
        </div>
        <div className="rounded-xl border border-emerald-800/50 bg-emerald-950/20 p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">OOS 게이트 열림</div>
          <div className="text-2xl font-mono text-emerald-400">{data.oos_gate_open_pct.toFixed(1)}%</div>
          <div className="text-xs text-gray-600 mt-1">{data.oos_period.start} ~ {data.oos_period.end}</div>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">임계값</div>
          <div className="text-2xl font-mono text-gray-300">{data.gate_params.threshold}</div>
          <div className="text-xs text-gray-600 mt-1">롤링 {data.gate_params.window}일, 연속 {data.gate_params.consec}일</div>
        </div>
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4 text-center">
          <div className="text-xs text-gray-500 mb-1">가중치</div>
          <div className="text-xs font-mono text-gray-300 mt-2 space-y-1">
            <div>Beta 0.40</div>
            <div>VRP 0.35</div>
            <div>Dir 0.25</div>
          </div>
        </div>
      </div>

      {/* Gate Score Chart */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-200">게이트 점수 시계열 (XLE)</h2>
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showComponents}
              onChange={(e) => setShowComponents(e.target.checked)}
              className="accent-emerald-500"
            />
            컴포넌트 표시
          </label>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={gatePoints} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#6b7280" }}
              tickFormatter={(v) => v.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis domain={[-0.05, 1.05]} tick={{ fontSize: 10, fill: "#6b7280" }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 11 }}
              formatter={(v, n) => [typeof v === "number" ? v.toFixed(4) : v, n]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {gatePoints.filter((p) => p.open).map((p, i) => (
              <ReferenceArea key={i} x1={p.date} x2={p.date}
                fill="#10b981" fillOpacity={0.12} />
            ))}
            <ReferenceLine y={data.gate_params.threshold} stroke="#10b981"
              strokeDasharray="4 2" strokeOpacity={0.5}
              label={{ value: `${data.gate_params.threshold}`, fontSize: 10, fill: "#10b981" }} />
            <ReferenceLine x={oosStart} stroke="#ef4444" strokeDasharray="4 2"
              label={{ value: "OOS", fontSize: 10, fill: "#ef4444" }} />
            <Line type="monotone" dataKey="gate_score" stroke="#10b981"
              dot={false} strokeWidth={2} name="Gate Score" />
            {showComponents && <>
              <Line type="monotone" dataKey="beta" stroke="#8b5cf6"
                dot={false} strokeWidth={1} strokeDasharray="3 2" name="Beta×0.40" />
              <Line type="monotone" dataKey="vrp" stroke="#f59e0b"
                dot={false} strokeWidth={1} strokeDasharray="3 2" name="VRP×0.35" />
              <Line type="monotone" dataKey="dir" stroke="#06b6d4"
                dot={false} strokeWidth={1} strokeDasharray="3 2" name="Dir×0.25" />
            </>}
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-600 mt-2">
          초록 음영 = 게이트 OPEN (연속 {data.gate_params.consec}일 ≥ {data.gate_params.threshold})
        </p>
      </div>

      {/* Strategy Performance Cards */}
      <div>
        <h2 className="text-sm font-semibold text-gray-200 mb-4">OOS 전략 성과 비교 (2026.01~05)</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {STRATEGY_ORDER.map((name) => {
            const s = data.strategies[name];
            if (!s) return null;
            return (
              <MetricCard
                key={name} name={name}
                cagr={s.oos_cagr} sharpe={s.oos_sharpe}
                mdd={s.oos_mdd} days={s.oos_active_days}
                capture={s.alpha_capture}
                rawReturn={s.oos_raw_return}
                bnhRaw={data.evaluation_meta?.bnh_raw_return}
                timingNull={s.timing_null}
                isBaseline={BASELINE_NAMES.has(name)}
                shadowNote={(name === "Pure ML" || name === "Gate+ML") && data.ml_shadow
                  ? `ML shadow 격리 중 — 사이징 미반영, 평가 표본 누적 중 (n=${data.ml_shadow.n_fire}/${data.ml_shadow.reeval_n})`
                  : undefined}
                color={STRATEGY_COLORS[name]}
              />
            );
          })}
        </div>
      </div>

      {/* Cumulative Return Chart */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
        <h2 className="text-sm font-semibold text-gray-200 mb-4">OOS 누적 수익 비교</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={perfPoints} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#6b7280" }}
              tickFormatter={(v: string) => v.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10, fill: "#6b7280" }}
              tickFormatter={(v: number) => `${((v - 1) * 100).toFixed(0)}%`} />
            <Tooltip
              contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 11 }}
              formatter={(v, name) => [
                typeof v === "number" ? `${((v - 1) * 100).toFixed(1)}%` : String(v), name,
              ]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine y={1} stroke="#374151" strokeDasharray="2 2" />
            {STRATEGY_ORDER.map((name) => (
              data.strategies[name] ? (
                <Line
                  key={name} type="monotone" dataKey={name}
                  stroke={STRATEGY_COLORS[name]}
                  dot={false}
                  strokeWidth={name === "BnH XLE" ? 1 : 1.8}
                  strokeDasharray={name === "BnH XLE" ? "4 3" : undefined}
                  connectNulls
                />
              ) : null
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Insight Box — 데이터 기반 (감사 라운드 7 P1-2/P1-3/P1-4) */}
      {(() => {
        const g  = data.strategies["Gate Only"];
        const vt = data.strategies["VolTgt XLE"];
        const ma = data.strategies["200DMA XLE"];
        const gp = g?.timing_null?.null_pctile ?? null;
        const beatsBaselines = g && vt && ma && (g.oos_sharpe >= vt.oos_sharpe && g.oos_sharpe >= ma.oos_sharpe);
        return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">핵심 해석 — 평가 재설계 결과</h2>
        <p className="text-xs text-gray-500 leading-relaxed">
          이 OOS 창(XLE/호르무즈)은 수혜 섹터 + 상방 쇼크라 B&amp;H가 정의상 강한 대조군이다.
          따라서 게이트의 가치는 CAGR이 아니라 ① 한 줄짜리 베이스라인 대비 우위와
          ② 랜덤 타이밍 대비 정보량으로 판정한다. 불리한 결과도 그대로 게시한다.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-gray-400">
          <div className="bg-gray-900 rounded-lg p-3">
            <div className={`font-medium mb-1 ${beatsBaselines ? "text-emerald-400" : "text-red-400"}`}>
              베이스라인 대비 {beatsBaselines ? "우위" : "열위"} (P1-2)
            </div>
            {g && vt && ma ? (
              <>Gate Only Sharpe {g.oos_sharpe.toFixed(2)} vs VolTgt {vt.oos_sharpe.toFixed(2)} ·
              200DMA {ma.oos_sharpe.toFixed(2)}. {beatsBaselines
                ? "게이트가 한 줄짜리들을 이김."
                : "한 줄짜리 vol-target/200DMA 필터가 게이트를 이김 → 6레이어의 한계 기여 ≈ 0."}</>
            ) : "데이터 없음"}
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <div className={`font-medium mb-1 ${gp == null ? "text-gray-400" : gp >= 0.95 ? "text-emerald-400" : "text-amber-400"}`}>
              랜덤 타이밍 널 (P1-3)
            </div>
            {gp != null ? (
              <>Gate Only 백분위 p={gp.toFixed(2)} ({data.evaluation_meta?.random_null.sims.toLocaleString()}회 시뮬).
              {gp >= 0.95 ? " 무작위 노출 대비 유의." : " 무작위 노출과 통계적으로 구분 안 됨 = 타이밍 정보량 거의 0."}</>
            ) : "데이터 없음"}
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <div className="text-sky-400 font-medium mb-1">Alpha Capture 산식 (P1-4)</div>
            {data.evaluation_meta
              ? <>{data.evaluation_meta.alpha_capture_def}. B&amp;H raw = {data.evaluation_meta.bnh_raw_return.toFixed(1)}%.
                카드의 raw/B&amp;H 비율로 재현 가능 (기존 13.7%는 IS 손익 오염값이었음).</>
              : "산식 메타 없음"}
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <div className="text-purple-400 font-medium mb-1">Gate+ML 기아 원인 (P1-5)</div>
            {data.gate_ml_concordance
              ? <>OOS ML 발화 {data.gate_ml_concordance.ml_active_total}일뿐 → φ={data.gate_ml_concordance.phi.toFixed(2)}
                (약한 양). 역상관이 아니라 ML이 이 레짐에서 거의 안 켜지는 게 원인 (감사 가설 정정).</>
              : "게이트·ML AND 결합이 거의 발화 안 함."}
          </div>
        </div>
      </div>
        );
      })()}

      {/* P1-1 하락 위기 홈그라운드 시험 */}
      {data.crisis_windows && Object.keys(data.crisis_windows).length > 0 && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-gray-200">하락 위기 홈그라운드 시험 (P1-1)</h2>
            <p className="text-xs text-gray-500 mt-1 leading-relaxed">
              호르무즈/수에즈는 상방 쇼크라 게이트의 본 주장(하락 손실 회피)을 시험하지 못한다.
              GFC·COVID·긴축 세 하락 창에 시점-aware로 동일 전략을 적용 — 핵심 지표는 <b className="text-gray-300">MDD</b>.
              한 줄짜리 200DMA 대비 우위 여부를 그대로 게시한다.
            </p>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {Object.values(data.crisis_windows).map((w) => {
              const order = Object.keys(w.strategies).sort((x, y) => {
                const rank = (k: string) => k.startsWith("Gate Only") ? 0 : k.startsWith("Gate+S2") ? 1
                  : k.startsWith("Pure ML") ? 2 : k.startsWith("Gate+ML") ? 3
                  : k.startsWith("VolTgt") ? 4 : k.startsWith("200DMA") ? 5 : 6;
                return rank(x) - rank(y);
              });
              const gate = w.strategies["Gate Only"];
              const dma = order.map((k) => k.startsWith("200DMA") ? w.strategies[k] : null).find(Boolean);
              const gateBeatsDma = gate && dma && Math.abs(gate.oos_mdd ?? 0) <= Math.abs(dma.oos_mdd ?? 0);
              return (
                <div key={w.label} className="rounded-lg border border-gray-800 bg-gray-900/50 p-3">
                  <div className="flex items-baseline justify-between mb-2">
                    <span className="text-sm font-medium text-gray-200">{w.label}</span>
                    <span className="text-[10px] text-gray-500 font-mono">{w.etf} · 열림 {w.oos_gate_open_pct}%</span>
                  </div>
                  <table className="w-full text-[11px] font-mono">
                    <thead><tr className="text-gray-600">
                      <th className="text-left font-normal">전략</th>
                      <th className="text-right font-normal">CAGR</th>
                      <th className="text-right font-normal">Shrp</th>
                      <th className="text-right font-normal">MDD</th>
                    </tr></thead>
                    <tbody>
                      {order.map((k) => {
                        const s = w.strategies[k];
                        const isGate = k === "Gate Only";
                        const isBnh = k.startsWith("BnH");
                        return (
                          <tr key={k} className={isGate ? "text-emerald-300" : isBnh ? "text-gray-500" : "text-gray-300"}>
                            <td className="text-left truncate max-w-[88px]">{isGate ? "▶ " : ""}{k}</td>
                            <td className="text-right">{s.oos_cagr >= 0 ? "+" : ""}{s.oos_cagr.toFixed(0)}%</td>
                            <td className="text-right">{s.oos_sharpe == null ? "—" : s.oos_sharpe.toFixed(2)}</td>
                            <td className={`text-right ${isBnh ? "" : "text-amber-400/90"}`}>{s.oos_mdd != null ? `${s.oos_mdd.toFixed(0)}%` : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <div className={`text-[11px] mt-2 ${gateBeatsDma ? "text-emerald-400" : "text-red-400"}`}>
                    {gate && dma
                      ? `게이트 MDD ${gate.oos_mdd?.toFixed(0)}% vs 200DMA ${dma.oos_mdd?.toFixed(0)}% → ${gateBeatsDma ? "게이트 우위" : "200DMA 우위"}`
                      : ""}
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-gray-500 leading-relaxed">
            결론: 게이트는 세 위기 모두 B&amp;H 대비 MDD를 크게 줄여 <b className="text-gray-300">브레이크로서 작동</b>한다
            (예: GFC -19% vs B&amp;H -74%). 그러나 한 줄짜리 200DMA와 비교하면 우열이 갈려
            (GFC·COVID 200DMA 근소 우위, 2022 게이트 우위) <b className="text-gray-300">명확한 우위는 미확정</b>.
            Gate+ML은 GFC에서 ML 미발화로 0일(P1-5와 동일 원인).
          </p>
        </div>
      )}

      {/* W2-A 감지→액션 분리: 현물 액션 3종 비교 (IV 가정 0) */}
      {data.action_comparison_hormuz && (() => {
        const windows: { label: string; ac: Record<string, ActionMetric> }[] = [
          { label: "호르무즈", ac: data.action_comparison_hormuz! },
          ...Object.values(data.crisis_windows ?? {})
            .filter((w) => w.action_comparison)
            .map((w) => ({ label: w.label, ac: w.action_comparison! })),
        ];
        const norm = (ac: Record<string, ActionMetric>, base: string) => {
          const k = Object.keys(ac).find((x) => x.startsWith(base));
          return k ? ac[k] : null;
        };
        const rows: { key: string; label: string; hl?: boolean }[] = [
          { key: "on_off", label: "① 현물 on/off (현행)" },
          { key: "exposure_var", label: "② 노출가변 1.5x" },
          { key: "spread_hedged", label: "③ β헤지 스프레드", hl: true },
          { key: "VolTgt", label: "VolTgt (베이스라인)" },
          { key: "200DMA", label: "200DMA (베이스라인)" },
        ];
        const cell = (ac: Record<string, ActionMetric>, key: string) => {
          const m = key === "VolTgt" || key === "200DMA" ? norm(ac, key) : ac[key];
          if (!m) return <span className="text-gray-600">—</span>;
          return <span>{m.oos_cagr >= 0 ? "+" : ""}{m.oos_cagr.toFixed(0)}% <span className="text-amber-400/80">{m.oos_mdd.toFixed(0)}%</span></span>;
        };
        return (
          <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-3">
            <div>
              <h2 className="text-sm font-semibold text-gray-200">감지 → 액션 분리 (W2-A · IV 가정 0)</h2>
              <p className="text-xs text-gray-500 mt-1 leading-relaxed">
                게이트 OPEN 신호의 수익화 방식 비교 — 셀은 <span className="text-gray-300">CAGR</span> /
                <span className="text-amber-400/80"> MDD</span>. 현물·가격만 사용(옵션 IV 가정 없음).
                핵심: 어떤 현물 액션이 한 줄짜리 베이스라인을 이기는가.
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px] font-mono">
                <thead><tr className="text-gray-600">
                  <th className="text-left font-normal pb-1">액션 \ 창</th>
                  {windows.map((w) => <th key={w.label} className="text-right font-normal pb-1 px-1">{w.label}</th>)}
                </tr></thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.key} className={r.hl ? "text-emerald-300" : r.key.match(/VolTgt|200DMA/) ? "text-sky-300/80" : "text-gray-300"}>
                      <td className="text-left py-0.5">{r.label}</td>
                      {windows.map((w) => <td key={w.label} className="text-right px-1 whitespace-nowrap">{cell(w.ac, r.key)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">
              결론: <b className="text-emerald-300">③ β헤지 스프레드가 현행 on/off를 4창 전부 MDD에서 이기고,
              200DMA 베이스라인을 3/4창에서 이긴다</b> (호르무즈 -5% · GFC -7% · 긴축 -2%).
              <b className="text-red-400"> ② 노출가변(OPEN 시 1.5x)은 재앙</b>(GFC MDD -76%) — 게이트의 가치는
              레버리지가 아니라 빠지는 것. 즉 게이트의 위기 수익화는 <b className="text-gray-300">미해결이 아니라
              스프레드가 더 나은 매핑</b>이다. 단 이 OOS는 P1-6 designer leakage 라벨 위의 숫자임.
            </p>

            {data.straddle_iv_hormuz && (() => {
              const sw: { label: string; iv: Record<string, StraddleIV> }[] = [
                { label: "호르무즈", iv: data.straddle_iv_hormuz! },
                ...Object.values(data.crisis_windows ?? {})
                  .filter((w) => w.straddle_iv)
                  .map((w) => ({ label: w.label, iv: w.straddle_iv! })),
              ];
              const mults = ["1x", "1.5x", "2x"];
              return (
                <div className="border-t border-gray-800 pt-3 mt-1">
                  <h3 className="text-xs font-semibold text-gray-300 mb-1">W2-B · ATM 스트래들 IV 배수 민감도 (총수익%)</h3>
                  <p className="text-[11px] text-gray-500 mb-2 leading-relaxed">
                    GARCH-IV × 배수로 BS 재평가. 위기 IV는 통상 GARCH의 1.5~2배 →
                    <b className="text-gray-400"> 1.0배에서만 이기는 결과는 기각</b>(단일 IV 결과 금지).
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-[11px] font-mono">
                      <thead><tr className="text-gray-600">
                        <th className="text-left font-normal pb-1">창 \ IV배수</th>
                        {mults.map((m) => <th key={m} className="text-right font-normal px-2 pb-1">{m}</th>)}
                      </tr></thead>
                      <tbody>
                        {sw.map((w) => (
                          <tr key={w.label} className="text-gray-300">
                            <td className="text-left py-0.5">{w.label}</td>
                            {mults.map((m) => {
                              const v = w.iv[m];
                              const val = v ? v.total_return : null;
                              return <td key={m} className={`text-right px-2 ${val == null ? "text-gray-600" : val > 0 ? "text-emerald-400" : "text-red-400"}`}>
                                {val == null ? "—" : `${val >= 0 ? "+" : ""}${val.toFixed(0)}%`}
                              </td>;
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[11px] text-red-400/90 mt-2 leading-relaxed">
                    결론: 스트래들은 현실적 IV(1.5~2배)에서 견디지 못한다 — 유일한 상방창(호르무즈)도
                    1.0배 +95% → 2.0배 -14%로 붕괴, 하락 3창은 전 배수 손실. 라이브 시스템의 실제
                    instrument(스트래들)보다 W2-A의 β헤지 스프레드가 우월.
                  </p>
                </div>
              );
            })()}
          </div>
        );
      })()}

      {/* P1-5 게이트↔ML 교차표 + P1-6 동결/민감도 진단 */}
      {(data.gate_ml_concordance || data.param_freeze) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {data.gate_ml_concordance && (
            <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
              <h2 className="text-sm font-semibold text-gray-200 mb-1">게이트 ↔ ML 일치 (P1-5)</h2>
              <p className="text-xs text-gray-500 mb-3">OOS 2×2 · φ 계수</p>
              {(() => {
                const ct = data.gate_ml_concordance!;
                return (
                  <>
                    <div className="grid grid-cols-3 text-xs font-mono text-center mb-3">
                      <div></div><div className="text-gray-500">ML 활성</div><div className="text-gray-500">ML 비활성</div>
                      <div className="text-gray-500 flex items-center justify-end pr-2">게이트 OPEN</div>
                      <div className="bg-emerald-950/40 text-emerald-300 py-2 rounded-l">{ct.crosstab.both}</div>
                      <div className="bg-gray-800/50 text-gray-300 py-2 rounded-r">{ct.crosstab.gate_only}</div>
                      <div className="text-gray-500 flex items-center justify-end pr-2">게이트 CLOSED</div>
                      <div className="bg-gray-800/50 text-gray-300 py-2 rounded-l">{ct.crosstab.ml_only}</div>
                      <div className="bg-gray-900 text-gray-500 py-2 rounded-r">{ct.crosstab.neither}</div>
                    </div>
                    <div className="text-xs text-gray-400 space-y-1">
                      <div>φ = <span className="font-mono text-gray-200">{ct.phi.toFixed(3)}</span> (약한 양의 상관)</div>
                      <div>both 관측 <span className="font-mono text-gray-200">{ct.observed_both}</span> vs 독립기대 <span className="font-mono text-gray-200">{ct.expected_both_if_independent}</span></div>
                      <div className="text-amber-400/80 pt-1">{ct.note}</div>
                      <div className="text-gray-500 pt-1">권고: AND 결합 자체가 아니라 ML 임계(0.55)/캘리브레이션이 병목. 레짐 스위치 결합 또는 ML 임계 하향을 다음 분기에서 검토.</div>
                    </div>
                  </>
                );
              })()}
            </div>
          )}
          {data.param_freeze && (
            <div className="rounded-xl border border-red-900/40 bg-red-950/10 p-5">
              <h2 className="text-sm font-semibold text-gray-200 mb-1">파라미터 동결 증명 (P1-6)</h2>
              <p className="text-xs text-red-400/80 mb-3">⚠ designer leakage 가능</p>
              <div className="text-xs text-gray-400 space-y-1.5">
                <div>최초 git 커밋 <span className="font-mono text-gray-200">{data.param_freeze.first_commit}</span> &gt; OOS 종료 <span className="font-mono text-gray-200">{data.param_freeze.oos_window.end}</span></div>
                <div className="text-gray-500">{data.param_freeze.note}</div>
                {data.threshold_sensitivity && (
                  <div className="pt-2">
                    <div className="text-gray-500 mb-1">임계값 민감도 (Gate Only OOS):</div>
                    <table className="w-full font-mono text-[11px]">
                      <thead><tr className="text-gray-600">
                        <th className="text-left">임계</th><th className="text-right">CAGR</th><th className="text-right">Sharpe</th><th className="text-right">활성일</th>
                      </tr></thead>
                      <tbody>
                        {data.threshold_sensitivity.map((r) => (
                          <tr key={r.threshold} className={r.threshold === data.param_freeze!.params.threshold ? "text-emerald-300" : "text-gray-300"}>
                            <td className="text-left">{r.threshold.toFixed(2)}{r.threshold === data.param_freeze!.params.threshold ? " ◀" : ""}</td>
                            <td className="text-right">{r.oos_cagr >= 0 ? "+" : ""}{r.oos_cagr.toFixed(1)}%</td>
                            <td className="text-right">{r.oos_sharpe.toFixed(2)}</td>
                            <td className="text-right">{r.oos_active_days}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
