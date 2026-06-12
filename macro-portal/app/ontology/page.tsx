"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ReferenceArea, ResponsiveContainer, Legend,
} from "recharts";

const CAUSAL_SIGNALS = [
  {
    ticker: "XLK", name: "Technology", delta_macro: "VIX", delta: 0.0447,
    chain: [
      { src: "OIL",       lag: "1m", stability: "stru", dir: "+" },
      { src: "CPI",       lag: "3m", stability: "emer", dir: "+" },
      { src: "US2Y",      lag: "1m", stability: "stru", dir: "+" },
      { src: "FED_FUNDS", lag: "3m", stability: "stru", dir: "-" },
      { src: "VIX",       lag: null, stability: null,   dir: null },
    ],
    total_lag: "8m", confidence: 0.671,
  },
  {
    ticker: "XLE", name: "Energy", delta_macro: "VIX", delta: 0.0391,
    chain: [
      { src: "OIL",       lag: "1m", stability: "stru", dir: "+" },
      { src: "CPI",       lag: "3m", stability: "emer", dir: "+" },
      { src: "US2Y",      lag: "1m", stability: "stru", dir: "+" },
      { src: "FED_FUNDS", lag: "3m", stability: "stru", dir: "-" },
      { src: "VIX",       lag: null, stability: null,   dir: null },
    ],
    total_lag: "8m", confidence: 0.643,
  },
  {
    ticker: "XLF", name: "Financials", delta_macro: "VIX", delta: 0.0362,
    chain: [
      { src: "CPI",       lag: "3m", stability: "emer", dir: "+" },
      { src: "US2Y",      lag: "1m", stability: "stru", dir: "+" },
      { src: "FED_FUNDS", lag: "3m", stability: "stru", dir: "-" },
      { src: "VIX",       lag: null, stability: null,   dir: null },
    ],
    total_lag: "7m", confidence: 0.618,
  },
  {
    ticker: "XLY", name: "Consumer Discr.", delta_macro: "VIX", delta: 0.0318,
    chain: [
      { src: "CPI",       lag: "3m", stability: "emer", dir: "+" },
      { src: "US2Y",      lag: "1m", stability: "stru", dir: "+" },
      { src: "FED_FUNDS", lag: "3m", stability: "stru", dir: "-" },
      { src: "VIX",       lag: null, stability: null,   dir: null },
    ],
    total_lag: "7m", confidence: 0.594,
  },
  {
    ticker: "XLC", name: "Comm. Services", delta_macro: "VIX", delta: 0.0297,
    chain: [
      { src: "CPI",       lag: "3m", stability: "emer", dir: "+" },
      { src: "US2Y",      lag: "1m", stability: "stru", dir: "+" },
      { src: "FED_FUNDS", lag: "3m", stability: "stru", dir: "-" },
      { src: "VIX",       lag: null, stability: null,   dir: null },
    ],
    total_lag: "7m", confidence: 0.572,
  },
  {
    ticker: "XLI", name: "Industrials", delta_macro: "VIX", delta: 0.0261,
    chain: [
      { src: "CREDIT_SPREAD", lag: "1m", stability: "emer", dir: "+" },
      { src: "VIX",           lag: null, stability: null,   dir: null },
    ],
    total_lag: "1m", confidence: 0.541,
  },
  {
    ticker: "XLB", name: "Materials", delta_macro: "VIX", delta: 0.0244,
    chain: [
      { src: "CREDIT_SPREAD", lag: "1m", stability: "emer", dir: "+" },
      { src: "VIX",           lag: null, stability: null,   dir: null },
    ],
    total_lag: "1m", confidence: 0.523,
  },
  {
    ticker: "XLRE", name: "Real Estate", delta_macro: "VIX", delta: 0.0231,
    chain: [
      { src: "FED_FUNDS", lag: "3m", stability: "stru", dir: "-" },
      { src: "VIX",       lag: null, stability: null,   dir: null },
    ],
    total_lag: "3m", confidence: 0.508,
  },
  {
    ticker: "XLU", name: "Utilities", delta_macro: "VIX", delta: 0.0198,
    chain: [
      { src: "FED_FUNDS", lag: "3m", stability: "stru", dir: "-" },
      { src: "VIX",       lag: null, stability: null,   dir: null },
    ],
    total_lag: "3m", confidence: 0.487,
  },
  {
    ticker: "XLV", name: "Health Care", delta_macro: "VIX", delta: 0.0143,
    chain: [
      { src: "CREDIT_SPREAD", lag: "1m", stability: "emer", dir: "+" },
      { src: "VIX",           lag: null, stability: null,   dir: null },
    ],
    total_lag: "1m", confidence: 0.456,
  },
];

type Step1 = {
  beta_mean: number; beta_median: number;
  beta_below_neg005_pct: number; beta_above_150_pct: number;
  vrp_mean: number; vrp_median: number; vrp_below_0_pct: number;
};
type Step2 = {
  beta_comp_pct: number; beta_contrib: number;
  vrp_comp_pct: number;  vrp_contrib: number;
  dir_comp_pct: number;  dir_contrib: number;
  total_score: number;
  weights: { beta: number; vrp: number; direction: number };
};
type Step3 = { percentiles: Record<string, number>; above_threshold_pct: number };
type Step4 = {
  n_runs: number; max_run: number; avg_run: number; runs_5plus: number;
  gate_open_days: number; gate_total_days: number; gate_open_pct: number;
};
type Timeseries = {
  dates: string[]; gate_score: number[];
  beta_comp: number[]; vrp_comp: number[]; dir_comp: number[];
  gate_open: boolean[];
};
type Crisis = {
  label: string; event_type: string; mechanism: string;
  direction: string; etf: string; oos_start: string;
  expected_gate: string; actual_gate: string; correct: boolean;
  step1: Step1; step2: Step2; step3: Step3; step4: Step4;
  timeseries: Timeseries;
};
type TraceData = {
  generated_at: string;
  gate_params: { window: number; consec: number; threshold: number };
  crises: Record<string, Crisis>;
};

const CRISIS_ORDER = ["hormuz", "gfc", "covid", "dotcom"];
const CRISIS_COLORS: Record<string, string> = {
  hormuz: "#10b981", gfc: "#ef4444", covid: "#3b82f6", dotcom: "#f59e0b",
};

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-900 rounded-lg p-3">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-sm font-mono text-gray-100">{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  );
}

function BarRow({ label, pct, contrib, color, max = 100 }: {
  label: string; pct: number; contrib: number; color: string; max?: number;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="font-mono text-gray-300">{pct.toFixed(1)}% → contrib {contrib.toFixed(3)}</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min((pct / max) * 100, 100)}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

type LiveCandidate = {
  ticker: string;
  confidence: number;
  rule: string;
  signal_type: string;
  regime: string;
  all_rules: string[];
  reasoning: string[];
  direction?: number;
};
type VarianceRow = {
  ticker: string;
  self_share: number;
  propagation_score: number;
  cholesky_order: number;
  top_sources: { src: string; share: number }[];
};
type JointSim = {
  horizon: number;
  n_sim: number;
  pair_threshold: number;
  per_threshold: { threshold: number; p_any: number; p_all: number; p_half: number; mean_hit_count: number }[];
  top_pair_co_drawdown: { a: string; b: string; p_co: number }[];
  sigmas_annual?: Record<string, number>;
};
type RegimeState = {
  confirmed: string;
  pending: string | null;
  pending_count: number;
  confirm_cycles: number;
  vix: number | null;
  ts: string | null;
};
type AuditRow = {
  sector: string;
  macro: string;
  delta_raw: number | null;
  t_raw: number | null;
  delta_ctrl: number | null;
  t_ctrl: number | null;
  verdict: "confirmed" | "killed" | "emerged" | "flipped";
};
type LiveSignals = {
  active_regime: string;
  regime_state?: RegimeState | null;
  generated: string;
  straddle_candidates: LiveCandidate[];
  directional_candidates: LiveCandidate[];
  short_straddle_candidates?: LiveCandidate[];
  sensitivity_audit?: AuditRow[];
  variance_decomposition: {
    by_sector: VarianceRow[];
    shock_propagators: { ticker: string; propagation_score: number; confidence: number; reasoning: string[] }[];
    variance_concentrated: { ticker: string; self_share: number; confidence: number; reasoning: string[] }[];
  };
  joint_simulation: JointSim | null;
};

type IVRecord = {
  ts: string;
  ticker: string;
  iv_atm: number;
  iv_call: number;
  iv_put: number;
  rv_20d: number;
  vrp_iv: number;
  dte: number;
};
type IVHistory = {
  generated: string;
  n_records: number;
  records: IVRecord[];
};

export default function OntologyPage() {
  const [data, setData] = useState<TraceData | null>(null);
  const [active, setActive] = useState<string>("hormuz");
  const [live, setLive] = useState<LiveSignals | null>(null);
  const [ivHist, setIvHist] = useState<IVHistory | null>(null);

  useEffect(() => {
    fetch("/data/ontology_trace.json")
      .then((r) => r.json())
      .then(setData);
    fetch("/data/ontology_signals.json")
      .then((r) => r.ok ? r.json() : null)
      .then(setLive)
      .catch(() => setLive(null));
    fetch("/data/iv_history.json")
      .then((r) => r.ok ? r.json() : null)
      .then(setIvHist)
      .catch(() => setIvHist(null));
  }, []);

  if (!data) {
    return <div className="text-gray-500 text-sm">Loading...</div>;
  }

  const crisis = data.crises[active];
  if (!crisis) return null;

  const color = CRISIS_COLORS[active];
  const tsPoints = crisis.timeseries.dates.map((d, i) => ({
    date: d,
    gate_score: crisis.timeseries.gate_score[i],
    beta:  +(crisis.timeseries.beta_comp[i] * 0.40).toFixed(4),
    vrp:   +(crisis.timeseries.vrp_comp[i]  * 0.35).toFixed(4),
    dir:   +(crisis.timeseries.dir_comp[i]  * 0.25).toFixed(4),
    open:  crisis.timeseries.gate_open[i],
  }));

  const oosStart = crisis.oos_start;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Ontology Decision Trace</h1>
        <p className="text-gray-400 text-sm mt-1">
          각 위기에서 게이트가 어떤 수치 근거로 OPEN/CLOSED를 판단했는지 Step별 추적
        </p>
      </div>

      {/* Crisis Tabs */}
      <div className="flex gap-2 flex-wrap">
        {CRISIS_ORDER.filter((k) => data.crises[k]).map((key) => {
          const c = data.crises[key];
          return (
            <button
              key={key}
              onClick={() => setActive(key)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                active === key
                  ? "text-gray-900"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
              style={active === key ? { backgroundColor: CRISIS_COLORS[key] } : {}}
            >
              {c.event_type}
              <span className={`ml-2 text-xs ${c.correct ? "opacity-80" : "opacity-60"}`}>
                {c.correct ? "✓" : "✗"}
              </span>
            </button>
          );
        })}
      </div>

      {/* Crisis Header */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span
                className="text-xs font-mono px-2 py-0.5 rounded"
                style={{ backgroundColor: color + "33", color }}
              >
                {crisis.event_type}
              </span>
              <span className="text-gray-400 text-xs">{crisis.direction} | {crisis.etf}</span>
            </div>
            <h2 className="text-lg font-semibold text-gray-100">{crisis.label}</h2>
            <p className="text-sm text-gray-400 mt-1 max-w-2xl">{crisis.mechanism}</p>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-500 mb-1">Gate 판정</div>
            <span className={`text-sm font-mono px-3 py-1 rounded-full ${
              crisis.correct
                ? "bg-emerald-900/50 text-emerald-400"
                : "bg-red-900/50 text-red-400"
            }`}>
              {crisis.actual_gate} {crisis.correct ? "✓ OK" : "✗ FAIL"}
            </span>
            <div className="text-xs text-gray-600 mt-1">예측: {crisis.expected_gate}</div>
          </div>
        </div>
      </div>

      {/* Steps Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Step 1 */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
          <h3 className="text-xs font-mono text-gray-500 mb-3">STEP 1 — Raw Indicators (OOS)</h3>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Beta(30d) 평균" value={crisis.step1.beta_mean.toFixed(3)} />
            <Stat label="Beta 중앙값" value={crisis.step1.beta_median.toFixed(3)} />
            <Stat
              label="Beta < -0.05 비율"
              value={`${crisis.step1.beta_below_neg005_pct.toFixed(1)}%`}
              sub="LONG 게이트 핵심 조건"
            />
            <Stat
              label="Beta > 1.50 비율"
              value={`${crisis.step1.beta_above_150_pct.toFixed(1)}%`}
              sub="SHORT 게이트 핵심 조건"
            />
            <Stat label="VRP 평균" value={crisis.step1.vrp_mean.toFixed(4)} />
            <Stat
              label="VRP < 0 비율"
              value={`${crisis.step1.vrp_below_0_pct.toFixed(1)}%`}
              sub="VRP 음전환 체제 확인"
            />
          </div>
        </div>

        {/* Step 2 */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
          <h3 className="text-xs font-mono text-gray-500 mb-3">
            STEP 2 — Gate Components (Rolling {data.gate_params.window}d avg)
          </h3>
          <div className="space-y-4">
            <BarRow
              label={crisis.direction === "LONG" ? "Beta 역전 (×0.40)" : "Beta 급등 (×0.40)"}
              pct={crisis.step2.beta_comp_pct}
              contrib={crisis.step2.beta_contrib}
              color="#8b5cf6"
            />
            <BarRow
              label="VRP 음전환 (×0.35)"
              pct={crisis.step2.vrp_comp_pct}
              contrib={crisis.step2.vrp_contrib}
              color="#f59e0b"
            />
            <BarRow
              label={crisis.direction === "LONG" ? "ETF 초과수익 (×0.25)" : "ETF 하회 (×0.25)"}
              pct={crisis.step2.dir_comp_pct}
              contrib={crisis.step2.dir_contrib}
              color="#06b6d4"
            />
            <div className="pt-2 border-t border-gray-800 flex justify-between items-center">
              <span className="text-xs text-gray-500">종합 게이트 점수 (OOS 평균)</span>
              <span className={`font-mono text-sm ${
                crisis.step2.total_score >= data.gate_params.threshold
                  ? "text-emerald-400" : "text-gray-400"
              }`}>
                {crisis.step2.total_score.toFixed(3)}
                <span className="text-gray-600 text-xs ml-2">
                  (임계값 {data.gate_params.threshold})
                </span>
              </span>
            </div>
          </div>
        </div>

        {/* Step 3 */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
          <h3 className="text-xs font-mono text-gray-500 mb-3">STEP 3 — Score Distribution (OOS)</h3>
          <div className="space-y-2 mb-4">
            <div className="flex gap-2 flex-wrap">
              {Object.entries(crisis.step3.percentiles).map(([k, v]) => (
                <div key={k} className="bg-gray-900 rounded px-3 py-1.5 text-center">
                  <div className="text-xs text-gray-500">{k.toUpperCase()}</div>
                  <div className={`font-mono text-sm ${
                    v >= data.gate_params.threshold ? "text-emerald-400" : "text-gray-300"
                  }`}>{v.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 flex justify-between items-center">
            <span className="text-xs text-gray-500">≥ {data.gate_params.threshold} 비율</span>
            <span className="font-mono text-sm text-gray-200">
              {crisis.step3.above_threshold_pct.toFixed(1)}%
            </span>
          </div>
          <p className="text-xs text-gray-600 mt-2">
            5일 연속 충족 시 게이트 개방 (GATE_CONSEC = {data.gate_params.consec})
          </p>
        </div>

        {/* Step 4 */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
          <h3 className="text-xs font-mono text-gray-500 mb-3">
            STEP 4 — Consecutive Pattern ({data.gate_params.consec}d threshold)
          </h3>
          <div className="grid grid-cols-2 gap-2 mb-4">
            <Stat label="연속 구간 수" value={`${crisis.step4.n_runs}회`} />
            <Stat label="최장 연속" value={`${crisis.step4.max_run}일`} />
            <Stat label="평균 연속" value={`${crisis.step4.avg_run.toFixed(1)}일`} />
            <Stat label={`${data.gate_params.consec}일↑ 구간`} value={`${crisis.step4.runs_5plus}회`} />
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">게이트 열림 (OOS)</span>
              <span className={`font-mono text-sm ${
                crisis.step4.gate_open_pct > 20 ? "text-emerald-400" : "text-gray-400"
              }`}>
                {crisis.step4.gate_open_days}/{crisis.step4.gate_total_days}일
                ({crisis.step4.gate_open_pct.toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Gate Score Timeseries Chart */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
        <h3 className="text-xs font-mono text-gray-500 mb-4">
          GATE SCORE TIMESERIES — {crisis.label}
        </h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={tsPoints} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "#6b7280" }}
              tickFormatter={(v) => v.slice(0, 7)}
              interval="preserveStartEnd"
            />
            <YAxis domain={[-0.05, 1.05]} tick={{ fontSize: 10, fill: "#6b7280" }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 11 }}
              formatter={(v, name) => [typeof v === "number" ? v.toFixed(4) : v, name]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {tsPoints.filter((p) => p.open).map((p, i) => (
              <ReferenceArea key={i} x1={p.date} x2={p.date}
                fill={color} fillOpacity={0.15} />
            ))}
            <ReferenceLine y={data.gate_params.threshold} stroke={color}
              strokeDasharray="4 2" strokeOpacity={0.6}
              label={{ value: `threshold ${data.gate_params.threshold}`, fontSize: 10, fill: color }} />
            <ReferenceLine x={oosStart} stroke="#ef4444"
              strokeDasharray="4 2"
              label={{ value: "OOS", fontSize: 10, fill: "#ef4444" }} />
            <Line type="monotone" dataKey="gate_score" stroke={color}
              dot={false} strokeWidth={1.8} name="Gate Score" />
            <Line type="monotone" dataKey="beta" stroke="#8b5cf6"
              dot={false} strokeWidth={1} strokeDasharray="3 2" name="Beta (×0.40)" />
            <Line type="monotone" dataKey="vrp" stroke="#f59e0b"
              dot={false} strokeWidth={1} strokeDasharray="3 2" name="VRP (×0.35)" />
            <Line type="monotone" dataKey="dir" stroke="#06b6d4"
              dot={false} strokeWidth={1} strokeDasharray="3 2" name="Dir (×0.25)" />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-600 mt-2">
          음영 = 게이트 OPEN 구간 | 빨간 점선 = OOS 시작({oosStart})
        </p>
      </div>

      {/* Causal Chain Monitor Section */}
      <div className="rounded-xl border border-purple-700/50 bg-purple-950/20 p-6 space-y-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">PCMCI+</span>
              <h3 className="text-sm font-semibold text-gray-100">Causal Chain Monitor</h3>
            </div>
            <p className="text-xs text-gray-400">
              TRANSMITS_TO 인과 체인을 통해 VIX에 간접 노출된 섹터. 변화점 2022-02 기준 structural/emerging 링크만 포함.
            </p>
          </div>
          <div className="text-right text-xs bg-gray-900/60 rounded-lg p-3 min-w-[140px]">
            <div className="text-gray-500 mb-0.5">활성 신호</div>
            <div className="font-mono text-purple-300 text-base">{CAUSAL_SIGNALS.length}개 섹터</div>
            <div className="text-gray-600 mt-0.5">MONITOR 등급</div>
          </div>
        </div>

        <div className="space-y-3">
          {CAUSAL_SIGNALS.map((sig) => (
            <div key={sig.ticker} className="bg-gray-900/40 rounded-lg p-3">
              <div className="flex items-center justify-between gap-4 mb-2">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm text-purple-300">{sig.ticker}</span>
                  <span className="text-xs text-gray-400">{sig.name}</span>
                  <span className="text-xs text-gray-600">
                    delta({sig.delta_macro})={sig.delta.toFixed(4)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">총 선행 {sig.total_lag}</span>
                  <div className="flex items-center gap-1">
                    <div className="w-20 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-purple-500"
                        style={{ width: `${sig.confidence * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs text-gray-400">{sig.confidence.toFixed(3)}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1 flex-wrap text-xs font-mono">
                {sig.chain.map((hop, i) => (
                  <span key={i} className="flex items-center gap-1">
                    <span className={`px-1.5 py-0.5 rounded ${
                      hop.src === "VIX" || hop.src === "US2Y" || hop.src === "US10Y" || hop.src === "DXY"
                        ? "bg-blue-900/50 text-blue-300"
                        : "bg-purple-900/50 text-purple-300"
                    }`}>
                      {hop.src}
                    </span>
                    {hop.lag && (
                      <span className={`text-gray-600`}>
                        --{hop.dir}{hop.lag}·{hop.stability}--&gt;
                      </span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <p className="text-xs text-gray-600 pt-1">
          stru = structural (2007-~2025 전체 + 2022-02 이후 모두 유의) | emer = emerging (2022-02 이후만 유의) |
          confidence = 0.50 + min_score × |delta| × 15 (상한 0.88)
        </p>
      </div>

      {/* Live Dual-Strategy Signals + Cholesky Variance Decomposition */}
      {live && (
        <LiveSignalsBlock live={live} />
      )}

      {/* IV history time series (Alpaca options snapshot, 매시간 누적) */}
      {ivHist && ivHist.n_records > 0 && (
        <IVHistoryBlock hist={ivHist} />
      )}
    </div>
  );
}

function CandidateRow({
  c, color, isDirectional,
}: { c: LiveCandidate; color: string; isDirectional?: boolean }) {
  const arrow = isDirectional
    ? (c.direction && c.direction > 0 ? "LONG " : "SHORT")
    : "";
  return (
    <div className="bg-gray-900/40 rounded-lg p-3">
      <div className="flex items-center justify-between gap-3 mb-1">
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm text-gray-100">{c.ticker}</span>
          {arrow && <span className={`text-xs font-mono ${arrow === "LONG " ? "text-emerald-400" : "text-red-400"}`}>{arrow}</span>}
          <span className="text-xs text-gray-500">[{c.signal_type}]</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-20 h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full"
              style={{ width: `${c.confidence * 100}%`, backgroundColor: color }} />
          </div>
          <span className="font-mono text-xs text-gray-400">{c.confidence.toFixed(3)}</span>
        </div>
      </div>
      <div className="text-xs text-gray-500 font-mono">
        rules: {c.all_rules.join(", ")}
      </div>
    </div>
  );
}

function VarianceBar({ row }: { row: VarianceRow }) {
  // 자기 + 외부 상위 소스 → 가로 막대 (합=1)
  const segments = [
    { src: row.ticker + " (self)", share: row.self_share, color: "#374151" },
    ...row.top_sources.map((s, i) => ({
      src: s.src,
      share: s.share,
      color: ["#8b5cf6", "#f59e0b", "#06b6d4"][i % 3],
    })),
  ];
  const used = segments.reduce((a, s) => a + s.share, 0);
  if (used < 0.999) segments.push({ src: "others", share: 1 - used, color: "#1f2937" });

  return (
    <div className="py-1.5">
      <div className="flex justify-between text-xs mb-1">
        <span className="font-mono text-gray-300">
          <span className="text-gray-500 mr-2">#{row.cholesky_order}</span>
          {row.ticker}
        </span>
        <span className="font-mono text-gray-500">
          self {(row.self_share * 100).toFixed(0)}%  ·  prop {row.propagation_score.toFixed(2)}
        </span>
      </div>
      <div className="h-3 bg-gray-900 rounded overflow-hidden flex">
        {segments.map((s, i) => (
          <div key={i}
            title={`${s.src}: ${(s.share * 100).toFixed(1)}%`}
            style={{ width: `${s.share * 100}%`, backgroundColor: s.color }} />
        ))}
      </div>
      {row.top_sources.length > 0 && (
        <div className="text-xs text-gray-600 mt-1 font-mono">
          {row.top_sources.map((s, i) => (
            <span key={i} className="mr-3">
              ← {s.src} {(s.share * 100).toFixed(1)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function JointSimBlock({ sim }: { sim: JointSim }) {
  return (
    <div className="rounded-lg bg-emerald-950/15 border border-emerald-800/40 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-xs font-mono text-emerald-300">JOINT DRAWDOWN — NORTA + Cholesky 시뮬</h4>
          <p className="text-[10px] text-gray-500 mt-0.5">
            horizon {sim.horizon}영업일 · n_sim {sim.n_sim.toLocaleString()} · 시그널 무관, 시장 상관/변동성 구조만으로 본 위험도
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        {sim.per_threshold.map((row) => (
          <div key={row.threshold} className="bg-gray-900/40 rounded p-3">
            <div className="text-xs text-gray-500 mb-2 font-mono">
              τ = {(row.threshold * 100).toFixed(1)}% drawdown
            </div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">P(≥1 섹터 hit)</span>
                <span className="font-mono text-emerald-300">{(row.p_any * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">P(≥절반 hit)</span>
                <span className="font-mono text-amber-300">{(row.p_half * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">P(전부 hit)</span>
                <span className="font-mono text-red-300">{(row.p_all * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between pt-1 border-t border-gray-800">
                <span className="text-gray-500">평균 hit 섹터</span>
                <span className="font-mono text-gray-400">{row.mean_hit_count.toFixed(2)} / 11</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-900/40 rounded p-3">
        <div className="text-xs text-gray-500 mb-2 font-mono">
          상위 페어 공동 drawdown (τ = {(sim.pair_threshold * 100).toFixed(0)}%)
        </div>
        <div className="flex flex-wrap gap-2 text-xs font-mono">
          {sim.top_pair_co_drawdown.slice(0, 8).map((p, i) => (
            <span key={i} className="bg-gray-900 rounded px-2 py-1">
              <span className="text-gray-300">{p.a}-{p.b}</span>
              <span className="ml-2 text-emerald-300">{(p.p_co * 100).toFixed(1)}%</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function LiveSignalsBlock({ live }: { live: LiveSignals }) {
  return (
    <div className="rounded-xl border border-cyan-700/50 bg-cyan-950/15 p-6 space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">LIVE</span>
            <h3 className="text-sm font-semibold text-gray-100">Dual-Strategy Signals + Variance Decomposition</h3>
          </div>
          <p className="text-xs text-gray-400">
            온톨로지 실시간 출력. 듀얼 전략: 스트래들(크기 베팅) ⊥ 방향성(현물 롱/숏).
            Cholesky 분산 분해는 평상시 가우시안 의존성으로 '왜 이 섹터가 움직였나'를 직교 충격으로 귀인.
          </p>
        </div>
        <div className="text-right text-xs bg-gray-900/60 rounded-lg p-3 min-w-[160px]">
          <div className="text-gray-500 mb-0.5">활성 레짐 (확정)</div>
          <div className="font-mono text-cyan-300 text-base">{live.active_regime}</div>
          {live.regime_state?.pending && (
            <div className="mt-1 inline-block text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-900/50 text-amber-300">
              raw {live.regime_state.pending} 보류 {live.regime_state.pending_count}/{live.regime_state.confirm_cycles}
            </div>
          )}
          {live.regime_state?.vix != null && (
            <div className="text-gray-500 mt-0.5 text-[10px]">VIX {live.regime_state.vix.toFixed(2)} · 연속 {live.regime_state?.confirm_cycles ?? 2}사이클 확인 후 전환</div>
          )}
          <div className="text-gray-600 mt-0.5 text-[10px]">{live.generated.slice(0, 16)}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Long Straddle candidates */}
        <div className="rounded-lg bg-gray-900/30 p-4 space-y-2">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-mono text-gray-500">LONG STRADDLE — 크기 베팅 ({live.straddle_candidates.length})</h4>
          </div>
          <p className="text-[10px] text-gray-600 mb-2">high_vix 위기 룰 → 롱 옵션</p>
          {live.straddle_candidates.length === 0 && (
            <p className="text-xs text-gray-600">발화 없음 (mid_vix 정상)</p>
          )}
          {live.straddle_candidates.slice(0, 8).map((c) => (
            <CandidateRow key={c.ticker} c={c} color="#8b5cf6" />
          ))}
        </div>

        {/* Directional candidates */}
        <div className="rounded-lg bg-gray-900/30 p-4 space-y-2">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-mono text-gray-500">DIRECTIONAL — 방향 베팅 ({live.directional_candidates.length})</h4>
          </div>
          <p className="text-[10px] text-gray-600 mb-2">rate beta · macro sensitivity</p>
          {live.directional_candidates.slice(0, 8).map((c) => (
            <CandidateRow key={c.ticker} c={c} color="#06b6d4" isDirectional />
          ))}
        </div>

        {/* Short Straddle candidates (NEW 2026-06-10) */}
        <div className="rounded-lg bg-emerald-950/15 border border-emerald-700/40 p-4 space-y-2">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-mono text-emerald-300">SHORT STRADDLE — vol 매도 ({live.short_straddle_candidates?.length ?? 0})</h4>
          </div>
          <p className="text-[10px] text-gray-600 mb-2">vol_overpriced ∩ thin_tail_greenlight</p>
          {(!live.short_straddle_candidates || live.short_straddle_candidates.length === 0) && (
            <p className="text-xs text-gray-600">발화 없음 (그린라이트 미충족)</p>
          )}
          {(live.short_straddle_candidates ?? []).slice(0, 8).map((c) => (
            <CandidateRow key={c.ticker} c={c} color="#10b981" />
          ))}
        </div>
      </div>

      {/* Sensitivity audit — raw vs controlled delta (2026-06-12) */}
      {live.sensitivity_audit && live.sensitivity_audit.length > 0 && (
        <SensitivityAuditBlock rows={live.sensitivity_audit} />
      )}

      {/* Variance attribution matrix */}
      <div className="rounded-lg bg-gray-900/30 p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-mono text-gray-500">CHOLESKY VARIANCE DECOMPOSITION</h4>
          <span className="text-[10px] text-gray-600">order ↑ = 시스템 충격원 (eigenvector centrality)</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
          {live.variance_decomposition.by_sector.map((row) => (
            <VarianceBar key={row.ticker} row={row} />
          ))}
        </div>
        <p className="text-xs text-gray-600 mt-3">
          진한 회색 = 자기 직교 충격 비중(self_share) | 보라/주황/청록 = 외부 충격 소스 상위 3 | prop = 자기 충격이 다른 섹터 분산에 기여한 총합
        </p>
      </div>

      {/* Joint simulation block */}
      {live.joint_simulation && <JointSimBlock sim={live.joint_simulation} />}

      {/* Shock propagators + variance concentrated */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-lg bg-purple-950/20 border border-purple-800/40 p-4">
          <h4 className="text-xs font-mono text-purple-300 mb-2">SHOCK PROPAGATOR ({live.variance_decomposition.shock_propagators.length})</h4>
          <p className="text-[10px] text-gray-500 mb-3">propagation_score ≥ 0.50 — 시스템 충격원 / 스트레스 진앙 prior</p>
          {live.variance_decomposition.shock_propagators.length === 0 && (
            <p className="text-xs text-gray-600">현재 발화 없음.</p>
          )}
          {live.variance_decomposition.shock_propagators.map((p) => (
            <div key={p.ticker} className="bg-gray-900/40 rounded p-2 mb-2">
              <div className="flex justify-between text-xs">
                <span className="font-mono text-purple-200">{p.ticker}</span>
                <span className="font-mono text-gray-400">prop {p.propagation_score.toFixed(2)} · conf {p.confidence.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-lg bg-amber-950/20 border border-amber-800/40 p-4">
          <h4 className="text-xs font-mono text-amber-300 mb-2">VARIANCE CONCENTRATED ({live.variance_decomposition.variance_concentrated.length})</h4>
          <p className="text-[10px] text-gray-500 mb-3">self_share ≥ 0.70 — 다른 섹터와 디커플링 / 헤지 효과 약함</p>
          {live.variance_decomposition.variance_concentrated.length === 0 && (
            <p className="text-xs text-gray-600">현재 발화 없음.</p>
          )}
          {live.variance_decomposition.variance_concentrated.map((c) => (
            <div key={c.ticker} className="bg-gray-900/40 rounded p-2 mb-2">
              <div className="flex justify-between text-xs">
                <span className="font-mono text-amber-200">{c.ticker}</span>
                <span className="font-mono text-gray-400">self {(c.self_share * 100).toFixed(0)}% · conf {c.confidence.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Sensitivity audit — 이변량(raw) vs 다변량 통제(ctrl) delta (2026-06-12)
// ────────────────────────────────────────────────────────────────────────

const VERDICT_STYLE: Record<AuditRow["verdict"], { label: string; cls: string }> = {
  killed:    { label: "KILLED",    cls: "bg-red-900/50 text-red-300" },
  flipped:   { label: "FLIPPED",   cls: "bg-red-900/70 text-red-200" },
  emerged:   { label: "EMERGED",   cls: "bg-blue-900/50 text-blue-300" },
  confirmed: { label: "CONFIRMED", cls: "bg-emerald-900/50 text-emerald-300" },
};

function SensitivityAuditBlock({ rows }: { rows: AuditRow[] }) {
  const nKilled = rows.filter((r) => r.verdict === "killed" || r.verdict === "flipped").length;
  return (
    <div className="rounded-lg bg-gray-900/30 border border-red-900/30 p-4">
      <div className="flex items-center justify-between mb-1">
        <h4 className="text-xs font-mono text-gray-400">
          SENSITIVITY AUDIT — raw vs controlled delta
        </h4>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-900/40 text-red-300">
          교란 판정 {nKilled}건
        </span>
      </div>
      <p className="text-[10px] text-gray-500 mb-3">
        raw = 이변량 OLS (매크로 1개씩) · ctrl = 전 매크로(OIL 포함) 동시 + SPY 통제 partial.
        KILLED = 통제 후 유의성 소멸(교란이었음 — 룰 발화 차단) · CONFIRMED = 양쪽 생존(진짜 민감도).
        rate_* 룰은 ctrl delta 로 판정.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-[10px]">
              <th className="text-left py-1.5 pr-3">Sector × Macro</th>
              <th className="text-right py-1.5 pr-3">δ raw (t)</th>
              <th className="text-right py-1.5 pr-3">δ ctrl (t)</th>
              <th className="text-left py-1.5">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const v = VERDICT_STYLE[r.verdict];
              return (
                <tr key={`${r.sector}-${r.macro}`} className="border-b border-gray-800/30 hover:bg-gray-900/20">
                  <td className="py-1.5 pr-3 font-mono">
                    <span className="text-gray-200">{r.sector}</span>
                    <span className="text-gray-600"> × </span>
                    <span className="text-purple-300">{r.macro}</span>
                  </td>
                  <td className="py-1.5 pr-3 text-right font-mono text-gray-400">
                    {r.delta_raw != null ? r.delta_raw.toFixed(4) : "—"}
                    <span className="text-gray-600"> ({r.t_raw != null ? r.t_raw.toFixed(1) : "—"})</span>
                  </td>
                  <td className="py-1.5 pr-3 text-right font-mono text-gray-300">
                    {r.delta_ctrl != null ? r.delta_ctrl.toFixed(4) : "—"}
                    <span className="text-gray-600"> ({r.t_ctrl != null ? r.t_ctrl.toFixed(1) : "—"})</span>
                  </td>
                  <td className="py-1.5">
                    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${v.cls}`}>{v.label}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// IV History block — Alpaca options snapshot 시계열 (2026-06-09~ 누적)
// ────────────────────────────────────────────────────────────────────────

const SECTOR_COLORS: Record<string, string> = {
  XLK: "#8b5cf6", XLF: "#06b6d4", XLE: "#f59e0b", XLV: "#10b981",
  XLY: "#ec4899", XLP: "#a78bfa", XLU: "#fbbf24", XLI: "#3b82f6",
  XLB: "#84cc16", XLRE: "#ef4444", XLC: "#22d3ee",
};

function IVHistoryBlock({ hist }: { hist: IVHistory }) {
  // records → {ts, XLK_vrp, XLF_vrp, ...} 형태로 pivot
  const byTs: Map<string, Record<string, number | string>> = new Map();
  for (const r of hist.records) {
    const ts = r.ts.slice(5, 16).replace("T", " "); // MM-DD HH:MM
    let row = byTs.get(ts);
    if (!row) { row = { ts }; byTs.set(ts, row); }
    row[`${r.ticker}_iv`] = +(r.iv_atm * 100).toFixed(2);
    row[`${r.ticker}_vrp`] = +(r.vrp_iv * 100).toFixed(2);
  }
  const points = Array.from(byTs.values()).sort(
    (a, b) => (a.ts as string).localeCompare(b.ts as string)
  );
  const tickers = Object.keys(SECTOR_COLORS);

  // 최신 시점 VRP 분포 (vol_overpriced 임계 +2% 위 섹터 파악용)
  const latest = points[points.length - 1] || {};
  const latestVrp = tickers
    .map((t) => ({ t, v: (latest[`${t}_vrp`] as number) ?? null }))
    .filter((x) => x.v !== null)
    .sort((a, b) => (b.v as number) - (a.v as number));

  return (
    <div className="rounded-xl border border-amber-700/50 bg-amber-950/15 p-6 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">
              IV TIMESERIES
            </span>
            <h3 className="text-sm font-semibold text-gray-100">
              섹터 ATM IV / VRP_true 시계열 (Alpaca options snapshot)
            </h3>
          </div>
          <p className="text-xs text-gray-400">
            매시간 누적 — 1주 후 ~1,848행. vol_overpriced 임계 +2% reference 선 위
            섹터가 STRADDLE ×0.5 페널티 대상. 2026-06-09부터 시작.
          </p>
        </div>
        <div className="text-right text-xs bg-gray-900/60 rounded-lg p-3 min-w-[140px]">
          <div className="text-gray-500 mb-0.5">누적 행</div>
          <div className="font-mono text-amber-300 text-base">{hist.n_records}</div>
          <div className="text-gray-600 mt-0.5 text-[10px]">{hist.generated.slice(0, 16)}</div>
        </div>
      </div>

      {/* VRP_true 시계열 LineChart */}
      <div className="rounded-lg bg-gray-900/30 p-4">
        <h4 className="text-xs font-mono text-gray-500 mb-3">VRP_true (= ATM_IV − RV_20d) 시계열, % pp 단위</h4>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={points} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="ts" tick={{ fontSize: 9, fill: "#6b7280" }}
                   interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 9, fill: "#6b7280" }}
                   label={{ value: "VRP (%pp)", angle: -90, position: "insideLeft",
                            fill: "#6b7280", fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: "#111827",
                                     border: "1px solid #374151", fontSize: 10 }} />
            <Legend wrapperStyle={{ fontSize: 9 }} />
            <ReferenceLine y={2} stroke="#f59e0b" strokeDasharray="4 2"
                           label={{ value: "+2pp 임계", fontSize: 9, fill: "#f59e0b" }} />
            <ReferenceLine y={0} stroke="#374151" strokeWidth={0.5} />
            {tickers.map((t) => (
              <Line key={t} type="monotone" dataKey={`${t}_vrp`}
                    stroke={SECTOR_COLORS[t]} dot={false} strokeWidth={1.2}
                    name={t} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 최신 시점 VRP 분포 (정렬) */}
      <div className="rounded-lg bg-gray-900/30 p-4">
        <h4 className="text-xs font-mono text-gray-500 mb-3">
          최신 VRP_true 분포 (높은 순) — +2pp 위 섹터가 STRADDLE 페널티 대상
        </h4>
        <div className="flex flex-wrap gap-2">
          {latestVrp.map(({ t, v }) => {
            const over = (v as number) > 2;
            return (
              <span key={t}
                    className={`font-mono text-xs px-2 py-1 rounded ${
                      over ? "bg-red-900/40 text-red-300 border border-red-700/40"
                           : "bg-gray-900 text-gray-300"
                    }`}>
                {t} <span className="ml-1 text-gray-500">{(v as number).toFixed(2)}pp</span>
              </span>
            );
          })}
        </div>
      </div>

      {/* ATM IV 시계열 LineChart */}
      <div className="rounded-lg bg-gray-900/30 p-4">
        <h4 className="text-xs font-mono text-gray-500 mb-3">ATM IV (연환산, %)</h4>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={points} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="ts" tick={{ fontSize: 9, fill: "#6b7280" }}
                   interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 9, fill: "#6b7280" }}
                   label={{ value: "IV (%)", angle: -90, position: "insideLeft",
                            fill: "#6b7280", fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: "#111827",
                                     border: "1px solid #374151", fontSize: 10 }} />
            <Legend wrapperStyle={{ fontSize: 9 }} />
            {tickers.map((t) => (
              <Line key={t} type="monotone" dataKey={`${t}_iv`}
                    stroke={SECTOR_COLORS[t]} dot={false} strokeWidth={1.2}
                    name={t} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
