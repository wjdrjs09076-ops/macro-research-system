"use client";

import { useEffect, useState } from "react";
import {
  ComposedChart, LineChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ReferenceArea, ResponsiveContainer, Legend,
} from "recharts";

type StrategyData = {
  oos_cagr: number; oos_sharpe: number; oos_mdd: number | null;
  oos_active_days: number; alpha_capture: number;
  cumulative: { dates: string[]; values: number[] };
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
};

const STRATEGY_COLORS: Record<string, string> = {
  "Pure ML" : "#ef4444",
  "Gate Only": "#10b981",
  "Gate+ML"  : "#8b5cf6",
  "Gate+S2"  : "#f59e0b",
  "BnH XLE"  : "#6b7280",
};

const STRATEGY_ORDER = ["Gate Only", "Gate+S2", "Pure ML", "Gate+ML", "BnH XLE"];

function MetricCard({
  name, cagr, sharpe, mdd, days, capture, color,
}: {
  name: string; cagr: number; sharpe: number; mdd: number | null;
  days: number; capture: number; color: string;
}) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-200">{name}</span>
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
          <div className="text-gray-500">Alpha Capture</div>
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
                capture={s.alpha_capture} color={STRATEGY_COLORS[name]}
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

      {/* Insight Box */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">핵심 해석</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-gray-400">
          <div className="bg-gray-900 rounded-lg p-3">
            <div className="text-emerald-400 font-medium mb-1">Gate Only (+23.5%)</div>
            온톨로지 레짐 필터 자체가 유효한 신호 — Pure ML(+8.8%) 대비 3배 CAGR
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <div className="text-purple-400 font-medium mb-1">Gate+ML 활성 2일만</div>
            게이트 조건 + ML 조건 동시 충족이 극히 드묾 — 신호 자체는 유효하나 과도한 필터링
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <div className="text-gray-400 font-medium mb-1">BnH XLE (+85.9%)</div>
            저VIX 구간(VIX&lt;20)이 전체 수익의 대부분 — 위기 신호 기반 전략의 구조적 gap
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <div className="text-amber-400 font-medium mb-1">다음 개선 방향</div>
            임계값 0.45를 MEF plot 기반 데이터-driven u로 교체 → Layer 3 Bridge 구현
          </div>
        </div>
      </div>
    </div>
  );
}
