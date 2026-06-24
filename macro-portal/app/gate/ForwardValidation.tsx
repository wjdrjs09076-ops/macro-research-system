"use client";

import { useEffect, useState, type ReactNode } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from "recharts";

type Trade = {
  exit_date: string; entry_date?: string; rule: string | null; strategy: string; ticker: string;
  regime: string | null; pnl: number; entry_cost: number;
  exit_reason: string | null; holding_days: number | null;
  validation?: string; exclude_reason?: string | null;
};
type FwdData = {
  generated: string;
  oos_forward_start: string;
  slippage_zero: { roundtrip_pct: number; basis: string };
  cluster_min_n: number;
  n_closed: number;
  clean_n?: number;
  excluded_n?: number;
  boundary_note?: string;
  timing_note?: string;
  churn_diagnostic?: { definition: string; flickers: number; cum_pnl: number; verdict_note: string };
  closed_trades: Trade[];
};

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export default function ForwardValidation() {
  const [data, setData] = useState<FwdData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/forward_validation.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setErr(String(e?.message || e)));
  }, []);

  const Section = ({ children }: { children: ReactNode }) => (
    <div className="border-t border-gray-800 pt-8 mt-8">
      <h2 className="text-xl font-bold text-gray-100">
        Forward Validation — Clustering &amp; Regime Distribution
      </h2>
      <p className="text-gray-400 text-sm mt-1 leading-relaxed">
        위 패널이 &quot;얼마 벌었나&quot;라면, 여기는 &quot;그 손익이 믿을 만한가&quot; — 운(소수 큰-무브) vs
        실력(룰 예측력)을 가른다. 닫힌 전향 표본만(미실현 제외), freeze v2(2026-06-16)부터. 평가 기준은
        표본을 보기 전에 동결한다.
      </p>
      {children}
    </div>
  );

  if (err)
    return (
      <Section>
        <div className="text-amber-400 text-sm mt-4">데이터 로드 실패 ({err}) — 새로고침(Ctrl+Shift+R).</div>
      </Section>
    );
  if (!data)
    return (
      <Section>
        <div className="text-gray-500 text-sm mt-4">Loading...</div>
      </Section>
    );

  const slipPct = data.slippage_zero.roundtrip_pct;

  // churn 분해 기준선 — 검증 보조(전시물 아님). 표본 손익을 채점할 때 깔려 있는
  // 메커니즘 비용을 보여, clean 표본 패배를 '테제 오류 vs churn 슬리피지'로 가르는 기준.
  const churn = data.churn_diagnostic;
  const ChurnBaseline = () =>
    churn ? (
      <div className="text-gray-600 text-[11px] mt-3 leading-relaxed">
        <span className="text-gray-500">churn 분해 기준선:</span> event_vol 임계경계 깜빡임 왕복{" "}
        {churn.flickers}회 · 누적 슬리피지 {churn.cum_pnl < 0 ? "−" : ""}$
        {Math.abs(churn.cum_pnl).toFixed(0)}. 표본 손익을 채점할 때 이만큼의 메커니즘 비용이 깔려 있다 —
        clean 표본이 지면 그 패배를 &quot;테제 오류 vs churn 슬리피지&quot;로 분해하는 기준선이다. (룰 임계
        z≥2.5는 freeze 불변, 발화 히스테리시스 판정은 ~6/30.)
      </div>
    ) : null;

  // ── 빈 상태 (전향 표본 누적 전) ──
  if (data.n_closed === 0) {
    return (
      <Section>
        <div className="mt-4 rounded border border-dashed border-gray-700 p-6">
          <div className="text-gray-200 text-sm font-semibold">전향 표본 누적 중 — 아직 0건</div>
          <div className="text-gray-400 text-xs mt-3 leading-relaxed">
            첫 표본 예상 ~6/30 (XLB DTE≤10), 페어 ~7/7. 표본이 닫힐 때마다 아래 타임라인에 점이 하나씩 찍힌다.
            <br />
            <span className="text-gray-300">★ 평가 프레임 (표본 보기 전 동결):</span> 영점은 0 이 아니라{" "}
            <b className="text-gray-200">왕복 슬리피지 비용대 ({(slipPct * 100).toFixed(0)}%, n=2 추정)</b>.
            비용을 넘긴 표본만 <span className="text-emerald-400">초록</span>, 부호만 양수면{" "}
            <span className="text-gray-300">회색</span>(슬리피지 미달), 음수는 <span className="text-red-400">빨강</span>.
            빨강이 시간적으로 뭉치면 missing-risk 신호. n&lt;{data.cluster_min_n} 이면 군집 판정 보류.
          </div>
        </div>
        <div className="text-gray-600 text-[11px] mt-3">
          View 1 (Excession Timeline) 먼저 배포 — 빈 패널이 매일 &quot;아직 0&quot;을 보여주는 게 &quot;손 떼고 기다리는 중&quot;의
          증언. View 2(레짐 분포)·3(추정/보수 분리)은 표본 쌓인 뒤.
        </div>
        <ChurnBaseline />
      </Section>
    );
  }

  // ── View 1 — Excession Timeline ──
  const cleanN = data.clean_n ?? data.n_closed;
  const excludedN = data.excluded_n ?? 0;
  const pts = data.closed_trades.map((t, i) => {
    const hurdle = Math.abs(t.entry_cost) * slipPct;
    const excluded = t.validation === "excluded";
    const cat = excluded ? "excluded" : t.pnl <= 0 ? "loss" : t.pnl >= hurdle ? "win" : "gray";
    return { ...t, idx: i, hurdle, cat };
  });
  const colorOf = (c: string) =>
    c === "win" ? "#34d399" : c === "loss" ? "#f87171" : c === "gray" ? "#9ca3af" : "#4b5563";
  const cleanHurdles = pts.filter((p) => p.cat !== "excluded").map((p) => p.hurdle);
  const medHurdle = median(cleanHurdles.length ? cleanHurdles : pts.map((p) => p.hurdle));
  const enoughN = cleanN >= data.cluster_min_n;

  return (
    <Section>
      <div className="mt-4">
        <div className="text-sm text-gray-200 font-semibold">
          View 1 — Excession Timeline (군집 검사) · clean n={cleanN} / 제외 {excludedN}
        </div>
        <div className="text-xs text-gray-500 mb-2">
          영점 0(실선) / 슬리피지 비용대 ±{(slipPct * 100).toFixed(0)}%(점선).
          <span className="text-emerald-400"> 초록</span>=비용 넘김 ·
          <span className="text-gray-300"> 회색</span>=부호만 양수(미달) ·
          <span className="text-red-400"> 빨강</span>=손실 ·
          <span style={{ color: "#4b5563" }}> 흐린회색</span>=검증 제외(오염).
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="exit_date" tick={{ fontSize: 11, fill: "#9ca3af" }} />
            <YAxis dataKey="pnl" tick={{ fontSize: 11, fill: "#9ca3af" }} />
            <ReferenceLine y={0} stroke="#6b7280" />
            <ReferenceLine y={medHurdle} stroke="#6b7280" strokeDasharray="4 4" />
            <ReferenceLine y={-medHurdle} stroke="#6b7280" strokeDasharray="4 4" />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 12 }}
              formatter={(value, _name, item) => {
                const d = (item?.payload ?? {}) as Trade & { hurdle: number };
                return [
                  `$${value} (${d.rule}|${d.strategy}, ${d.regime}, 비용대 $${(d.hurdle ?? 0).toFixed(0)})`,
                  d.ticker ?? "",
                ];
              }}
            />
            <Scatter data={pts}>
              {pts.map((p, i) => (
                <Cell key={i} fill={colorOf(p.cat)} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        {cleanN === 0 && data.n_closed > 0 && (
          <div className="text-amber-400 text-xs mt-2">
            아직 clean 전향 표본 0 — 닫힌 {data.n_closed}건 전부 임계/origin 오염으로 검증 제외. 진짜 첫 표본 = freeze 후 새 이벤트 발화부터.
          </div>
        )}
        {!enoughN && cleanN > 0 && (
          <div className="text-gray-500 text-xs mt-2">
            clean 표본 부족 (n={cleanN} &lt; {data.cluster_min_n}) — 군집 판정 보류.
          </div>
        )}
        {data.boundary_note && (
          <div className="text-gray-600 text-[11px] mt-2">경계: {data.boundary_note}</div>
        )}
        {data.timing_note && (
          <div className="text-gray-600 text-[11px] mt-1">타이밍: {data.timing_note}</div>
        )}
        <div className="text-gray-600 text-[11px] mt-1">영점 기준: {data.slippage_zero.basis}</div>
        <ChurnBaseline />
      </div>
    </Section>
  );
}
