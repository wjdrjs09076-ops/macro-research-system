"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Legend, Cell,
} from "recharts";

// ── 타입 ──────────────────────────────────────────────────────────────────

type DiagPoint = { u: number; mef?: number | null; n?: number };
type HillPoint = { k: number; hill: number; u: number };
type StabPoint = { u: number; xi: number | null; beta_star: number | null; n: number };

type CrisisData = {
  label: string; etf: string; color: string; direction: string;
  garch_params: Record<string, number | string>;
  threshold: {
    u_star: number; u_star_quantile: number;
    n_exceed: number | null; n_total: number | null; zeta_u: number | null;
  };
  gpd: { xi: number | null; beta: number | null; interpretation: string | null };
  tail_asymmetry: {
    xi_loss: number; xi_gain: number;
    asymmetry: number; interpretation: string;
  };
  risk_measures: { var_99: number; es_99: number; var_95: number; es_95: number };
  diagnostics: { mef: DiagPoint[]; hill: HillPoint[]; stability: StabPoint[] };
};

type EvtData = {
  generated_at: string;
  crises: Record<string, CrisisData>;
};

// ── 상수 ──────────────────────────────────────────────────────────────────

const ORDER  = ["hormuz", "gfc", "covid", "dotcom"];
const LABELS: Record<string, string> = {
  hormuz: "호르무즈", gfc: "GFC", covid: "COVID", dotcom: "닷컴",
};
type DiagTab = "mef" | "hill" | "stability";

// ── 서브 컴포넌트 ─────────────────────────────────────────────────────────

function Badge({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span className="inline-block text-xs font-mono px-2 py-0.5 rounded"
      style={{ backgroundColor: color + "33", color }}>
      {children}
    </span>
  );
}

function StatBox({ label, value, sub, highlight }: {
  label: string; value: string; sub?: string; highlight?: boolean;
}) {
  return (
    <div className={`rounded-lg p-3 ${highlight ? "bg-emerald-950/40 border border-emerald-800/50" : "bg-gray-900"}`}>
      <div className="text-xs text-gray-500 mb-0.5">{label}</div>
      <div className={`font-mono text-sm ${highlight ? "text-emerald-300" : "text-gray-100"}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  );
}

// ── 진단 플롯들 ───────────────────────────────────────────────────────────

function MefChart({ data, uStar, color }: { data: DiagPoint[]; uStar: number; color: string }) {
  const pts = data.filter(d => d.mef != null).map(d => ({ u: +d.u.toFixed(4), mef: +(d.mef as number).toFixed(4) }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={pts} margin={{ top: 5, right: 15, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis dataKey="u" tick={{ fontSize: 9, fill: "#6b7280" }} tickFormatter={v => v.toFixed(3)} />
        <YAxis tick={{ fontSize: 9, fill: "#6b7280" }} />
        <Tooltip contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 10 }}
          formatter={(v) => [typeof v === "number" ? v.toFixed(4) : v, "e(u)"]} />
        <ReferenceLine x={uStar} stroke="white" strokeDasharray="3 2"
          label={{ value: `u*=${uStar.toFixed(3)}`, fontSize: 9, fill: "white" }} />
        <Line type="monotone" dataKey="mef" stroke={color} dot={false} strokeWidth={1.8} name="e(u)" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function HillChart({ data, xi, color }: { data: HillPoint[]; xi: number | null; color: string }) {
  const pts = data.filter(d => !isNaN(d.hill)).map(d => ({ k: d.k, hill: +d.hill.toFixed(4) }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={pts} margin={{ top: 5, right: 15, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis dataKey="k" tick={{ fontSize: 9, fill: "#6b7280" }} />
        <YAxis tick={{ fontSize: 9, fill: "#6b7280" }} />
        <Tooltip contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 10 }}
          formatter={(v) => [typeof v === "number" ? v.toFixed(4) : v, "H_k"]} />
        {xi != null && (
          <ReferenceLine y={xi} stroke="white" strokeDasharray="3 2"
            label={{ value: `ξ=${xi.toFixed(3)}`, fontSize: 9, fill: "white" }} />
        )}
        <ReferenceLine y={0} stroke="#374151" />
        <Line type="monotone" dataKey="hill" stroke={color} dot={false} strokeWidth={1.5} name="H_k" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function StabilityChart({ data, uStar, color }: { data: StabPoint[]; uStar: number; color: string }) {
  const pts = data.filter(d => d.xi != null).map(d => ({
    u: +d.u.toFixed(4),
    xi: d.xi != null ? +d.xi.toFixed(4) : null,
    beta_star: d.beta_star != null ? +d.beta_star.toFixed(4) : null,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={pts} margin={{ top: 5, right: 15, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis dataKey="u" tick={{ fontSize: 9, fill: "#6b7280" }} tickFormatter={v => (+v).toFixed(3)} />
        <YAxis tick={{ fontSize: 9, fill: "#6b7280" }} />
        <Tooltip contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 10 }}
          formatter={(v) => [typeof v === "number" ? v.toFixed(4) : v]} />
        <ReferenceLine x={uStar} stroke="white" strokeDasharray="3 2"
          label={{ value: `u*=${uStar.toFixed(3)}`, fontSize: 9, fill: "white" }} />
        <ReferenceLine y={0} stroke="#374151" />
        <Line type="monotone" dataKey="xi" stroke={color} dot={false} strokeWidth={1.8} name="ξ(u)" />
        <Line type="monotone" dataKey="beta_star" stroke="#f59e0b" dot={false} strokeWidth={1.2}
          strokeDasharray="3 2" name="β*(u)" />
        <Legend wrapperStyle={{ fontSize: 10 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── 메인 페이지 ───────────────────────────────────────────────────────────

export default function EvtPage() {
  const [data,    setData]    = useState<EvtData | null>(null);
  const [active,  setActive]  = useState("hormuz");
  const [diagTab, setDiagTab] = useState<DiagTab>("mef");

  useEffect(() => {
    fetch("/data/evt_bridge.json").then(r => r.json()).then(setData);
  }, []);

  if (!data) return <div className="text-gray-500 text-sm">Loading...</div>;

  const crisis = data.crises[active];
  if (!crisis) return null;

  const { color } = crisis;
  const xi   = crisis.gpd.xi;
  const asym = crisis.tail_asymmetry;

  // 비교용 모든 위기 데이터
  const compRows = ORDER.filter(k => data.crises[k]).map(k => {
    const c = data.crises[k];
    return {
      name: LABELS[k],
      key: k,
      color: c.color,
      xi:    c.gpd.xi ?? 0,
      es_99: c.risk_measures.es_99,
      asym:  c.tail_asymmetry.asymmetry,
      u_pct: c.threshold.u_star_quantile,
    };
  });

  return (
    <div className="space-y-8">

      {/* 헤더 */}
      <div>
        <h1 className="text-2xl font-bold text-gray-100">EVT Bridge — Layer 3</h1>
        <p className="text-gray-400 text-sm mt-1 max-w-3xl">
          GJR-GARCH 표준화 잔차에 MEF·Hill·Stability 진단으로 임계값 u*를 데이터에서 결정.
          u* 이하 = 본체(Moments), u* 초과 = 꼬리(GPD). 현재 gate_score 0.45를 대체하는 근거.
        </p>
      </div>

      {/* 위기 탭 */}
      <div className="flex gap-2 flex-wrap">
        {ORDER.filter(k => data.crises[k]).map(key => {
          const c = data.crises[key];
          return (
            <button key={key} onClick={() => setActive(key)}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all"
              style={active === key
                ? { backgroundColor: c.color, color: "#030712" }
                : { backgroundColor: "#1f2937", color: "#9ca3af" }}>
              {LABELS[key]}
              <span className="ml-1.5 text-xs opacity-70">{c.gpd.xi?.toFixed(3) ?? "—"}</span>
            </button>
          );
        })}
      </div>

      {/* Crisis 요약 */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-5">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <Badge color={color}>{crisis.label.split("/")[1]?.trim()}</Badge>
              <span className="text-gray-400 text-xs">{crisis.direction} | {crisis.etf}</span>
            </div>
            <h2 className="text-base font-semibold text-gray-100">{crisis.label}</h2>
          </div>
        </div>

        {/* GARCH + EVT 핵심 수치 */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
          <StatBox label="u* (임계값)"
            value={crisis.threshold.u_star.toFixed(4)}
            sub={`P${crisis.threshold.u_star_quantile.toFixed(0)}`}
            highlight />
          <StatBox label="초과 관측 수"
            value={`${crisis.threshold.n_exceed ?? "—"}`}
            sub={`/ ${crisis.threshold.n_total ?? "—"} 전체`} />
          <StatBox label="GPD ξ (꼬리 지수)"
            value={xi != null ? xi.toFixed(3) : "—"}
            sub={crisis.gpd.interpretation ?? ""}
            highlight={xi != null && xi > 0.1} />
          <StatBox label="ES @99%"
            value={crisis.risk_measures.es_99.toFixed(4) + "σ"}
            sub="표준화 잔차 기준" />
          <StatBox label="VaR @99%"
            value={crisis.risk_measures.var_99.toFixed(4) + "σ"} />
          <StatBox
            label="꼬리 비대칭"
            value={(asym.asymmetry >= 0 ? "+" : "") + asym.asymmetry.toFixed(3)}
            sub="loss_ξ − gain_ξ"
            highlight={asym.asymmetry > 0.15} />
          <StatBox label="GJR gamma"
            value={(crisis.garch_params.gamma as number)?.toFixed(3) ?? "—"}
            sub="레버리지 효과" />
        </div>

        {/* 꼬리 비대칭 해석 */}
        <div className={`mt-3 rounded-lg px-4 py-2.5 text-xs ${
          asym.asymmetry > 0.15
            ? "bg-emerald-950/40 border border-emerald-800/40 text-emerald-300"
            : "bg-gray-800/60 text-gray-400"
        }`}>
          <span className="font-medium">꼬리 비대칭 해석: </span>{asym.interpretation}
          <span className="ml-3 text-gray-500">
            (loss ξ={asym.xi_loss.toFixed(3)} / gain ξ={asym.xi_gain.toFixed(3)})
          </span>
        </div>
      </div>

      {/* 진단 플롯 */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5">
        {/* 탭 */}
        <div className="flex gap-2 mb-4">
          {([
            ["mef",       "MEF Plot",             "선형 구간 시작 = u*"],
            ["hill",      "Hill Plot",             "plateau = ξ 추정"],
            ["stability", "Param Stability",       "ξ·β* 안정 구간 = u*"],
          ] as [DiagTab, string, string][]).map(([tab, label, sub]) => (
            <button key={tab} onClick={() => setDiagTab(tab)}
              className={`px-3 py-1.5 rounded-lg text-xs transition-all ${
                diagTab === tab ? "text-gray-900 font-medium" : "bg-gray-800 text-gray-400"
              }`}
              style={diagTab === tab ? { backgroundColor: color } : {}}>
              {label}
              <span className="ml-1 opacity-60 hidden sm:inline">— {sub}</span>
            </button>
          ))}
        </div>

        {/* 플롯 설명 */}
        <div className="text-xs text-gray-500 mb-3">
          {diagTab === "mef" && "e(u) = E[X−u | X>u]. GPD 데이터에서 u에 대해 선형 — 선형 구간 시작점이 u*"}
          {diagTab === "hill" && "H_k = (1/k) Σ log(X_(i)/X_(k+1)). 안정적인 plateau 구간의 값이 ξ 추정값"}
          {diagTab === "stability" && "여러 u에서 GPD를 반복 피팅. ξ(u)와 β*(u)가 안정되는 구간의 좌측 경계 = u*"}
        </div>

        {diagTab === "mef" && (
          <MefChart data={crisis.diagnostics.mef} uStar={crisis.threshold.u_star} color={color} />
        )}
        {diagTab === "hill" && (
          <HillChart data={crisis.diagnostics.hill} xi={xi} color={color} />
        )}
        {diagTab === "stability" && (
          <StabilityChart data={crisis.diagnostics.stability} uStar={crisis.threshold.u_star} color={color} />
        )}
      </div>

      {/* 4개 위기 비교 */}
      <div>
        <h2 className="text-sm font-semibold text-gray-200 mb-4">4개 위기 EVT 파라미터 비교</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">

          {/* ξ 비교 */}
          <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-4">
            <div className="text-xs text-gray-500 mb-3">GPD ξ (꼬리 지수)</div>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={compRows} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 9, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 10 }}
                  formatter={(v) => [typeof v === "number" ? v.toFixed(3) : v, "ξ"]} />
                <ReferenceLine y={0} stroke="#374151" />
                <Bar dataKey="xi" radius={[3,3,0,0]}>
                  {compRows.map(r => <Cell key={r.key} fill={r.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-600 mt-1">ξ&gt;0: 파레토형 / ξ&lt;0: 유한 꼬리</p>
          </div>

          {/* ES@99% 비교 */}
          <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-4">
            <div className="text-xs text-gray-500 mb-3">ES @99% (표준화 잔차 기준)</div>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={compRows} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 9, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 10 }}
                  formatter={(v) => [typeof v === "number" ? v.toFixed(3) + "σ" : v, "ES"]} />
                <Bar dataKey="es_99" radius={[3,3,0,0]}>
                  {compRows.map(r => <Cell key={r.key} fill={r.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-600 mt-1">단위: 표준화 잔차 σ (실제 손실 = σ × ES)</p>
          </div>

          {/* 꼬리 비대칭 비교 */}
          <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-4">
            <div className="text-xs text-gray-500 mb-3">꼬리 비대칭 (loss_ξ − gain_ξ)</div>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={compRows} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="name" tick={{ fontSize: 9, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 9, fill: "#6b7280" }} />
                <Tooltip contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 10 }}
                  formatter={(v) => [typeof v === "number" ? (v >= 0 ? "+" : "") + v.toFixed(3) : v, "비대칭"]} />
                <ReferenceLine y={0} stroke="#374151" />
                <Bar dataKey="asym" radius={[3,3,0,0]}>
                  {compRows.map(r => (
                    <Cell key={r.key} fill={r.asym > 0.1 ? r.color : "#4b5563"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-600 mt-1">&gt;0.1: 방향성 신뢰 / ≈0: 양방향 위험</p>
          </div>
        </div>

        {/* 핵심 해석 테이블 */}
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500">
                <th className="text-left py-2 pr-4">위기</th>
                <th className="text-right py-2 pr-4">u* 분위수</th>
                <th className="text-right py-2 pr-4">ξ</th>
                <th className="text-right py-2 pr-4">ES@99%</th>
                <th className="text-right py-2 pr-4">꼬리 비대칭</th>
                <th className="text-left py-2">포지션 신뢰도</th>
              </tr>
            </thead>
            <tbody>
              {compRows.map(r => {
                const c = data.crises[r.key];
                const reliable = r.asym > 0.1;
                return (
                  <tr key={r.key} className="border-b border-gray-800/50 hover:bg-gray-900/30">
                    <td className="py-2.5 pr-4">
                      <span className="font-mono text-xs px-1.5 py-0.5 rounded"
                        style={{ backgroundColor: r.color + "22", color: r.color }}>
                        {c.label}
                      </span>
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono text-gray-300">
                      P{r.u_pct.toFixed(0)}
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono text-gray-300">
                      {r.xi.toFixed(3)}
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono text-gray-300">
                      {r.es_99.toFixed(3)}σ
                    </td>
                    <td className={`py-2.5 pr-4 text-right font-mono ${
                      reliable ? "text-emerald-400" : "text-gray-500"
                    }`}>
                      {r.asym >= 0 ? "+" : ""}{r.asym.toFixed(3)}
                    </td>
                    <td className="py-2.5 text-xs">
                      {reliable
                        ? <span className="text-emerald-400">✓ 단방향 — {c.direction} 신뢰</span>
                        : <span className="text-amber-400">△ 양방향 꼬리 — 포지션 주의</span>
                      }
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* gate_score 0.45 vs u* 비교 설명 */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-5 space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Layer 3 Bridge — gate_score 0.45 대체 근거</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="bg-gray-900 rounded-lg p-3 border-l-2 border-amber-500">
            <div className="text-amber-400 font-medium mb-1">현재 (수동 임계값)</div>
            <code className="text-gray-300 block mb-1">gate_score &gt;= 0.45</code>
            <div className="text-gray-500">
              0.45의 근거 없음. 자산 클래스와 위기 유형에 무관하게 동일 적용.
              분포 구조를 보지 않고 경험적 비율만 사용.
            </div>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 border-l-2 border-emerald-500">
            <div className="text-emerald-400 font-medium mb-1">EVT Bridge (데이터 기반)</div>
            <code className="text-gray-300 block mb-1">L_t &gt; u* (MEF 선형화 지점)</code>
            <div className="text-gray-500">
              GPD 피팅이 안정적으로 시작되는 분위수를 데이터가 알려줌.
              위기 유형별로 다른 u* (호르무즈 P70, GFC P77, 닷컴 P74).
              꼬리 체제 진입 시 ES_EVT로 포지션 사이징.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
