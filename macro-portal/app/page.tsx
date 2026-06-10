import Link from "next/link";

const CAUSAL_LINKS = [
  { src: "OIL", dst: "CPI",           lag: 1,  coef: "+0.65", stability: "structural", dir: "+" },
  { src: "CPI", dst: "US2Y",          lag: 3,  coef: "+0.84", stability: "emerging",   dir: "+" },
  { src: "CPI", dst: "US10Y",         lag: 2,  coef: "+0.68", stability: "structural", dir: "+" },
  { src: "US2Y", dst: "FED_FUNDS",    lag: 1,  coef: "+0.92", stability: "structural", dir: "+" },
  { src: "FED_FUNDS", dst: "VIX",     lag: 3,  coef: "-0.78", stability: "structural", dir: "-" },
  { src: "FED_FUNDS", dst: "CREDIT_SPREAD", lag: 1, coef: "+0.71", stability: "structural", dir: "+" },
  { src: "CREDIT_SPREAD", dst: "VIX", lag: 1,  coef: "+0.83", stability: "emerging",   dir: "+" },
  { src: "US10Y", dst: "DXY",         lag: 2,  coef: "-0.62", stability: "emerging",   dir: "-" },
];

const CAUSAL_SECTORS = [
  { ticker: "XLK",  name: "Technology",            delta: 0.0447, chain: "CPI(3m)→US2Y(1m)→FED_FUNDS(3m)→VIX", conf: 0.671 },
  { ticker: "XLE",  name: "Energy",                delta: 0.0391, chain: "OIL(1m)→CPI(3m)→US2Y(1m)→FED_FUNDS(3m)→VIX", conf: 0.643 },
  { ticker: "XLF",  name: "Financials",            delta: 0.0362, chain: "CPI(3m)→US2Y(1m)→FED_FUNDS(3m)→VIX", conf: 0.618 },
  { ticker: "XLY",  name: "Consumer Discr.",       delta: 0.0318, chain: "CPI(3m)→US2Y(1m)→FED_FUNDS(3m)→VIX", conf: 0.594 },
  { ticker: "XLC",  name: "Comm. Services",        delta: 0.0297, chain: "CPI(3m)→US2Y(1m)→FED_FUNDS(3m)→VIX", conf: 0.572 },
  { ticker: "XLI",  name: "Industrials",           delta: 0.0261, chain: "CREDIT_SPREAD(1m)→VIX",               conf: 0.541 },
  { ticker: "XLB",  name: "Materials",             delta: 0.0244, chain: "CREDIT_SPREAD(1m)→VIX",               conf: 0.523 },
  { ticker: "XLRE", name: "Real Estate",           delta: 0.0231, chain: "FED_FUNDS(3m)→VIX",                   conf: 0.508 },
  { ticker: "XLU",  name: "Utilities",             delta: 0.0198, chain: "FED_FUNDS(3m)→VIX",                   conf: 0.487 },
  { ticker: "XLV",  name: "Health Care",           delta: 0.0143, chain: "CREDIT_SPREAD(1m)→VIX",               conf: 0.456 },
];

const LAYERS = [
  {
    id: "L5",
    name: "Layer 5 — Dynamics",
    color: "border-violet-500 bg-violet-950/30",
    badge: "bg-violet-500/20 text-violet-300",
    items: [
      { label: "GJR-GARCH(1,1)", desc: "Asymmetric volatility — 하락 충격이 상승보다 변동성 더 키움" },
      { label: "Conditional σ_t", desc: "일별 조건부 표준편차" },
      { label: "Standardized z_t = r_t / σ_t", desc: "이후 Layer 1·2의 공통 입력" },
    ],
    current: "EWMA(span=32) — 비대칭 미반영",
    note: "현재 compute_vrp()에서 사용 중",
  },
  {
    id: "L4",
    name: "Layer 4 — Dependence",
    color: "border-blue-500 bg-blue-950/30",
    badge: "bg-blue-500/20 text-blue-300",
    items: [
      { label: "t-Copula", desc: "대칭적 꼬리 의존성 — λ = 2·t_{ν+1}(…)" },
      { label: "Clayton Copula", desc: "하방 꼬리 의존성 강화 — LONG 포지션에 보수적" },
      { label: "Vine Copula", desc: "고차원 pair-wise 의존성 트리" },
    ],
    current: "Ledoit-Wolf 선형 공분산 — 위기 시 상관 급등 미반영",
    note: "optimize_portfolio()에서 사용 중",
  },
  {
    id: "bridge",
    name: "Bridge — Semi-parametric Threshold",
    color: "border-emerald-500 bg-emerald-950/30",
    badge: "bg-emerald-500/20 text-emerald-300",
    items: [
      { label: "Mean Excess Function Plot", desc: "GPD 피팅이 안정적인 임계값 u 탐색" },
      { label: "Hill Plot", desc: "꼬리 지수 ξ의 안정 구간 확인" },
      { label: "F_body(u) = F_GPD(u)", desc: "봉합 지점 연속성 제약" },
    ],
    current: "gate_score >= 0.45 (수동 지정) — 데이터 기반 임계값 없음",
    note: "현재 GATE_THRESH 파라미터",
  },
  {
    id: "L2",
    name: "Layer 2 — Tail (EVT/GPD)",
    color: "border-orange-500 bg-orange-950/30",
    badge: "bg-orange-500/20 text-orange-300",
    items: [
      { label: "POT: Peaks Over Threshold", desc: "임계값 초과 사건만 GPD에 피팅" },
      { label: "GPD: F(z|u) = 1-(1+ξ(z-u)/β)^(-1/ξ)", desc: "꼬리 지수 ξ, 스케일 β 추정" },
      { label: "ES_EVT (CVaR 대체)", desc: "u + β/(1-ξ) + σ_t·꼬리평균" },
    ],
    current: "CVaR 최적화는 있지만 분포 가정이 암묵적 정규",
    note: "gate OPEN 구간에서만 활성화",
  },
  {
    id: "L1",
    name: "Layer 1 — Body (Moments)",
    color: "border-sky-500 bg-sky-950/30",
    badge: "bg-sky-500/20 text-sky-300",
    items: [
      { label: "LPM_n(MAR)", desc: "n=1: 기대 하방 손실 / n=2: 하방 분산 / n=3,4: 꼬리 가중" },
      { label: "Sortino, Omega, Kappa(n)", desc: "분모 모멘트 차수만 다른 같은 계열" },
      { label: "Cornish-Fisher / Gram-Charlier", desc: "왜도·첨도로 정규분포 VaR 보정" },
    ],
    current: "Sharpe (MAR=0 암묵적), Z-score 정규화",
    note: "gate CLOSED 구간 — 본체 체제",
  },
];

const EVENTS = [
  { type: "SUPPLY_SHOCK", crisis: "호르무즈", gate: "OPEN", xi: "0.2~0.5", copula: "Clayton", correct: true },
  { type: "STRUCTURAL_COLLAPSE", crisis: "GFC", gate: "OPEN", xi: "0.4~0.8", copula: "t-Copula", correct: true },
  { type: "SUBSECTOR_SPECIFIC", crisis: "COVID", gate: "CLOSED", xi: "0.1~0.3", copula: "Gaussian", correct: true },
  { type: "VALUATION_CORRECTION", crisis: "닷컴", gate: "CLOSED*", xi: "0.3~0.6", copula: "t-Copula", correct: false },
];

export default function ArchitecturePage() {
  return (
    <div className="space-y-10">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-100">GARCH–EVT–Copula Framework</h1>
        <p className="mt-2 text-gray-400 text-sm leading-relaxed max-w-3xl">
          5개 층이 어디서 만나고 갈라지는지를 명시적으로 설계한 구조.
          본체는 모멘트, 꼬리는 EVT, 의존성은 코퓰라, 동학은 GARCH, 영역 전환은 명시적 임계값.
        </p>
      </div>

      {/* Pipeline Flow */}
      <div className="flex flex-col gap-3">
        {LAYERS.map((layer, i) => (
          <div key={layer.id} className={`rounded-xl border ${layer.color} p-5`}>
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-3">
                  <span className={`text-xs font-mono px-2 py-0.5 rounded ${layer.badge}`}>
                    {layer.id}
                  </span>
                  <h3 className="text-sm font-semibold text-gray-100">{layer.name}</h3>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {layer.items.map((item) => (
                    <div key={item.label} className="bg-gray-900/50 rounded-lg p-3">
                      <div className="text-xs font-mono text-gray-200 mb-1">{item.label}</div>
                      <div className="text-xs text-gray-400">{item.desc}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="min-w-[200px] bg-gray-900/60 rounded-lg p-3 text-right">
                <div className="text-xs text-gray-500 mb-1">현재 구현</div>
                <div className="text-xs text-amber-400">{layer.current}</div>
                <div className="text-xs text-gray-600 mt-1">{layer.note}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Arrow */}
      <div className="flex items-center gap-3 text-gray-500 text-sm">
        <div className="flex-1 h-px bg-gray-800" />
        <span>Ontology routes event type → ConditionalDistribution → selects active layers</span>
        <div className="flex-1 h-px bg-gray-800" />
      </div>

      {/* Event Type Routing Table */}
      <div>
        <h2 className="text-base font-semibold text-gray-200 mb-4">이벤트 유형별 층 라우팅</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400 text-xs">
                <th className="text-left py-2 pr-4">Event Type</th>
                <th className="text-left py-2 pr-4">위기 사례</th>
                <th className="text-left py-2 pr-4">Gate</th>
                <th className="text-left py-2 pr-4">꼬리 지수 ξ</th>
                <th className="text-left py-2 pr-4">코퓰라</th>
                <th className="text-left py-2">게이트 판정</th>
              </tr>
            </thead>
            <tbody>
              {EVENTS.map((e) => (
                <tr key={e.type} className="border-b border-gray-800/50 hover:bg-gray-900/30">
                  <td className="py-3 pr-4">
                    <span className="font-mono text-xs text-emerald-400">{e.type}</span>
                  </td>
                  <td className="py-3 pr-4 text-gray-300">{e.crisis}</td>
                  <td className="py-3 pr-4">
                    <span className={`text-xs px-2 py-0.5 rounded font-mono ${
                      e.gate.startsWith("OPEN")
                        ? "bg-emerald-900/50 text-emerald-400"
                        : "bg-gray-800 text-gray-400"
                    }`}>
                      {e.gate}
                    </span>
                  </td>
                  <td className="py-3 pr-4 font-mono text-xs text-orange-300">{e.xi}</td>
                  <td className="py-3 pr-4 text-xs text-blue-300">{e.copula}</td>
                  <td className="py-3">
                    {e.correct ? (
                      <span className="text-xs text-emerald-400">✓ OK</span>
                    ) : (
                      <span className="text-xs text-red-400">✗ FAIL — ξ 만으로 STRUCTURAL 구분 불가</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Causal Discovery Section */}
      <div className="rounded-xl border border-purple-700/50 bg-purple-950/20 p-6 space-y-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">
                PCMCI+
              </span>
              <h3 className="text-sm font-semibold text-gray-100">Layer 6 — Causal Discovery</h3>
            </div>
            <p className="text-xs text-gray-400 max-w-2xl leading-relaxed">
              tigramite PCMCI+로 8개 매크로 변수 간 인과 관계 발견. MCI 검정이 X·Y의 과거 + 전체 변수 조건화 →
              허위 상관 제거. 2022-02 구조 변화점 기준 full/recent 두 기간 모두 유의한 링크 = structural,
              recent 기간에만 유의 = emerging.
            </p>
          </div>
          <div className="min-w-[180px] text-right bg-gray-900/60 rounded-lg p-3">
            <div className="text-xs text-gray-500 mb-1">2026-06-03 기준</div>
            <div className="font-mono text-sm text-purple-300">42개 링크</div>
            <div className="text-xs text-gray-500 mt-1">structural 5 / emerging 22 / historical 15</div>
            <div className="text-xs text-amber-400/80 mt-1">변화점: 2022-02 (Ukraine+Fed)</div>
          </div>
        </div>

        {/* Key Chain */}
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-xs text-gray-500 mb-2">핵심 인과 전파 체인 (총 7개월 선행)</div>
          <div className="flex items-center gap-1 flex-wrap text-xs font-mono">
            {[
              { node: "OIL", type: "causal" },
              { arrow: "+1m·stru" },
              { node: "CPI", type: "causal" },
              { arrow: "+3m·emer" },
              { node: "US2Y", type: "macro" },
              { arrow: "+1m·stru" },
              { node: "FED_FUNDS", type: "causal" },
              { arrow: "-3m·stru" },
              { node: "VIX", type: "macro" },
              { arrow: "→ 11 sectors" },
            ].map((item, i) =>
              "arrow" in item ? (
                <span key={i} className="text-gray-600 px-1">--{item.arrow}--&gt;</span>
              ) : (
                <span key={i} className={`px-2 py-0.5 rounded ${
                  item.type === "causal"
                    ? "bg-purple-900/60 text-purple-300"
                    : "bg-blue-900/60 text-blue-300"
                }`}>
                  {item.node}
                </span>
              )
            )}
          </div>
        </div>

        {/* Causal Links Table */}
        <div>
          <div className="text-xs text-gray-500 mb-2">TRANSMITS_TO 엣지 (structural + emerging)</div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-800 text-gray-500">
                  <th className="text-left py-1.5 pr-3">Source</th>
                  <th className="text-left py-1.5 pr-3">Target</th>
                  <th className="text-left py-1.5 pr-3">Lag</th>
                  <th className="text-left py-1.5 pr-3">Coef</th>
                  <th className="text-left py-1.5">Stability</th>
                </tr>
              </thead>
              <tbody>
                {CAUSAL_LINKS.map((l, i) => (
                  <tr key={i} className="border-b border-gray-800/30 hover:bg-gray-900/20">
                    <td className="py-1.5 pr-3 font-mono text-purple-300">{l.src}</td>
                    <td className="py-1.5 pr-3 font-mono text-blue-300">{l.dst}</td>
                    <td className="py-1.5 pr-3 text-gray-400">{l.lag}m</td>
                    <td className={`py-1.5 pr-3 font-mono ${l.dir === "+" ? "text-emerald-400" : "text-red-400"}`}>
                      {l.coef}
                    </td>
                    <td className="py-1.5">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${
                        l.stability === "structural"
                          ? "bg-emerald-900/40 text-emerald-400"
                          : "bg-amber-900/40 text-amber-400"
                      }`}>
                        {l.stability}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Sector Signal Summary */}
        <div>
          <div className="text-xs text-gray-500 mb-2">
            causal_chain_monitor 신호 — 인과 체인을 통해 VIX에 노출된 섹터 (상위 10개)
          </div>
          <div className="space-y-1.5">
            {CAUSAL_SECTORS.map((s) => (
              <div key={s.ticker} className="flex items-center gap-3 text-xs">
                <span className="font-mono text-purple-300 w-10">{s.ticker}</span>
                <span className="text-gray-500 w-28">{s.name}</span>
                <span className="text-gray-600 w-8">δ={s.delta.toFixed(3)}</span>
                <div className="flex-1 min-w-0">
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-purple-500"
                      style={{ width: `${s.conf * 100}%` }}
                    />
                  </div>
                </div>
                <span className="font-mono text-gray-400 w-10 text-right">{s.conf.toFixed(3)}</span>
                <span className="text-gray-600 hidden lg:block truncate max-w-[280px]">{s.chain}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Navigation Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4">
        <Link href="/ontology" className="group block rounded-xl border border-gray-800 bg-gray-900/50 p-6 hover:border-emerald-500/50 hover:bg-emerald-950/20 transition-all">
          <div className="text-emerald-400 text-xs font-mono mb-2">PAGE 2</div>
          <div className="text-gray-100 font-semibold mb-2">Ontology Trace →</div>
          <div className="text-gray-400 text-sm">
            4개 위기별 판단 경로 — beta/VRP/방향 컴포넌트가 게이트 점수를 어떻게 구성했는지 Step 1~4 추적
          </div>
        </Link>
        <Link href="/gate" className="group block rounded-xl border border-gray-800 bg-gray-900/50 p-6 hover:border-blue-500/50 hover:bg-blue-950/20 transition-all">
          <div className="text-blue-400 text-xs font-mono mb-2">PAGE 3</div>
          <div className="text-gray-100 font-semibold mb-2">Gate Timeline →</div>
          <div className="text-gray-400 text-sm">
            호르무즈(XLE) 게이트 점수 타임라인 + 4-전략 OOS 성과 비교
          </div>
        </Link>
      </div>
    </div>
  );
}
