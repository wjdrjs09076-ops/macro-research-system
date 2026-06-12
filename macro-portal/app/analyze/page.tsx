"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ─── Types ────────────────────────────────────────────────────────────────────

interface ClassifyResult {
  ticker: string;
  sector: string;
  direction: "LONG" | "SHORT";
  event_type: string;
  start: string;
  oos_start: string;
  end: string;
  year: number | null;
  confidence: "high" | "medium" | "low";
  can_proceed: boolean;
  name_score: number;
  context_score: number;
  context_year?: number | null;
  resolved_title?: string | null;
  warnings?: string[];
  news: { title: string; snippet: string; url: string }[];
}

interface HistoryEntry {
  query: string;
  normalized: string;
  ticker: string;
  direction: "LONG" | "SHORT";
  event_type: string;
  start: string;
  oos_start: string;
  confirmed: boolean;
  corrected: boolean;
  timestamp: string;
}

interface QuantResult {
  event: { name: string; ticker: string; direction: string; type: string; start: string; oos_start: string };
  gate: {
    params: { thresh: number; consec: number; weights: { beta: number; vrp: number; dir: number } };
    oos_open_pct: number;
    is_open_pct: number;
    timeseries: { date: string; gate_score: number; beta_comp: number; vrp_comp: number; dir_comp: number; gate_open: boolean }[];
  };
  performance: {
    all_oos:   { cagr: number; total_return: number; sharpe: number; mdd: number; n_days: number };
    gate_open: { cagr: number; total_return: number; sharpe: number; mdd: number; n_days: number };
    bnh_spy:   { cagr: number; total_return: number; sharpe: number; mdd: number; n_days: number };
  };
  evt: {
    mef:        { u: number; mef: number; n: number }[];
    hill:       { k: number; hill: number; u: number }[];
    stability:  { u: number; xi: number; beta: number; beta_star: number; n: number }[];
    gpd:        { xi: number; beta: number; u_star: number; u_pct: number; n_exceed: number; zeta_u: number } | null;
    risk:       { var99: number; es99: number } | null;
    tail_asymmetry: { xi_loss: number; xi_gain: number; asymmetry: number; interpretation: string } | null;
  };
  returns: { date: string; sector: number; spy?: number }[];
  straddle: {
    simulation: {
      long_pnl: number; short_pnl: number; combined_pnl: number;
      long_hit: boolean; short_hit: boolean;
      long_day: number; short_day: number;
      long_date: string; short_date: string;
      target: number; n_days: number;
      path: { date: string; ret: number }[];
    } | null;
    mc: {
      e_combined: number; e_long: number; e_short: number;
      p_long_hit: number; p_short_hit: number; p_both_hit: number;
      sigma_annual: number; drift_annual: number; n_sim: number;
    };
  };
}

// ─── localStorage helpers ─────────────────────────────────────────────────────

const LS_KEY = "macro_event_history";

function normalizeQuery(q: string) {
  return q.toLowerCase().replace(/\d{4}/g, "").replace(/\s+/g, " ").trim();
}

function loadHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(entries: HistoryEntry[]) {
  localStorage.setItem(LS_KEY, JSON.stringify(entries.slice(0, 100)));
}

function findInHistory(query: string): HistoryEntry | null {
  const norm  = normalizeQuery(query);
  const hist  = loadHistory();
  const confirmed = hist.filter(h => h.confirmed);
  // Match: normalized query shares ≥5 chars with stored normalized key
  for (const h of confirmed) {
    const minLen = Math.min(norm.length, h.normalized.length, 5);
    if (minLen >= 3 && (norm.includes(h.normalized.slice(0, minLen)) ||
                        h.normalized.includes(norm.slice(0, minLen)))) {
      return h;
    }
  }
  return null;
}

function upsertHistory(entry: HistoryEntry) {
  const hist     = loadHistory();
  const norm     = entry.normalized;
  const filtered = hist.filter(h => !h.normalized.startsWith(norm.slice(0, 5)));
  saveHistory([entry, ...filtered]);
}

// ─── UI primitives ────────────────────────────────────────────────────────────

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase tracking-wide">{label}</label>
      {children}
    </div>
  );
}

function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input {...props}
      className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100
                 focus:outline-none focus:border-emerald-500 transition-colors w-full" />
  );
}

type SelectOption = string | { group: string; options: string[] };

function Select({ options, ...props }: { options: SelectOption[] } & React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select {...props}
      className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100
                 focus:outline-none focus:border-emerald-500 transition-colors w-full">
      {options.map((o) =>
        typeof o === "string"
          ? <option key={o} value={o}>{o}</option>
          : <optgroup key={o.group} label={o.group}>
              {o.options.map(t => <option key={t} value={t}>{t}</option>)}
            </optgroup>
      )}
    </select>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-800/60 rounded-lg p-4 border border-gray-700/50">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-xl font-mono font-semibold text-gray-100">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function PerfCard({ label, m, color }: {
  label: string;
  m: { cagr: number; total_return: number; sharpe: number; mdd: number; n_days: number };
  color: string;
}) {
  const pct = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
  const years = m.n_days / 252;
  return (
    <div className={`rounded-lg p-4 border ${color}`}>
      <div className="text-xs font-semibold text-gray-300 mb-3">{label}</div>
      {/* 보유기간 수익률 강조 */}
      <div className="mb-3 pb-3 border-b border-gray-700/50">
        <div className="text-xs text-gray-500 mb-0.5">보유기간 수익률</div>
        <div className={`text-2xl font-bold font-mono ${m.total_return >= 0 ? "text-emerald-400" : "text-red-400"}`}>
          {pct(m.total_return)}
        </div>
        <div className="text-xs text-gray-500 mt-0.5">{m.n_days}일 ({years.toFixed(1)}년)</div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div><span className="text-gray-400">CAGR </span>
          <span className={m.cagr >= 0 ? "text-emerald-400" : "text-red-400"}>{pct(m.cagr)}</span></div>
        <div><span className="text-gray-400">Sharpe </span>
          <span className="text-gray-200">{m.sharpe.toFixed(2)}</span></div>
        <div><span className="text-gray-400">MDD </span>
          <span className="text-red-400">{pct(m.mdd)}</span></div>
        <div><span className="text-gray-400">연환산 </span>
          <span className="text-gray-500 text-xs">{years.toFixed(1)}yr 기준</span></div>
      </div>
    </div>
  );
}

// ─── ETF / direction / type options ──────────────────────────────────────────

const ETF_OPTIONS: SelectOption[] = [
  { group: "US Sectors", options: ["SOXX","XLE","XLF","XLK","XLV","XAR","XTN","XLY","VNQ","XLB","XLU","XLC","XLI"] },
  { group: "US Broad / Commodities", options: ["QQQ","SPY","SMH","IYW","GLD","TLT","USO"] },
  { group: "Asia Pacific", options: ["EWY","EWJ","FXI","MCHI","KWEB","INDA","EWA"] },
  { group: "Europe", options: ["EWG","EWU","EZU","EWQ"] },
  { group: "Emerging Markets / Global", options: ["EEM","EFA","EWZ"] },
];
const DIR_OPTIONS: ("LONG"|"SHORT")[] = ["SHORT", "LONG"];
const TYPE_OPTIONS = [
  "SUPPLY_SHOCK","VALUATION_CORRECTION","STRUCTURAL_COLLAPSE","SUBSECTOR_SPECIFIC","UNKNOWN",
];

// ─── Main page ────────────────────────────────────────────────────────────────

export default function AnalyzePage() {
  const [eventName, setEventName]       = useState("");
  const [ticker,    setTicker]          = useState("SOXX");
  const [direction, setDirection]       = useState<"LONG"|"SHORT">("SHORT");
  const [eventType, setEventType]       = useState("SUPPLY_SHOCK");
  const [startDate, setStartDate]       = useState("2019-01-01");
  const [oosDate,   setOosDate]         = useState("2019-07-01");

  const [classifying, setClassifying]   = useState(false);
  const [analyzing,   setAnalyzing]     = useState(false);
  const [classResult, setClassResult]   = useState<ClassifyResult | null>(null);
  const [quantResult, setQuantResult]   = useState<QuantResult | null>(null);
  const [error,       setError]         = useState<string | null>(null);
  const [historyHit,  setHistoryHit]    = useState<HistoryEntry | null>(null);

  // Feedback state
  const [feedbackState, setFeedbackState] = useState<"idle"|"shown"|"correcting"|"done">("idle");
  const [corrTicker,    setCorrTicker]    = useState("");
  const [corrDir,       setCorrDir]       = useState<"LONG"|"SHORT">("SHORT");

  // History panel
  const [showHistory, setShowHistory]   = useState(false);
  const [history,     setHistory]       = useState<HistoryEntry[]>([]);

  const [chartTab, setChartTab] = useState<"gate"|"returns"|"straddle"|"mef"|"hill"|"stability">("gate");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => { setHistory(loadHistory()); }, []);

  // ── Auto-classify ──
  const handleEventName = useCallback((val: string) => {
    setEventName(val);
    setClassResult(null);
    setQuantResult(null);
    setFeedbackState("idle");
    setHistoryHit(null);
    setError(null);

    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (val.trim().length < 4) return;

    debounceRef.current = setTimeout(async () => {
      // Check localStorage first
      const cached = findInHistory(val);
      if (cached) {
        setHistoryHit(cached);
        setTicker(cached.ticker);
        setDirection(cached.direction);
        setEventType(cached.event_type);
        setStartDate(cached.start);
        setOosDate(cached.oos_start);
        return;
      }

      setClassifying(true);
      try {
        const res  = await fetch("/api/classify", {
          method:  "POST",
          headers: { "Content-Type": "application/json; charset=utf-8" },
          body:    JSON.stringify({ event_name: val }),
        });
        const data: ClassifyResult = await res.json();
        setClassResult(data);
        if (data.can_proceed) {
          setTicker(data.ticker);
          setDirection(data.direction);
          setEventType(data.event_type);
          setStartDate(data.start);
          setOosDate(data.oos_start);
        }
      } catch {
        // silent — user can fill manually
      } finally {
        setClassifying(false);
      }
    }, 1000);
  }, []);

  // ── Run analysis ──
  const handleAnalyze = async () => {
    setAnalyzing(true);
    setQuantResult(null);
    setFeedbackState("idle");
    setError(null);
    try {
      const res = await fetch("/api/quantify", {
        method:  "POST",
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body:    JSON.stringify({
          event_name: eventName,
          ticker, direction, event_type: eventType,
          start: startDate, oos_start: oosDate,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setQuantResult(data as QuantResult);
      setChartTab("gate");
      setFeedbackState("shown");
      setCorrTicker(ticker);
      setCorrDir(direction);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAnalyzing(false);
    }
  };

  // ── Feedback handlers ──
  const handleFeedback = (confirmed: boolean) => {
    const entry: HistoryEntry = {
      query:      eventName,
      normalized: normalizeQuery(eventName),
      ticker, direction, event_type: eventType,
      start:    startDate,
      oos_start: oosDate,
      confirmed,
      corrected: false,
      timestamp: new Date().toISOString(),
    };
    upsertHistory(entry);
    setHistory(loadHistory());
    setFeedbackState(confirmed ? "done" : "correcting");
  };

  const handleCorrection = () => {
    const entry: HistoryEntry = {
      query:      eventName,
      normalized: normalizeQuery(eventName),
      ticker:    corrTicker,
      direction: corrDir,
      event_type: eventType,
      start:    startDate,
      oos_start: oosDate,
      confirmed: true,
      corrected: true,
      timestamp: new Date().toISOString(),
    };
    upsertHistory(entry);
    setHistory(loadHistory());
    setFeedbackState("done");
  };

  const loadFromHistory = (h: HistoryEntry) => {
    setEventName(h.query);
    setTicker(h.ticker);
    setDirection(h.direction);
    setEventType(h.event_type);
    setStartDate(h.start);
    setOosDate(h.oos_start);
    setHistoryHit(h);
    setClassResult(null);
    setQuantResult(null);
    setFeedbackState("idle");
    setShowHistory(false);
  };

  // ── Derived state ──
  const canProceed = historyHit
    ? true
    : classResult
      ? classResult.can_proceed
      : eventName.trim().length > 3; // allow if not classified yet (manual)

  const confidenceColor = { high: "text-emerald-400", medium: "text-yellow-400", low: "text-red-400" };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100 mb-1">이벤트 분석기</h1>
        <p className="text-gray-400 text-sm">
          시장 사건명을 입력하면 AI가 뉴스를 검색해 섹터·ETF·방향을 자동 감지합니다.
        </p>
        <p className="text-[11px] text-amber-400/80 mt-2 max-w-3xl leading-relaxed">
          ⚠ 방법론 한계: 이미 알려진 위기를 이름으로 입력하는 구조는 <b>사후 선택</b>입니다 —
          여기서 나온 게이트/성과 수치는 &ldquo;그 사건을 미리 알았다면&rdquo;의 조건부 결과이지
          전향적 예측력의 증거가 아닙니다. 전향 검증은 Vol Monitor 플래그 적중률 패널이 담당합니다.
        </p>
      </div>

      {/* ── History panel ── */}
      {history.filter(h => h.confirmed).length > 0 && (
        <div className="border border-gray-800 rounded-xl overflow-hidden">
          <button
            onClick={() => setShowHistory(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 bg-gray-900/50
                       text-sm text-gray-300 hover:text-gray-100 transition-colors"
          >
            <span className="font-medium">
              확인된 이벤트 히스토리 ({history.filter(h => h.confirmed).length}건)
            </span>
            <span className="text-gray-500">{showHistory ? "▲" : "▼"}</span>
          </button>
          {showHistory && (
            <div className="divide-y divide-gray-800 max-h-56 overflow-y-auto">
              {history.filter(h => h.confirmed).map((h, i) => (
                <button
                  key={i}
                  onClick={() => loadFromHistory(h)}
                  className="w-full text-left px-4 py-2.5 hover:bg-gray-800/50 transition-colors"
                >
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm text-gray-200 truncate max-w-xs">{h.query}</span>
                    <span className="text-xs font-mono bg-gray-800 px-1.5 py-0.5 rounded text-gray-300">
                      {h.ticker}
                    </span>
                    <span className={`text-xs font-semibold ${h.direction === "LONG" ? "text-emerald-400" : "text-red-400"}`}>
                      {h.direction}
                    </span>
                    {h.corrected && (
                      <span className="text-xs text-yellow-500">수정됨</span>
                    )}
                    <span className="text-xs text-gray-600 ml-auto">
                      {h.timestamp.slice(0, 10)}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Input form ── */}
      <div className="bg-gray-900/70 border border-gray-800 rounded-xl p-6 space-y-5">
        <Field label="이벤트명 (한글/영문 모두 지원)">
          <div className="relative">
            <Input
              value={eventName}
              onChange={e => handleEventName(e.target.value)}
              placeholder="예: 수에즈 운하 에버기븐호 좌초 사고 2021"
            />
            {classifying && (
              <span className="absolute right-3 top-2 text-xs text-emerald-400 animate-pulse">
                뉴스 서치 중…
              </span>
            )}
          </div>
        </Field>

        {/* History hit badge */}
        {historyHit && (
          <div className="flex items-center gap-2 px-3 py-2 bg-emerald-900/30 border border-emerald-700/40 rounded-lg text-xs text-emerald-400">
            <span>★</span>
            <span>히스토리에서 불러옴 — {historyHit.confirmed && !historyHit.corrected ? "사용자 확인됨" : "사용자 수정됨"}</span>
          </div>
        )}

        {/* Classification result / warning */}
        {!historyHit && classResult && (
          <div className={`border rounded-lg p-4 space-y-3 ${
            classResult.can_proceed
              ? "border-gray-700 bg-gray-800/40"
              : "border-red-700/50 bg-red-900/20"
          }`}>
            {!classResult.can_proceed ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-red-400 font-semibold text-sm">
                  <span>⚠</span>
                  <span>이벤트를 분류할 수 없습니다</span>
                </div>
                <p className="text-xs text-red-300/80">
                  &ldquo;{eventName}&rdquo;에 대한 금융 섹터 신호를 찾지 못했습니다.
                  이벤트명을 더 구체적으로 입력하거나 영문 키워드를 포함해 주세요.
                </p>
                <p className="text-xs text-gray-500">
                  예시: &ldquo;수에즈 운하 선박 좌초 2021&rdquo;, &ldquo;Japan semiconductor export restriction 2019&rdquo;
                </p>
              </div>
            ) : (
              <div className="flex items-center gap-3 flex-wrap">
                <span className="text-xs text-gray-400">AI 분류</span>
                <span className={`text-xs font-semibold ${confidenceColor[classResult.confidence]}`}>
                  신뢰도: {classResult.confidence.toUpperCase()}
                </span>
                <span className="text-xs text-gray-500">
                  이벤트명 신호 {classResult.name_score}
                  {classResult.context_score > 0 && ` + 뉴스 컨텍스트 ${classResult.context_score}`}
                </span>
                <span className="text-xs text-gray-400">
                  섹터: <span className="text-gray-200">{classResult.sector}</span>
                </span>
              </div>
            )}

            {classResult.can_proceed && (classResult.warnings?.length ?? 0) > 0 && (
              <div className="space-y-1.5 border-l-2 border-amber-500/60 pl-3">
                {classResult.warnings!.map((w, i) => (
                  <div key={i} className="text-xs text-amber-300 flex gap-1.5">
                    <span>⚠</span>
                    <span>{w}</span>
                  </div>
                ))}
              </div>
            )}

            {classResult.can_proceed && classResult.resolved_title && (
              <div className="text-[11px] text-gray-500">
                위키 해석: <span className="text-gray-400">{classResult.resolved_title}</span>
                {classResult.context_year && (
                  <span> · 컨텍스트 연도 <span className="text-gray-400">{classResult.context_year}</span></span>
                )}
              </div>
            )}

            {classResult.can_proceed && classResult.news.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs text-gray-500 uppercase tracking-wide">검색된 뉴스 컨텍스트</div>
                {classResult.news.slice(0, 3).map((n, i) => (
                  <div key={i} className="border-l-2 border-gray-600 pl-3">
                    <div className="text-xs text-emerald-400 font-medium truncate">{n.title}</div>
                    <div className="text-xs text-gray-400 mt-0.5 line-clamp-2">{n.snippet}</div>
                  </div>
                ))}
              </div>
            )}

            {classResult.can_proceed && classResult.news.length === 0 && (
              <div className="text-xs text-gray-500 italic">뉴스 컨텍스트 없음 — LLM 이벤트명 단독 분류</div>
            )}
          </div>
        )}

        {/* Editable params */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <Field label="ETF Ticker">
            <Select options={ETF_OPTIONS} value={ticker} onChange={e => setTicker(e.target.value)} />
          </Field>
          <Field label="이벤트 성향 (게이트 방향)">
            <Select options={DIR_OPTIONS} value={direction}
              onChange={e => setDirection(e.target.value as "LONG"|"SHORT")} />
          </Field>
          <Field label="이벤트 유형">
            <Select options={TYPE_OPTIONS} value={eventType} onChange={e => setEventType(e.target.value)} />
          </Field>
          <Field label="IS 시작">
            <Input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </Field>
          <Field label="OOS 시작 (이벤트 발생)">
            <Input type="date" value={oosDate} onChange={e => setOosDate(e.target.value)} />
          </Field>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={analyzing || !eventName.trim() || !canProceed}
          className="w-full py-2.5 rounded-lg font-semibold text-sm transition-all
                     bg-emerald-600 hover:bg-emerald-500 text-white
                     disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed"
        >
          {analyzing
            ? "분석 중… (15–30초)"
            : !canProceed
              ? "⚠ 이벤트명을 다시 입력해 주세요"
              : "분석 시작 →"}
        </button>

        {error && (
          <div className="bg-red-900/30 border border-red-700/50 rounded-lg p-3 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

      {/* ── Results ── */}
      {quantResult && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-lg font-bold text-gray-100">{quantResult.event.name}</h2>
            <span className="px-2 py-0.5 rounded bg-gray-800 text-xs text-gray-300 font-mono">
              {quantResult.event.ticker}
            </span>
            <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
              quantResult.event.direction === "LONG"
                ? "bg-emerald-900/40 text-emerald-400"
                : "bg-red-900/40 text-red-400"
            }`}>{quantResult.event.direction} 성향</span>
            <span className="text-xs text-gray-600">스트래들 양방향 진입</span>
            <span className="px-2 py-0.5 rounded bg-blue-900/30 text-xs text-blue-300">
              {quantResult.event.type}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="OOS 게이트 개방률" value={`${quantResult.gate.oos_open_pct}%`} sub="임계값 0.45" />
            <Stat label="IS 게이트 개방률"  value={`${quantResult.gate.is_open_pct}%`}  sub="정상 구간" />
            {quantResult.evt.gpd && (
              <Stat label="EVT 임계값 u*"
                value={quantResult.evt.gpd.u_star.toFixed(4)}
                sub={`P${quantResult.evt.gpd.u_pct.toFixed(0)} | ξ=${quantResult.evt.gpd.xi.toFixed(3)}`}
              />
            )}
            {quantResult.evt.tail_asymmetry && (
              <Stat label="꼬리 비대칭"
                value={`${quantResult.evt.tail_asymmetry.asymmetry >= 0 ? "+" : ""}${quantResult.evt.tail_asymmetry.asymmetry.toFixed(3)}`}
                sub={quantResult.evt.tail_asymmetry.interpretation}
              />
            )}
          </div>

          <div>
            <h3 className="text-sm font-semibold text-gray-300 mb-3">성과 비교 (OOS 기간)</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <PerfCard label="전체 OOS 보유"  m={quantResult.performance.all_oos}   color="border-gray-700" />
              <PerfCard label="게이트 개방 시만" m={quantResult.performance.gate_open} color="border-emerald-700/50" />
              <PerfCard label="SPY 벤치마크"   m={quantResult.performance.bnh_spy}   color="border-blue-700/50" />
            </div>
          </div>

          {/* ── Straddle Section ── */}
          {quantResult.straddle?.simulation && (() => {
            const sim = quantResult.straddle.simulation;
            const mc  = quantResult.straddle.mc;
            const pct = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
            const hitBadge = (hit: boolean, day: number) => hit
              ? <span className="text-xs bg-emerald-900/40 text-emerald-400 px-1.5 py-0.5 rounded font-semibold">HIT ✓ {day}일</span>
              : <span className="text-xs bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">MISS — MTM</span>;
            return (
              <div className="rounded-xl border border-gray-700 bg-gray-900/40 p-5 space-y-4">
                <div className="flex items-center gap-3">
                  <h3 className="text-sm font-semibold text-gray-200">Straddle 시뮬레이션</h3>
                  <span className="text-xs text-gray-500">±{(sim.target * 100).toFixed(0)}% 목표 · OOS 진입일 기준 · {sim.n_days}일 보유</span>
                </div>

                {/* 3 leg cards */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="rounded-lg border border-emerald-700/40 bg-emerald-900/10 p-4 text-center">
                    <div className="text-xs text-gray-400 mb-1">Combined PnL</div>
                    <div className={`text-3xl font-bold font-mono ${sim.combined_pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {pct(sim.combined_pnl)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">2 leg 평균</div>
                  </div>
                  <div className="rounded-lg border border-gray-700 bg-gray-800/40 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Long 레그</span>
                      {hitBadge(sim.long_hit, sim.long_day)}
                    </div>
                    <div className={`text-2xl font-bold font-mono ${sim.long_pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {pct(sim.long_pnl)}
                    </div>
                    <div className="text-xs text-gray-600 mt-1">{sim.long_date}</div>
                  </div>
                  <div className="rounded-lg border border-gray-700 bg-gray-800/40 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Short 레그</span>
                      {hitBadge(sim.short_hit, sim.short_day)}
                    </div>
                    <div className={`text-2xl font-bold font-mono ${sim.short_pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {pct(sim.short_pnl)}
                    </div>
                    <div className="text-xs text-gray-600 mt-1">{sim.short_date}</div>
                  </div>
                </div>

                {/* MC stats */}
                <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
                  <div className="text-xs text-gray-400 mb-3 uppercase tracking-wide">
                    몬테카를로 사전 확률 — σ={`${(mc.sigma_annual * 100).toFixed(1)}%`} 연율 기준 (N={mc.n_sim.toLocaleString()})
                  </div>
                  <div className="grid grid-cols-3 md:grid-cols-6 gap-3 text-center">
                    {[
                      { label: "E[Combined]", value: pct(mc.e_combined), accent: mc.e_combined >= 0 ? "text-emerald-400" : "text-red-400" },
                      { label: "E[Long]",     value: pct(mc.e_long),     accent: mc.e_long >= 0 ? "text-emerald-300" : "text-red-300" },
                      { label: "E[Short]",    value: pct(mc.e_short),    accent: mc.e_short >= 0 ? "text-emerald-300" : "text-red-300" },
                      { label: "P(Long HIT)", value: `${(mc.p_long_hit * 100).toFixed(0)}%`,  accent: "text-gray-200" },
                      { label: "P(Short HIT)",value: `${(mc.p_short_hit * 100).toFixed(0)}%`, accent: "text-gray-200" },
                      { label: "P(둘 다 HIT)",value: `${(mc.p_both_hit * 100).toFixed(0)}%`,  accent: mc.p_both_hit >= 0.2 ? "text-emerald-400" : "text-yellow-400" },
                    ].map(({ label, value, accent }) => (
                      <div key={label} className="bg-gray-800/60 rounded p-2">
                        <div className="text-xs text-gray-500 mb-0.5">{label}</div>
                        <div className={`text-sm font-mono font-semibold ${accent}`}>{value}</div>
                      </div>
                    ))}
                  </div>
                  <div className="text-xs text-gray-600 mt-2">
                    * 사전 변동성: IS 기간 마지막 20일 실현 변동성 기준 · 드리프트 {pct(mc.drift_annual)} 반영
                  </div>
                </div>
              </div>
            );
          })()}

          <div>
            <div className="flex gap-1 mb-4 flex-wrap">
              {(["gate","returns","straddle","mef","hill","stability"] as const).map(tab => (
                <button key={tab} onClick={() => setChartTab(tab)}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    chartTab === tab ? "bg-emerald-700 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
                  }`}>
                  {tab === "gate" ? "게이트 스코어" : tab === "returns" ? "누적 수익률" :
                   tab === "straddle" ? "스트래들 경로" :
                   tab === "mef"  ? "MEF"          : tab === "hill"    ? "Hill 추정"  : "안정성"}
                </button>
              ))}
            </div>
            <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-4">
              {chartTab === "gate"      && <GateChart data={quantResult.gate.timeseries} oos_start={quantResult.event.oos_start} thresh={quantResult.gate.params.thresh} />}
              {chartTab === "returns"   && <ReturnsChart data={quantResult.returns} />}
              {chartTab === "straddle"  && quantResult.straddle?.simulation && (
                <StraddleChart
                  path={quantResult.straddle.simulation.path}
                  target={quantResult.straddle.simulation.target}
                  long_hit={quantResult.straddle.simulation.long_hit}
                  short_hit={quantResult.straddle.simulation.short_hit}
                  long_date={quantResult.straddle.simulation.long_date}
                  short_date={quantResult.straddle.simulation.short_date}
                />
              )}
              {chartTab === "mef"       && <MefChart data={quantResult.evt.mef} u_star={quantResult.evt.gpd?.u_star} />}
              {chartTab === "hill"      && <HillChart data={quantResult.evt.hill} />}
              {chartTab === "stability" && <StabilityChart data={quantResult.evt.stability} u_star={quantResult.evt.gpd?.u_star} />}
            </div>
          </div>

          {quantResult.evt.tail_asymmetry && (
            <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-semibold text-gray-300">EVT 해석 — 게이트 신뢰도</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div><div className="text-xs text-gray-400 mb-1">손실 꼬리 ξ</div>
                  <div className="font-mono text-red-400">{quantResult.evt.tail_asymmetry.xi_loss.toFixed(4)}</div></div>
                <div><div className="text-xs text-gray-400 mb-1">수익 꼬리 ξ</div>
                  <div className="font-mono text-emerald-400">{quantResult.evt.tail_asymmetry.xi_gain.toFixed(4)}</div></div>
                <div><div className="text-xs text-gray-400 mb-1">비대칭 (손실−수익)</div>
                  <div className={`font-mono font-semibold ${quantResult.evt.tail_asymmetry.asymmetry > 0.1 ? "text-emerald-400" : "text-yellow-400"}`}>
                    {quantResult.evt.tail_asymmetry.asymmetry >= 0 ? "+" : ""}
                    {quantResult.evt.tail_asymmetry.asymmetry.toFixed(4)}
                  </div></div>
                <div><div className="text-xs text-gray-400 mb-1">게이트 신뢰도</div>
                  <div className={`text-sm font-semibold ${
                    quantResult.evt.tail_asymmetry.asymmetry > 0.2 ? "text-emerald-400" :
                    quantResult.evt.tail_asymmetry.asymmetry > 0.05 ? "text-yellow-400" : "text-red-400"
                  }`}>
                    {quantResult.evt.tail_asymmetry.asymmetry > 0.2 ? "HIGH ✓" :
                     quantResult.evt.tail_asymmetry.asymmetry > 0.05 ? "MEDIUM △" : "LOW ✗"}
                  </div></div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Feedback banner (fixed bottom-right) ── */}
      {feedbackState === "shown" && quantResult && (
        <div className="fixed bottom-5 right-5 z-50 w-72 bg-gray-850 border border-gray-700
                        rounded-xl shadow-2xl p-4 space-y-3 bg-gray-900">
          <div className="text-xs text-gray-400 uppercase tracking-wide">분류 검증</div>
          <p className="text-sm text-gray-200">
            <span className="font-mono font-semibold text-emerald-400">{quantResult.event.ticker}</span>
            {" / "}
            <span className={`font-semibold ${quantResult.event.direction === "LONG" ? "text-emerald-400" : "text-red-400"}`}>
              {quantResult.event.direction}
            </span>
            {" — 이 분류가 맞나요?"}
          </p>
          <div className="flex gap-2">
            <button onClick={() => handleFeedback(true)}
              className="flex-1 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-white rounded text-sm font-medium transition-colors">
              ✓ 맞아요
            </button>
            <button onClick={() => setFeedbackState("correcting")}
              className="flex-1 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm font-medium transition-colors">
              ✗ 수정
            </button>
            <button onClick={() => setFeedbackState("done")}
              className="px-2 text-gray-500 hover:text-gray-300 text-sm transition-colors">
              ✕
            </button>
          </div>
        </div>
      )}

      {/* ── Correction modal ── */}
      {feedbackState === "correcting" && (
        <div className="fixed bottom-5 right-5 z-50 w-80 bg-gray-900 border border-yellow-700/50
                        rounded-xl shadow-2xl p-4 space-y-3">
          <div className="text-xs text-yellow-400 font-semibold uppercase tracking-wide">
            올바른 분류 입력
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Field label="ETF">
              <Select options={ETF_OPTIONS} value={corrTicker} onChange={e => setCorrTicker(e.target.value)} />
            </Field>
            <Field label="방향">
              <Select options={DIR_OPTIONS} value={corrDir}
                onChange={e => setCorrDir(e.target.value as "LONG"|"SHORT")} />
            </Field>
          </div>
          <div className="flex gap-2">
            <button onClick={handleCorrection}
              className="flex-1 py-1.5 bg-yellow-700 hover:bg-yellow-600 text-white rounded text-sm font-medium transition-colors">
              저장하기
            </button>
            <button onClick={() => setFeedbackState("done")}
              className="px-3 text-gray-500 hover:text-gray-300 text-sm transition-colors">
              취소
            </button>
          </div>
        </div>
      )}

      {/* Done badge */}
      {feedbackState === "done" && (
        <div className="fixed bottom-5 right-5 z-50 bg-gray-900 border border-gray-700
                        rounded-xl shadow px-4 py-2.5 text-xs text-gray-300 flex items-center gap-2">
          <span className="text-emerald-400">✓</span>
          히스토리에 저장됨 — 다음 검색 시 자동 적용
        </div>
      )}
    </div>
  );
}

// ─── Chart components ─────────────────────────────────────────────────────────

function GateChart({ data, oos_start, thresh }: {
  data: QuantResult["gate"]["timeseries"]; oos_start: string; thresh: number;
}) {
  const slim = data.filter((_, i) => i % Math.max(1, Math.floor(data.length / 400)) === 0);
  return (
    <div>
      <div className="text-xs text-gray-400 mb-3">게이트 스코어 타임라인 — 임계값 {thresh}</div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={slim} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#9ca3af" }}
            tickFormatter={v => v.slice(0, 7)} interval="preserveStartEnd" />
          <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "#9ca3af" }} />
          <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 6 }}
            formatter={(v, n) => [typeof v === "number" ? v.toFixed(4) : v, n]} />
          <ReferenceLine y={thresh}     stroke="#ef4444" strokeDasharray="4 2" />
          <ReferenceLine x={oos_start}  stroke="#f59e0b" strokeDasharray="4 2"
            label={{ value: "OOS", fill: "#f59e0b", fontSize: 10 }} />
          <Line type="monotone" dataKey="gate_score" stroke="#10b981" strokeWidth={1.5} dot={false} name="게이트" />
          <Line type="monotone" dataKey="beta_comp"  stroke="#60a5fa" strokeWidth={1}   dot={false} name="Beta" />
          <Line type="monotone" dataKey="vrp_comp"   stroke="#a78bfa" strokeWidth={1}   dot={false} name="VRP" />
          <Line type="monotone" dataKey="dir_comp"   stroke="#fb923c" strokeWidth={1}   dot={false} name="방향" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ReturnsChart({ data }: { data: QuantResult["returns"] }) {
  const slim = data.filter((_, i) => i % Math.max(1, Math.floor(data.length / 400)) === 0);
  return (
    <div>
      <div className="text-xs text-gray-400 mb-3">OOS 누적 수익률</div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={slim} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#9ca3af" }}
            tickFormatter={v => v.slice(0, 7)} interval="preserveStartEnd" />
          <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: "#9ca3af" }} />
          <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 6 }}
            formatter={(v, n) => [typeof v === "number" ? `${(v * 100).toFixed(2)}%` : v, n]} />
          <ReferenceLine y={0} stroke="#6b7280" />
          <Line type="monotone" dataKey="sector" stroke="#10b981" strokeWidth={1.5} dot={false} name="섹터 ETF" />
          <Line type="monotone" dataKey="spy"    stroke="#60a5fa" strokeWidth={1.5} dot={false} name="SPY" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function MefChart({ data, u_star }: { data: { u: number; mef: number }[]; u_star?: number }) {
  return (
    <div>
      <div className="text-xs text-gray-400 mb-3">MEF — 평균 초과 함수</div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="u" tick={{ fontSize: 10, fill: "#9ca3af" }} tickFormatter={v => v.toFixed(3)} />
          <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} />
          <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 6 }}
            formatter={(v, n) => [typeof v === "number" ? v.toFixed(5) : v, n]} />
          {u_star !== undefined && <ReferenceLine x={u_star} stroke="#f59e0b" strokeDasharray="4 2"
            label={{ value: "u*", fill: "#f59e0b", fontSize: 10 }} />}
          <Line type="monotone" dataKey="mef" stroke="#34d399" strokeWidth={1.5} dot={false} name="e(u)" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function HillChart({ data }: { data: { k: number; hill: number }[] }) {
  return (
    <div>
      <div className="text-xs text-gray-400 mb-3">Hill 추정량 — 꼬리 지수 ξ</div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="k" tick={{ fontSize: 10, fill: "#9ca3af" }} />
          <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} />
          <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 6 }}
            formatter={(v, n) => [typeof v === "number" ? v.toFixed(5) : v, n]} />
          <Line type="monotone" dataKey="hill" stroke="#a78bfa" strokeWidth={1.5} dot={false} name="H_k" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function StraddleChart({ path, target, long_hit, short_hit, long_date, short_date }: {
  path: { date: string; ret: number }[];
  target: number; long_hit: boolean; short_hit: boolean;
  long_date: string; short_date: string;
}) {
  const shortTarget = -(1 - 1 / (1 + target)); // p0/p-1=target → ret = -target/(1+target)
  return (
    <div>
      <div className="text-xs text-gray-400 mb-1">스트래들 가격 경로 — OOS 진입 기준 누적 수익률</div>
      <div className="flex gap-4 text-xs text-gray-500 mb-3">
        <span className="text-emerald-400/80">— Long 목표 +{(target*100).toFixed(0)}%{long_hit ? ` (${long_date} HIT)` : " (미달성)"}</span>
        <span className="text-red-400/80">— Short 목표 {(shortTarget*100).toFixed(1)}%{short_hit ? ` (${short_date} HIT)` : " (미달성)"}</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={path} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#9ca3af" }}
            tickFormatter={v => v.slice(0, 7)} interval="preserveStartEnd" />
          <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: "#9ca3af" }} />
          <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 6 }}
            formatter={(v, n) => [typeof v === "number" ? `${(v * 100).toFixed(2)}%` : v, n]} />
          <ReferenceLine y={0}            stroke="#6b7280" strokeDasharray="2 2" />
          <ReferenceLine y={target}       stroke="#10b981" strokeDasharray="5 3"
            label={{ value: `Long +${(target*100).toFixed(0)}%`, fill: "#10b981", fontSize: 9, position: "right" }} />
          <ReferenceLine y={shortTarget}  stroke="#ef4444" strokeDasharray="5 3"
            label={{ value: `Short ${(shortTarget*100).toFixed(1)}%`, fill: "#ef4444", fontSize: 9, position: "right" }} />
          {long_hit  && <ReferenceLine x={long_date}  stroke="#10b981" strokeDasharray="3 3"
            label={{ value: "L-HIT", fill: "#10b981", fontSize: 9 }} />}
          {short_hit && <ReferenceLine x={short_date} stroke="#ef4444" strokeDasharray="3 3"
            label={{ value: "S-HIT", fill: "#ef4444", fontSize: 9 }} />}
          <Line type="monotone" dataKey="ret" stroke="#60a5fa" strokeWidth={1.5} dot={false} name="가격 경로" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function StabilityChart({ data, u_star }: { data: { u: number; xi: number; beta_star: number }[]; u_star?: number }) {
  return (
    <div>
      <div className="text-xs text-gray-400 mb-3">파라미터 안정성 — ξ(u), β*(u)</div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="u" tick={{ fontSize: 10, fill: "#9ca3af" }} tickFormatter={v => v.toFixed(3)} />
          <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} />
          <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 6 }}
            formatter={(v, n) => [typeof v === "number" ? v.toFixed(5) : v, n]} />
          {u_star !== undefined && <ReferenceLine x={u_star} stroke="#f59e0b" strokeDasharray="4 2"
            label={{ value: "u*", fill: "#f59e0b", fontSize: 10 }} />}
          <Line type="monotone" dataKey="xi"        stroke="#60a5fa" strokeWidth={1.5} dot={false} name="ξ(u)" />
          <Line type="monotone" dataKey="beta_star" stroke="#fb923c" strokeWidth={1.5} dot={false} name="β*(u)" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
