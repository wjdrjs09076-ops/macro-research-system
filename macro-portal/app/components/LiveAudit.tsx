"use client";

import { useEffect, useState } from "react";

// 감사 P0-1: 홈의 민감도 감사표를 하드코딩 스냅샷에서 라이브(단일 진실원)로 전환.
// 데이터 = ontology_signals.json 의 sensitivity_audit (trigger 가 매 사이클 생성,
// 직교화 lvl+2s10s + VIF + BH FDR q<0.10 기준). /ontology 페이지와 동일 소스.

type AuditRow = {
  sector: string;
  macro: string;
  delta_raw: number | null;
  t_raw: number | null;
  delta_ctrl: number | null;
  t_ctrl: number | null;
  q_ctrl?: number | null;
  vif?: number | null;
  verdict: "confirmed" | "killed" | "emerged" | "flipped";
};

const VERDICT_CLS: Record<string, string> = {
  killed: "bg-red-900/50 text-red-300",
  flipped: "bg-red-900/70 text-red-200",
  emerged: "bg-blue-900/50 text-blue-300",
  confirmed: "bg-emerald-900/50 text-emerald-300",
};

export default function LiveAudit() {
  const [rows, setRows] = useState<AuditRow[] | null>(null);
  const [generated, setGenerated] = useState<string>("");

  useEffect(() => {
    fetch("/data/ontology_signals.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.sensitivity_audit) {
          setRows(d.sensitivity_audit);
          setGenerated(d.generated ?? "");
        }
      })
      .catch(() => setRows(null));
  }, []);

  if (!rows) {
    return <div className="text-xs text-gray-600 italic">감사 데이터 로드 중…</div>;
  }

  const nKilled = rows.filter((r) => r.verdict === "killed" || r.verdict === "flipped").length;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs text-gray-500">
          민감도 감사 (라이브 — 단일 진실원) · 직교화(lvl+2s10s)+VIF+BH FDR q&lt;0.10 · {generated.slice(0, 16)}
        </div>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-900/40 text-red-300">
          교란/노이즈 판정 {nKilled}건
        </span>
      </div>
      <p className="text-[10px] text-gray-600 mb-3 max-w-3xl leading-relaxed">
        이 표는 매 사이클 자동 갱신되는 rule_sector_state 의 근거 데이터와 동일 소스입니다.
        KILLED = 통제 후 FDR 탈락 (룰 발화·포지션 보유 모두 차단). 문서·실행기·이 표가 같은 verdict 를 출력합니다.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 text-[10px]">
              <th className="text-left py-1.5 pr-3">Sector × Macro</th>
              <th className="text-right py-1.5 pr-3">δ raw (t)</th>
              <th className="text-right py-1.5 pr-3">δ ctrl (t)</th>
              <th className="text-right py-1.5 pr-3">q</th>
              <th className="text-right py-1.5 pr-3">VIF</th>
              <th className="text-left py-1.5">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.sector}-${r.macro}`} className="border-b border-gray-800/30 hover:bg-gray-900/20">
                <td className="py-1.5 pr-3 font-mono text-purple-300">{r.sector} × {r.macro}</td>
                <td className="py-1.5 pr-3 text-right font-mono text-gray-400">
                  {r.delta_raw?.toFixed(4) ?? "—"} <span className="text-gray-600">({r.t_raw?.toFixed(1) ?? "—"})</span>
                </td>
                <td className="py-1.5 pr-3 text-right font-mono text-gray-300">
                  {r.delta_ctrl?.toFixed(4) ?? "—"} <span className="text-gray-600">({r.t_ctrl?.toFixed(1) ?? "—"})</span>
                </td>
                <td className="py-1.5 pr-3 text-right font-mono text-gray-500">{r.q_ctrl?.toFixed(3) ?? "—"}</td>
                <td className="py-1.5 pr-3 text-right font-mono text-gray-500">{r.vif?.toFixed(1) ?? "—"}</td>
                <td className="py-1.5">
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${VERDICT_CLS[r.verdict]}`}>
                    {r.verdict.toUpperCase()}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
