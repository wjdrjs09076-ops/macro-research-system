# -*- coding: utf-8 -*-
"""
classify.py — LLM-powered event classification
Groq API (Llama 3.3 70B) + DuckDuckGo / Wikipedia / Guardian 컨텍스트
Guardian 뉴스를 LLM 호출 전에 수집해 분류 정확도를 높임
"""

from http.server import BaseHTTPRequestHandler
import json, re, os, urllib.request, urllib.parse, urllib.error

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

ETFS = (
    # US Sectors
    "SOXX (Semiconductors), XLE (Energy/Oil), XLF (Financials/Banking), "
    "XLK (Technology/Software), XLV (Healthcare/Biotech), XAR (Defense/Aerospace), "
    "XTN (Transportation/Shipping/Logistics), XLY (Consumer Discretionary), "
    "VNQ (Real Estate/REIT), XLB (Materials/Steel/Chemicals), XLU (Utilities), "
    "XLC (Communication Services), XLI (Industrials/Manufacturing), "
    "QQQ (Nasdaq-100), SMH (Semiconductors alt), GLD (Gold), "
    "TLT (Long-term Treasury), USO (Crude Oil), SPY (S&P 500 broad), "
    # Asia Pacific
    "EWY (South Korea/KOSPI), EWJ (Japan/Nikkei), "
    "FXI (China Large-Cap/CSI), MCHI (China Broad/MSCI), KWEB (China Internet/Tech), "
    "INDA (India/Nifty), EWA (Australia), "
    # Europe
    "EWG (Germany/DAX), EWU (UK/FTSE 100), EZU (Eurozone), EWQ (France/CAC 40), "
    # Emerging Markets / Global
    "EEM (Emerging Markets Broad), EFA (Developed ex-US/EAFE), EWZ (Brazil)"
)

SYSTEM = f"""You are a quantitative financial analyst for an ETF event-driven strategy system.

Given a market event name (any language) and optional news context, return ONLY a JSON object:

{{
  "ticker":     "best matching ETF ticker from the allowed list",
  "sector":     "sector name in English",
  "direction":  "LONG or SHORT",
  "event_type": "SUPPLY_SHOCK | VALUATION_CORRECTION | STRUCTURAL_COLLAPSE | SUBSECTOR_SPECIFIC",
  "start":      "YYYY-MM-DD — IS period start (18-24 months before oos_start)",
  "oos_start":  "YYYY-MM-DD — date the crisis actually began impacting markets",
  "end":        "YYYY-MM-DD — 12-18 months after the main crisis peak",
  "mechanism":  "one sentence causal chain in English",
  "confidence": "high | medium | low",
  "can_proceed": true | false
}}

Allowed ETFs: {ETFS}

Decision rules — US sectors:
- LONG  = event causes the sector ETF to RISE  (e.g. oil supply disruption → XLE LONG)
- SHORT = event causes the sector ETF to FALL  (e.g. financial crisis → XLF SHORT)
- Shipping / canal blockage / port disruption  → XTN LONG
- Semiconductor supply shock / export ban      → SOXX SHORT
- Oil / gas supply disruption                  → XLE LONG
- Financial contagion / banking collapse       → XLF SHORT
- Pandemic affecting broad healthcare          → XLV ambiguous → SUBSECTOR_SPECIFIC
- Defense conflicts / geopolitical tension     → XAR LONG

Decision rules — International:
- Korea geopolitical tension (North Korea, KOSPI shock)  → EWY SHORT
- Japan market crash / BOJ policy shock / Nikkei crisis  → EWJ SHORT
- China regulatory crackdown / tech ban / Alibaba-style  → KWEB SHORT
- China broad macro slowdown / property crisis           → FXI SHORT
- Germany manufacturing recession / energy crisis        → EWG SHORT
- UK political shock / Brexit uncertainty                → EWU SHORT
- Eurozone sovereign debt / ECB policy shock             → EZU SHORT
- India election shock / RBI surprise                    → INDA SHORT or LONG
- Broad emerging market stress / capital outflows        → EEM SHORT
- Global developed market contagion                      → EFA SHORT
- If event is clearly about a specific non-US market, prefer the matching regional ETF over SPY

General:
- can_proceed = false ONLY if the event has zero connection to financial markets
- Dates must reflect historical reality when the event is a known historical event
- Respond with the JSON object only — no markdown, no explanation"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _guardian_query(event_name: str) -> str | None:
    """Return an ASCII-safe Guardian search query, or None to skip."""
    # Count meaningful ASCII letters (exclude pure digits)
    ascii_letters = re.findall(r'[A-Za-z]', event_name)
    if len(ascii_letters) >= 4:
        return event_name          # English or mixed — use as-is
    # Pure Korean / CJK with no English words → skip Guardian pre-LLM
    return None


# ─── Web context search (runs BEFORE LLM) ────────────────────────────────────

def search_context(event_name: str, year: int = None) -> tuple:
    context, news = "", []
    ua    = "MacroPortalBot/1.0"
    query = f"{event_name} {year}" if year else event_name

    # ── DuckDuckGo Instant Answer ──
    try:
        q   = urllib.parse.quote(query + " market sector financial impact ETF")
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.loads(r.read().decode("utf-8"))
        abstract = d.get("Abstract", "")
        if abstract:
            context += abstract + " "
            news.append({"title": d.get("Heading", ""), "snippet": abstract[:250],
                         "url": d.get("AbstractURL", "")})
        for t in d.get("RelatedTopics", [])[:3]:
            if isinstance(t, dict):
                context += t.get("Text", "") + " "
    except Exception:
        pass

    # ── English Wikipedia ──
    try:
        q   = urllib.parse.quote(query)
        url = (f"https://en.wikipedia.org/w/api.php?action=query&list=search"
               f"&srsearch={q}&format=json&srlimit=3&utf8=1")
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.loads(r.read().decode("utf-8"))
        for item in d.get("query", {}).get("search", [])[:2]:
            snip  = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            title = item.get("title", "")
            context += snip + " "
            news.append({"title": title, "snippet": snip[:250],
                         "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"})
    except Exception:
        pass

    # ── Korean Wikipedia ──
    try:
        q   = urllib.parse.quote(event_name)
        url = (f"https://ko.wikipedia.org/w/api.php?action=query&list=search"
               f"&srsearch={q}&format=json&srlimit=2&utf8=1")
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.loads(r.read().decode("utf-8"))
        for item in d.get("query", {}).get("search", [])[:1]:
            context += re.sub(r"<[^>]+>", "", item.get("snippet", "")) + " "
    except Exception:
        pass

    # ── The Guardian (date-filtered, LLM에 제공되는 실제 기사) ──
    if year:
        guardian_q = _guardian_query(event_name)
        if guardian_q:
            try:
                from_dt = f"{year - 1}-01-01"
                to_dt   = f"{year + 1}-06-30"
                q   = urllib.parse.quote(guardian_q)
                url = (f"https://content.guardianapis.com/search"
                       f"?q={q}&from-date={from_dt}&to-date={to_dt}"
                       f"&api-key=test&page-size=6&order-by=relevance&show-fields=trailText")
                req = urllib.request.Request(url, headers={"User-Agent": ua})
                with urllib.request.urlopen(req, timeout=8) as r:
                    d = json.loads(r.read().decode("utf-8"))
                for item in d.get("response", {}).get("results", [])[:5]:
                    title = item.get("webTitle", "")
                    href  = item.get("webUrl", "")
                    snip  = (item.get("fields") or {}).get("trailText", "") or ""
                    date  = item.get("webPublicationDate", "")[:10]
                    if title:
                        # Both fed to LLM (context) and shown to user (news)
                        context += title + ". " + re.sub(r"<[^>]+>", "", snip)[:200] + " "
                        news.append({
                            "title":   title,
                            "snippet": f"The Guardian · {date}" + (
                                f" — {re.sub(r'<[^>]+>', '', snip)[:160]}" if snip else ""),
                            "url":     href,
                        })
            except Exception:
                pass

    return context[:4000].strip(), news


# ─── Groq LLM call ───────────────────────────────────────────────────────────

def call_groq(event_name: str, context: str, api_key: str) -> dict:
    user_msg = f"Event: {event_name}"
    if context:
        user_msg += f"\n\nNews/Wikipedia/Guardian context:\n{context}"

    payload = json.dumps({
        "model":   GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens":  512,
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_URL, data=payload,
        headers={
            "Authorization":  f"Bearer {api_key}",
            "Content-Type":   "application/json",
            "User-Agent":     "groq-python/0.11.0",
            "Accept":         "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8-sig")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq HTTP {e.code}: {body[:300]}")
    data    = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    content = content.strip().lstrip("﻿")
    return json.loads(content)


# ─── Main classify ────────────────────────────────────────────────────────────

def classify(event_name: str) -> dict:
    year          = next((int(m) for m in re.findall(r"(?:19|20)\d{2}", event_name)), None)
    context, news = search_context(event_name, year)   # Guardian included here
    raw_key       = os.environ.get("GROQ_API_KEY", "")
    api_key       = raw_key.encode("utf-8").lstrip(b"\xef\xbb\xbf").decode("utf-8").strip()

    if not api_key:
        return _err("GROQ_API_KEY environment variable not set", news)

    try:
        result = call_groq(event_name, context, api_key)
        for f in ("ticker", "direction", "event_type", "start", "oos_start", "can_proceed"):
            if f not in result:
                raise ValueError(f"LLM response missing field: {f}")
        result.setdefault("confidence", "medium")
        result.setdefault("sector", "")
        result.setdefault("mechanism", "")
        result.setdefault("end", "")
        result["news"]          = news
        result["name_score"]    = 2
        result["context_score"] = 1 if context else 0
        return result

    except Exception as e:
        return _err(str(e), news)


def _err(msg: str, news: list) -> dict:
    return {
        "ticker": "", "sector": "", "direction": "SHORT",
        "event_type": "UNKNOWN", "start": "", "oos_start": "", "end": "",
        "mechanism": msg, "confidence": "low", "can_proceed": False,
        "name_score": 0, "context_score": 0, "news": news,
        "error": msg,
    }


# ─── Vercel handler ───────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self): self._send(200, {})

    def do_POST(self):
        try:
            n   = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b"{}"
            body = json.loads(raw.decode("utf-8"))
            self._send(200, classify(body.get("event_name", "")))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def _send(self, code, data):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length",  str(len(payload)))
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_): pass
