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

# Guardian: 정식 무료 키가 있으면 사용, 없으면 데모('test') 폴백 (무중단).
# 발급: https://open-platform.theguardian.com/access/  →  Vercel env GUARDIAN_API_KEY
GUARDIAN_KEY = os.environ.get("GUARDIAN_API_KEY", "test")

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

Disambiguation rules (CRITICAL):
- If the event name mentions a SPECIFIC entity (ship, company, person, product),
  classify THAT entity's event — NOT a different historical event at the same location.
  e.g. "Suez Canal Ever Given grounding" (에버기븐호 좌초) = the March 2021 canal
  obstruction (oos_start 2021-03-23), NOT the 1956 Suez Crisis.
- If the provided news/Wikipedia context contains explicit dates or years,
  those dates OVERRIDE your prior knowledge. Anchor start/oos_start/end to the
  context dates, even if you associate the location with another famous event.
- Korean event names: 좌초=grounding, 파산=bankruptcy, 봉쇄=blockade, 침공=invasion.

General:
- can_proceed = false ONLY if the event has zero connection to financial markets
- Dates must reflect historical reality when the event is a known historical event
- Respond with the JSON object only — no markdown, no explanation"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

UA = "MacroPortalBot/1.0"


def _get_json(url: str, timeout: int = 6) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _is_korean_query(event_name: str) -> bool:
    """의미있는 ASCII 단어가 거의 없으면 한글/CJK 전용 쿼리로 취급."""
    return len(re.findall(r"[A-Za-z]", event_name)) < 4


def _ko_to_en_title(event_name: str) -> str | None:
    """ko.wikipedia 검색 1위 문서의 영문(langlinks) 제목 → 영문 검색 쿼리로 사용.

    한글 전용 이벤트명이 en-wiki/Guardian 에서 0건이 되는 문제 해결 (2026-06-12).
    예: '수에즈 운하 에버기븐호 좌초 사건' → ko 문서 → en 'Ever Given obstruction...'
    """
    try:
        q = urllib.parse.quote(event_name)
        d = _get_json(f"https://ko.wikipedia.org/w/api.php?action=query&list=search"
                      f"&srsearch={q}&format=json&srlimit=1&utf8=1")
        hits = d.get("query", {}).get("search", [])
        if not hits:
            return None
        ko_title = hits[0].get("title", "")
        if not ko_title:
            return None
        t = urllib.parse.quote(ko_title)
        d2 = _get_json(f"https://ko.wikipedia.org/w/api.php?action=query&prop=langlinks"
                       f"&lllang=en&titles={t}&format=json&utf8=1")
        for page in d2.get("query", {}).get("pages", {}).values():
            for ll in page.get("langlinks", []) or []:
                en = ll.get("*") or ll.get("title")
                if en:
                    return en
    except Exception:
        pass
    return None


def _extract_context_year(context: str) -> int | None:
    """컨텍스트에서 가장 많이 등장한 연도 (이벤트 발생 연도 추정)."""
    from collections import Counter
    yrs = [int(y) for y in re.findall(r"(?:19|20)\d{2}", context)
           if 1900 <= int(y) <= 2030]
    if not yrs:
        return None
    return Counter(yrs).most_common(1)[0][0]


def _wiki_intro(title: str, lang: str = "en", timeout: int = 6) -> str:
    """위키 문서 도입부 본문(extract). search snippet(HTML 조각 ~250자) 보다
    훨씬 풍부하고 발생 날짜를 그대로 담고 있어 LLM 의 연도 앵커링에 결정적
    (2026-08-03: DuckDuckGo IA 가 이벤트에 빈 응답이라 위키 본문으로 보강)."""
    if not title:
        return ""
    try:
        t = urllib.parse.quote(title)
        d = _get_json(f"https://{lang}.wikipedia.org/w/api.php?action=query"
                      f"&prop=extracts&exintro&explaintext&redirects=1"
                      f"&titles={t}&format=json&utf8=1", timeout=timeout)
        for page in d.get("query", {}).get("pages", {}).values():
            ex = page.get("extract", "")
            if ex:
                return ex.strip()
    except Exception:
        pass
    return ""


# ─── Web context search (runs BEFORE LLM) ────────────────────────────────────

def search_context(event_name: str, year: int = None, en_hint: str = None) -> tuple:
    """반환: (context 텍스트, news 리스트, 컨텍스트 추정 연도, 영문 해석 제목).

    en_hint: LLM 정규화가 만든 영문 검색어(있으면 우선). ko→en 위키 해석보다
    우선하며, 구어 쿼리에서도 Guardian/en-wiki 가 제대로 걸리게 한다.
    """
    context, news = "", []

    # 영문 쿼리 결정: LLM 정규화(en_hint) > ko-wiki langlink 해석 > 원문
    en_title = None
    if en_hint is None and _is_korean_query(event_name):
        en_title = _ko_to_en_title(event_name)
    en_query = en_hint or en_title or event_name
    query    = f"{en_query} {year}" if year else en_query

    # ── DuckDuckGo Instant Answer ──
    try:
        q = urllib.parse.quote(query + " market sector financial impact ETF")
        d = _get_json(f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1")
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

    # ── English Wikipedia (영문 해석 제목 우선) ──
    try:
        q = urllib.parse.quote(query)
        d = _get_json(f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                      f"&srsearch={q}&format=json&srlimit=3&utf8=1")
        hits = d.get("query", {}).get("search", [])
        # 1위 문서는 도입부 본문까지 (snippet 보다 날짜·맥락이 풍부)
        intro = _wiki_intro(hits[0].get("title", ""), "en") if hits else ""
        if intro:
            context += "[WIKIPEDIA] " + intro[:1500] + " "
        for i, item in enumerate(hits[:2]):
            snip  = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            title = item.get("title", "")
            # 1위는 본문을 이미 넣었으니 snippet 중복 회피 (news 링크는 유지)
            if not (i == 0 and intro):
                context += snip + " "
            news.append({"title": title, "snippet": snip[:250],
                         "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"})
    except Exception:
        pass

    # ── Korean Wikipedia (context + news 양쪽에 포함) ──
    try:
        q = urllib.parse.quote(event_name)
        d = _get_json(f"https://ko.wikipedia.org/w/api.php?action=query&list=search"
                      f"&srsearch={q}&format=json&srlimit=2&utf8=1")
        ko_hits = d.get("query", {}).get("search", [])
        ko_intro = _wiki_intro(ko_hits[0].get("title", ""), "ko") if ko_hits else ""
        if ko_intro:
            context += "[한국어 위키] " + ko_intro[:1200] + " "
        for i, item in enumerate(ko_hits[:1]):
            snip  = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            title = item.get("title", "")
            if not (i == 0 and ko_intro):   # 1위 본문 중복 회피
                context += snip + " "
            if title:
                # 기존엔 context 에만 들어가 UI 에 '뉴스 없음' 으로 표시되던 것 수정
                news.append({"title": f"{title} (한국어 위키)", "snippet": snip[:250],
                             "url": f"https://ko.wikipedia.org/wiki/{urllib.parse.quote(title)}"})
    except Exception:
        pass

    # ── The Guardian — 연도 있으면 날짜 필터, 없으면 relevance 검색 (2026-06-12) ──
    guardian_q = en_query if len(re.findall(r"[A-Za-z]", en_query)) >= 4 else None
    if guardian_q:
        try:
            date_filter = ""
            if year:
                date_filter = f"&from-date={year - 1}-01-01&to-date={year + 1}-06-30"
            q = urllib.parse.quote(guardian_q)
            d = _get_json(f"https://content.guardianapis.com/search"
                          f"?q={q}{date_filter}"
                          f"&api-key={GUARDIAN_KEY}&page-size=6&order-by=relevance&show-fields=trailText",
                          timeout=8)
            for item in d.get("response", {}).get("results", [])[:5]:
                title = item.get("webTitle", "")
                href  = item.get("webUrl", "")
                snip  = (item.get("fields") or {}).get("trailText", "") or ""
                date  = item.get("webPublicationDate", "")[:10]
                if title:
                    # Both fed to LLM (context) and shown to user (news)
                    context += f"[GUARDIAN {date}] " + title + ". " + re.sub(r"<[^>]+>", "", snip)[:200] + " "
                    news.append({
                        "title":   title,
                        "snippet": f"The Guardian · {date}" + (
                            f" — {re.sub(r'<[^>]+>', '', snip)[:160]}" if snip else ""),
                        "url":     href,
                    })
        except Exception:
            pass

    context = context[:5000].strip()
    return context, news, _extract_context_year(context), (en_hint or en_title)


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


# ─── Query normalization (runs BEFORE context search) ────────────────────────

NORMALIZE_SYSTEM = """You turn a market-event name (any language, possibly colloquial \
or slang) into an English web-search query for news/Wikipedia retrieval.

Return ONLY a JSON object:
{
  "english_query":  "concise English search keywords: the event, key entities, and place (NOT a sentence)",
  "canonical_name": "the widely-used English name of the event if you know it, else \"\"",
  "year":           4-digit year the event occurred, or null if unsure
}

Rules:
- Expand local jargon to its real meaning. e.g. Korean '소부장' = semiconductor
  materials/parts/equipment; '소부장 사태' = the 2019 Japan-South Korea export-
  restriction dispute over semiconductor materials (canonical_name
  'Japan-South Korea trade dispute', year 2019).
- english_query = keywords a news search engine would match, not prose.
- Set year only when confident; otherwise null.
- JSON object only — no markdown, no explanation."""


def _normalize_query(event_name: str, api_key: str, timeout: int = 12) -> dict | None:
    """구어/타언어 이벤트명 → 영문 검색어 + 연도. 컨텍스트 수집 전 1회 호출.
    한국식 구어('소부장 사태')가 ko→en 위키 해석에 실패해 Guardian/en-wiki 가
    통째 스킵되던 문제 해결 (2026-08-03). 실패 시 None → 기존 경로로 폴백."""
    payload = json.dumps({
        "model":   GROQ_MODEL,
        "messages": [
            {"role": "system", "content": NORMALIZE_SYSTEM},
            {"role": "user",   "content": f"Event: {event_name}"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens":  200,
    }).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL, data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 "User-Agent":   "groq-python/0.11.0",
                 "Accept":       "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.loads(r.read().decode("utf-8-sig"))["choices"][0]["message"]["content"]
        d = json.loads(content.strip().lstrip("﻿"))
    except Exception:
        return None

    # 연도 정합성
    y = d.get("year")
    if isinstance(y, str) and re.match(r"^\d{4}$", y.strip()):
        y = int(y)
    d["year"] = y if (isinstance(y, int) and 1900 <= y <= 2030) else None
    # 검색 폭 확보: canonical_name + keywords 결합
    eq = (d.get("english_query") or "").strip()
    cn = (d.get("canonical_name") or "").strip()
    combined = f"{cn} {eq}".strip() if cn else eq
    d["english_query"] = combined or None
    return d


# ─── Main classify ────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = ("ticker", "direction", "event_type", "start", "oos_start", "can_proceed")
MIN_ETF_YEAR = 1993   # SPY 상장(1993) 이전엔 미국 섹터 ETF 데이터 자체가 없음


def _year_of(date_str) -> int | None:
    m = re.match(r"^(\d{4})", str(date_str or ""))
    return int(m.group(1)) if m else None


def _check_fields(result: dict) -> None:
    for f in _REQUIRED_FIELDS:
        if f not in result:
            raise ValueError(f"LLM response missing field: {f}")


def classify(event_name: str) -> dict:
    year = next((int(m) for m in re.findall(r"(?:19|20)\d{2}", event_name)), None)
    raw_key = os.environ.get("GROQ_API_KEY", "")
    api_key = raw_key.encode("utf-8").lstrip(b"\xef\xbb\xbf").decode("utf-8").strip()

    # ── LLM 쿼리 정규화 (컨텍스트 수집 전) — 구어/타언어 → 영문 검색어 + 연도 ──
    en_hint = None
    norm_query = None
    if api_key:
        norm = _normalize_query(event_name, api_key)
        if norm:
            en_hint    = norm.get("english_query")
            norm_query = en_hint
            year       = year or norm.get("year")   # 이벤트명 명시 연도 우선

    context, news, ctx_year, en_title = search_context(event_name, year, en_hint=en_hint)

    if not api_key:
        return _err("GROQ_API_KEY environment variable not set", news)

    try:
        result = call_groq(event_name, context, api_key)
        _check_fields(result)

        warnings: list[str] = []
        # 사건 연도 힌트: 이벤트명 연도 > 컨텍스트 최빈 연도
        hint_year = year or ctx_year
        oos_year  = _year_of(result.get("oos_start"))

        # ── 연도 불일치 → 1회 재질의 (동명 사건 혼동 보정, 2026-06-12) ──
        # 예: '수에즈 운하 에버기븐호'(2021) 를 1956 수에즈 위기로 오인하는 케이스
        if hint_year and oos_year and abs(oos_year - hint_year) > 2:
            first_year = oos_year
            hint = (f"\n\nIMPORTANT CORRECTION HINT: reliable context indicates this event "
                    f"occurred in {hint_year}. Your previous answer dated it {first_year}, "
                    f"which likely confuses a DIFFERENT historical event at the same "
                    f"location. Re-classify anchored to {hint_year}.")
            try:
                retry = call_groq(event_name, (context or "") + hint, api_key)
                _check_fields(retry)
                retry_year = _year_of(retry.get("oos_start"))
                if retry_year and abs(retry_year - hint_year) <= 2:
                    result, oos_year = retry, retry_year
                    warnings.append(
                        f"1차 분류가 {first_year}년 사건으로 오인 → "
                        f"컨텍스트 연도({hint_year}) 기준 재질의로 보정됨")
            except Exception:
                pass   # 재질의 실패 시 1차 결과 유지 (아래 가드가 경고)

        result.setdefault("confidence", "medium")
        result.setdefault("sector", "")
        result.setdefault("mechanism", "")
        result.setdefault("end", "")

        # ── 날짜 sanity 가드 ──
        if oos_year and oos_year < MIN_ETF_YEAR:
            result["confidence"] = "low"
            warnings.append(
                f"oos_start {result.get('oos_start')} — {MIN_ETF_YEAR}년 이전은 "
                f"미국 섹터 ETF 데이터가 없어 분석 불가. 날짜를 직접 확인하세요.")
        if hint_year and oos_year and abs(oos_year - hint_year) > 2:
            result["confidence"] = "low"
            warnings.append(
                f"LLM 연도({oos_year})가 컨텍스트 연도({hint_year})와 불일치 — "
                f"같은 장소의 다른 역사적 사건과 혼동했을 가능성. 날짜를 직접 확인하세요.")

        # ── 정직한 스코어 (기존: name_score=2 하드코딩, context_score=존재 여부) ──
        result["news"]          = news
        result["name_score"]    = 2 if year else 1
        result["context_score"] = 2 if len(news) >= 2 else (1 if context else 0)
        result["context_year"]  = ctx_year
        result["resolved_title"] = en_title   # 사용된 영문 쿼리 (정규화 or ko→en 위키)
        result["normalized_query"] = norm_query   # LLM 정규화 검색어 (디버그/표시용)
        result["warnings"]      = warnings
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
