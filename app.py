from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

TZ = "Europe/Istanbul"
DB_PATH = Path(__file__).parent / "journal.db"

UNIVERSE = ["AKBNK", "ARCLK", "ASELS", "BIMAS", "EREGL", "FROTO", "GARAN", "ISCTR", "KCHOL", "SAHOL",
            "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"]
try:
    _all = sorted(set((Path(__file__).parent / "hisseler.txt").read_text(encoding="utf-8").upper().split()))
    if len(_all) > 50:
        UNIVERSE = _all
except Exception:
    pass

DEFAULT_BANNED = {"DIRIT", "CASA", "COSMO", "EKIZ", "ETILR", "KUVVA", "MARKA", "MMCAS", "ROYAL", "SELGD", "UFUK", "UMPAS"}

# Profil: (min RVOL, değişim alt, üst, RSI alt, üst, min günlük hacim M TL, min günlük hareket %)
PROFILES = {
    "🛡️ Sakin ve Güvenli": (2.0, 1.5, 4.0, 50, 65, 100, 2.5),
    "⚖️ Dengeli (Önerilen)": (1.5, 1.5, 6.0, 50, 70, 100, 2.5),
    "🔥 Hızlı ve Hareketli": (1.2, 1.0, 7.0, 50, 75, 50, 2.0)
}

st.set_page_config(page_title="BIST Kolay Tarayıcı", page_icon="📈", layout="wide")
st.markdown("""<style>
#MainMenu,footer{visibility:hidden}
.block-container{max-width:980px;padding-top:1.2rem}
.hdr{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;padding:20px 24px;border-radius:16px;margin-bottom:14px}
.hdr h1{margin:0;font-size:1.55rem;color:#fff;padding:0}
.hdr p{margin:6px 0 0;opacity:.85;font-size:.9rem}
.card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:18px 20px;margin:14px 0 8px;background:rgba(128,128,128,.05)}
.card-live{border:2px solid #16a34a;border-radius:16px;padding:18px 20px;margin:14px 0 8px;background:rgba(22,163,74,.05)}
.top{display:flex;justify-content:space-between;align-items:center;gap:10px}
.tk{font-size:1.6rem;font-weight:800;letter-spacing:.5px}
.sub{opacity:.75;font-size:.85rem;margin-top:2px}
.pill{padding:6px 14px;border-radius:999px;font-weight:700;font-size:.82rem;color:#fff;white-space:nowrap}
.live-badge{display:inline-block;padding:2px 8px;border-radius:6px;background:#16a34a;color:#fff;font-size:.72rem;font-weight:700;margin-left:8px;vertical-align:middle}
.summary-box{margin:12px 0;padding:10px 14px;border-radius:10px;background:rgba(37,99,235,.08);font-size:.88rem;line-height:1.45}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:12px}
@media(min-width:650px){.grid{grid-template-columns:repeat(4,1fr)}}
.box{border-radius:12px;padding:10px 12px;background:rgba(128,128,128,.1)}
.box small{opacity:.75;display:block;font-size:.74rem;margin-bottom:2px}
.box b{font-size:1.08rem}
.bar{display:flex;align-items:center;gap:8px;font-size:.8rem;margin:6px 0}
.bar span{width:120px;opacity:.85}
.track{flex:1;height:8px;border-radius:6px;background:rgba(128,128,128,.25);overflow:hidden}
.fill{height:100%;border-radius:6px}
.bar b{width:32px;text-align:right}
</style>""", unsafe_allow_html=True)


# ---------- Veritabanı (İşlemlerim) ----------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                mode TEXT,
                ticker TEXT,
                entry REAL,
                stop REAL,
                target1 REAL,
                target2 REAL,
                lot INTEGER,
                status TEXT DEFAULT 'AÇIK',
                exit_price REAL,
                pnl REAL
            )
        """)


def add_trade(mode, ticker, entry, stop, t1, t2, lot):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO trades (created_at, mode, ticker, entry, stop, target1, target2, lot) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%d.%m.%Y %H:%M"), mode, ticker, round(entry, 2), round(stop, 2), round(t1, 2), round(t2, 2), int(lot))
        )


def get_trades():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)


def update_trade_status(trade_id, status, exit_price=None):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT entry, lot FROM trades WHERE id = ?", (trade_id,)).fetchone()
        pnl = None
        if row and exit_price is not None and exit_price > 0:
            pnl = round((exit_price - row[0]) * row[1], 2)
        conn.execute("UPDATE trades SET status = ?, exit_price = ?, pnl = ? WHERE id = ?", (status, exit_price, pnl, trade_id))


def delete_trade(trade_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))


init_db()


# ---------- Yardımcılar ----------
def tl(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def ok_num(x):
    return x is not None and isinstance(x, (int, float)) and x == x


def color(v):
    return "#16a34a" if v >= 70 else "#d97706" if v >= 50 else "#dc2626"


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def flow(d):
    d = d.tail(60)
    hl = (d["High"] - d["Low"]).replace(0, np.nan)
    mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / hl
    vol_sum = d["Volume"].tail(20).sum()
    cmf = float((mfm * d["Volume"]).tail(20).sum() / vol_sum) if vol_sum > 0 else 0.0
    obv = (np.sign(d["Close"].diff()).fillna(0) * d["Volume"]).cumsum()
    obv_up = bool(obv.iloc[-1] > obv.rolling(20).mean().iloc[-1])
    big = d["Volume"] > 1.5 * d["Volume"].rolling(20).mean().shift()
    acc = int((big & (d["Close"] > d["Open"])).tail(10).sum())
    dist = int((big & (d["Close"] < d["Open"])).tail(10).sum())
    pts = int(cmf > 0.05) + int(obv_up) + int(acc > dist)
    return pts, cmf, acc, dist


def fund_score(i):
    items = [
        ("Fiyat / Kazanç Oranı (Ucuzluk)", i.get("trailingPE"), 20, lambda v: 0 < v <= 20, lambda v: f"{v:.1f}", "20 ve altı"),
        ("Piyasa / Defter Değeri", i.get("priceToBook"), 15, lambda v: 0 < v <= 3, lambda v: f"{v:.2f}", "3 ve altı"),
        ("Özkaynak Kârlılığı", i.get("returnOnEquity"), 25, lambda v: v >= 0.15, lambda v: f"%{v*100:.0f}", "%15 ve üzeri"),
        ("Net Kâr Marjı", i.get("profitMargins"), 15, lambda v: v >= 0.08, lambda v: f"%{v*100:.0f}", "%8 ve üzeri"),
        ("Kâr Büyümesi", i.get("earningsGrowth"), 10, lambda v: v > 0, lambda v: f"%{v*100:.0f}", "Pozitif"),
        ("Borç Oranı", i.get("debtToEquity"), 15, lambda v: v <= 100, lambda v: f"%{v:.0f}", "%100 ve altı"),
    ]
    pts = mx = 0
    rows = []
    for name, v, w, test, fmt, rule in items:
        if ok_num(v):
            mx += w
            good = bool(test(v))
            pts += w if good else 0
            rows.append((name, fmt(v), rule, good))
    return (round(pts / mx * 100) if mx >= 40 else None), mx, rows


def overall_intraday(tek, para):
    return 0.65 * tek + 0.35 * para


def overall_swing(tek, temel, para):
    w = {"t": 0.40, "f": 0.35, "p": 0.25}
    parts = {"t": tek, "p": para}
    if temel is not None:
        parts["f"] = temel
    tot = sum(w[k] for k in parts)
    return sum(w[k] * v for k, v in parts.items()) / tot


def verdict(g):
    return ("ÇOK GÜÇLÜ", "#16a34a") if g >= 75 else ("UYGUN", "#2563eb") if g >= 60 else ("DİKKATLİ OL", "#d97706")


# ---------- Veri İndirme ----------
def _download(tickers, period, interval):
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        try:
            raw = yf.download([t + ".IS" for t in chunk], period=period, interval=interval,
                              group_by="ticker", progress=False, threads=True, auto_adjust=False)
        except Exception:
            continue
        for t in chunk:
            try:
                d = raw[t + ".IS"].dropna(subset=["Close"]) if len(chunk) > 1 else raw.dropna(subset=["Close"])
            except KeyError:
                continue
            if len(d) < 30:
                continue
            d.index = d.index.tz_localize("UTC").tz_convert(TZ) if d.index.tz is None else d.index.tz_convert(TZ)
            out[t] = d
    return out


@st.cache_data(ttl=3600, show_spinner="Borsa hisseleri taranıyor, lütfen bekleyin...")
def get_daily(tickers):
    return _download(list(tickers), "1y", "1d")


@st.cache_data(ttl=240, show_spinner="Anlık hareketler inceleniyor...")
def get_intraday(tickers):
    return _download(list(tickers), "5d", "15m")


@st.cache_data(ttl=240)
def get_xu100():
    try:
        d15 = yf.download("XU100.IS", period="5d", interval="15m", progress=False, auto_adjust=False).dropna(subset=["Close"])
        d1d = yf.download("XU100.IS", period="6mo", interval="1d", progress=False, auto_adjust=False).dropna(subset=["Close"])
        if isinstance(d15.columns, pd.MultiIndex):
            d15.columns = d15.columns.get_level_values(0)
        if isinstance(d1d.columns, pd.MultiIndex):
            d1d.columns = d1d.columns.get_level_values(0)
        if not d15.empty:
            d15.index = d15.index.tz_localize("UTC").tz_convert(TZ) if d15.index.tz is None else d15.index.tz_convert(TZ)
        if not d1d.empty:
            d1d.index = d1d.index.tz_localize("UTC").tz_convert(TZ) if d1d.index.tz is None else d1d.index.tz_convert(TZ)
        return d15, d1d
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner="Şirket bilançoları kontrol ediliyor...")
def get_fund(tickers):
    def one(t):
        try:
            return t, yf.Ticker(t + ".IS").info or {}
        except Exception:
            return t, {}
    with ThreadPoolExecutor(4) as ex:
        return dict(ex.map(one, tickers[:40]))


def get_single_stock_data(ticker):
    t = ticker.strip().upper().replace(".IS", "")
    try:
        d1d = yf.download(t + ".IS", period="1y", interval="1d", progress=False, auto_adjust=False).dropna(subset=["Close"])
        d15 = yf.download(t + ".IS", period="5d", interval="15m", progress=False, auto_adjust=False).dropna(subset=["Close"])
        if isinstance(d1d.columns, pd.MultiIndex):
            d1d.columns = d1d.columns.get_level_values(0)
        if isinstance(d15.columns, pd.MultiIndex):
            d15.columns = d15.columns.get_level_values(0)
        if not d1d.empty:
            d1d.index = d1d.index.tz_localize("UTC").tz_convert(TZ) if d1d.index.tz is None else d1d.index.tz_convert(TZ)
        if not d15.empty:
            d15.index = d15.index.tz_localize("UTC").tz_convert(TZ) if d15.index.tz is None else d15.index.tz_convert(TZ)
        info = yf.Ticker(t + ".IS").info or {}
        return t, d1d, d15, info
    except Exception:
        return t, pd.DataFrame(), pd.DataFrame(), {}


def check_market_regime(d15, d1d):
    info = {"ok": True, "chg": 0.0, "msg": "Borsa genel durumu normal görünüyor."}
    if d1d.empty or len(d1d) < 20:
        return info
    c_daily = float(d1d["Close"].iloc[-1])
    prev_close = float(d1d["Close"].iloc[-2])
    ema20_daily = float(ema(d1d["Close"], 20).iloc[-1])
    chg = (c_daily / prev_close - 1) * 100

    if not d15.empty and len(d15) >= 20:
        d15 = d15.copy()
        d15["day"] = d15.index.date
        last_day = d15["day"].iloc[-1]
        today = d15[d15["day"] == last_day]
        prev = d15[d15["day"] < last_day]
        if not prev.empty and not today.empty:
            c15 = float(today["Close"].iloc[-1])
            ema20_15 = float(ema(d15["Close"], 20).iloc[-1])
            chg = (c15 / float(prev["Close"].iloc[-1]) - 1) * 100
            is_healthy = (chg > -1.0) and (c15 >= ema20_15 * 0.995)
            status_txt = "Borsa yükseliş / yatay seyrinde (Alım için uygun)" if is_healthy else "Borsada genel satış var (Alım yapmak riskli!)"
            return {"ok": is_healthy, "chg": chg, "msg": f"BIST 100 Endeksi: %{chg:+.2f} • {status_txt}"}

    is_healthy = (chg > -1.0) and (c_daily >= ema20_daily * 0.99)
    status_txt = "Borsa genel görünümü olumlu" if is_healthy else "Borsada düşüş eğilimi var (Dikkatli olun)"
    return {"ok": is_healthy, "chg": chg, "msg": f"BIST 100 Endeksi: %{chg:+.2f} • {status_txt}"}


def daily_table(daily, min_tl_m, min_atr, banned_set):
    today = pd.Timestamp.now(TZ).date()
    rows = []
    for t, d in daily.items():
        if t in banned_set:
            continue
        d_prev = d[d.index.date < today]
        if len(d_prev) < 30:
            continue
        c = float(d_prev["Close"].iloc[-1])
        vol = float((d_prev["Close"] * d_prev["Volume"]).tail(20).mean() / 1e6)
        daily_atr_val = float(atr(d_prev).iloc[-1])
        a_pct = daily_atr_val / c * 100
        ema20 = float(ema(d_prev["Close"], 20).iloc[-1])
        up = c > ema20
        fp, cmf, acc, dist = flow(d_prev)
        rows.append({
            "Hisse": t,
            "Kapanış": c,
            "İşlem (M TL)": vol,
            "Hareket %": a_pct,
            "DailyATR": daily_atr_val,
            "ParaP": fp,
            "CMF": cmf,
            "Alım günü": acc,
            "Satım günü": dist,
            "Geçti": bool(vol >= min_tl_m and a_pct >= min_atr and up and c >= 5)
        })
    return pd.DataFrame(rows)


def check_wick_health(row):
    high, low, open_, close = float(row["High"]), float(row["Low"]), float(row["Open"]), float(row["Close"])
    bar_range = high - low
    if bar_range <= 0:
        return True, 0.0
    upper_wick = high - max(open_, close)
    wick_ratio = upper_wick / bar_range
    return wick_ratio <= 0.45, wick_ratio * 100


def apply_live_price_to_df(df, live_price):
    """
    Kullanıcının aracı kurumundan girip yazdığı anlık fiyatı serinin son barına işler.
    Böylece EMA, RSI, VWAP, günlük değişim % ve fitil yapısı anlık fiyata göre baştan hesaplanır!
    """
    if df is None or df.empty or live_price is None or live_price <= 0:
        return df
    df = df.copy()
    idx = df.index[-1]
    df.loc[idx, "Close"] = float(live_price)
    if live_price > float(df.loc[idx, "High"]):
        df.loc[idx, "High"] = float(live_price)
    if live_price < float(df.loc[idx, "Low"]):
        df.loc[idx, "Low"] = float(live_price)
    return df


def compute_smart_plan(entry, raw_support, daily_atr, capital, risk_pct, temel_puan, tek_puan, is_swing):
    if is_swing and temel_puan is not None:
        if temel_puan >= 65:
            atr_buffer = 0.65
            rr1, rr2 = 2.2, 3.8
            plan_note = f"Bilanço güçlü ({temel_puan}/100): Silkelemelere karşı destek payı korundu, hedefler genişletildi."
        elif temel_puan >= 45:
            atr_buffer = 0.50
            rr1, rr2 = 1.8, 3.0
            plan_note = f"Bilanço orta ({temel_puan}/100): Standart teknik destek ve kâr hedefleri uygulandı."
        else:
            atr_buffer = 0.35
            rr1, rr2 = 1.4, 2.2
            plan_note = f"Bilanço zayıf ({temel_puan}/100): Riski sınırlamak için stop sıkılaştırıldı, hedefler yakın tutuldu."
    else:
        atr_buffer = 0.30 if not is_swing else 0.50
        rr1 = 1.6 if tek_puan >= 80 else 1.4
        rr2 = 2.6 if tek_puan >= 80 else 2.2
        plan_note = "Teknik destek (EMA/VWAP) ve günlük oynaklık (ATR) baz alınarak hesaplandı."

    stop = raw_support - atr_buffer * daily_atr
    min_stop_dist = 0.35 * daily_atr
    if (entry - stop) < min_stop_dist:
        stop = entry - min_stop_dist

    max_stop_dist = (1.8 * daily_atr) if is_swing else (0.9 * daily_atr)
    is_wide_warning = False
    if (entry - stop) > max_stop_dist:
        is_wide_warning = True
        stop = entry - max_stop_dist

    risk = max(entry - stop, 0.01)
    target1 = entry + rr1 * risk
    target2 = entry + rr2 * risk

    max_tl_risk = capital * (risk_pct / 100.0)
    raw_lot = min(max_tl_risk / risk, (capital * 0.40) / entry)
    lot = max(1, int(raw_lot))

    return {
        "stop": round(stop, 2),
        "risk": risk,
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "lot": lot,
        "is_wide": is_wide_warning,
        "plan_note": plan_note,
        "rr1": rr1,
        "rr2": rr2
    }


def analyze_intraday(t, d, p, daily_atr, live_price=None):
    is_live = live_price is not None and live_price > 0
    if is_live:
        d = apply_live_price_to_df(d, live_price)
    else:
        d = d.copy()

    d["day"] = d.index.date
    last_day = d["day"].iloc[-1]
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    d["vwap"] = (tp * d["Volume"]).groupby(d["day"]).cumsum() / d.groupby("day")["Volume"].cumsum()
    d["cumv"] = d.groupby("day")["Volume"].cumsum()
    d["ema9"], d["ema20"] = ema(d["Close"], 9), ema(d["Close"], 20)
    d["rsi"], d["atr"] = rsi(d["Close"]), atr(d)

    today, prev = d[d["day"] == last_day], d[d["day"] < last_day]
    if prev.empty or today.empty:
        return None

    n = len(today)
    pc = [g["cumv"].iloc[n - 1] for _, g in prev.groupby("day") if len(g) >= n]
    rvol = float(today["cumv"].iloc[-1] / np.mean(pc)) if pc and np.mean(pc) > 0 else np.nan
    chg = float((d["Close"].iloc[-1] / prev["Close"].iloc[-1] - 1) * 100)
    r = d.iloc[-1]
    wick_ok, _ = check_wick_health(r)

    checks = [
        (f"Fiyat gün içi ortalamanın ({tl(r['vwap'])} TL) üzerinde güçlü duruyor", bool(r["Close"] > r["vwap"])),
        ("Kısa vadeli yükseliş trendi aktif (EMA9 > EMA20)", bool(r["ema9"] > r["ema20"] and r["Close"] > r["ema20"])),
        (f"Fiyat henüz aşırı şişmemiş (Bugün %{chg:+.1f})", bool(p["chg_lo"] <= chg <= p["chg_hi"])),
        (f"Alım iştahı dengeli (RSI {r['rsi']:.0f})", bool(p["rsi_lo"] <= r["rsi"] <= p["rsi_hi"])),
        ("Hisseye normalin üzerinde işlem hacmi giriyor", bool(rvol >= p["rvol"])),
        ("Tepeden sert satış baskısı (fitil) yok", wick_ok),
    ]

    entry = float(r["Close"])
    raw_support = max(min(float(r["vwap"]), float(r["ema20"])), float(today["Low"].min()))
    puan = sum(ok for _, ok in checks)
    toplam = len(checks)
    tek = puan / toplam * 100

    plan = compute_smart_plan(entry, raw_support, daily_atr, p["capital"], p["risk_pct"], None, tek, is_swing=False)

    return {
        "Hisse": t, "Fiyat": entry, "Değişim": chg, "Kontroller": checks,
        "Puan": puan, "Toplam": toplam, "Tek": tek,
        "Stop": plan["stop"], "Risk": plan["risk"], "Hedef1": plan["target1"], "Hedef2": plan["target2"],
        "Lot": plan["lot"], "IsWide": plan["is_wide"], "PlanNote": plan["plan_note"],
        "IsLive": is_live, "_ts": d.index[-1], "RVOL": rvol
    }


def analyze_swing(t, d, p, daily_atr, temel_puan=None, live_price=None):
    if len(d) < 55:
        return None
    is_live = live_price is not None and live_price > 0
    if is_live:
        d = apply_live_price_to_df(d, live_price)
    else:
        d = d.copy()

    d["ema20"] = ema(d["Close"], 20)
    d["ema50"] = ema(d["Close"], 50)
    d["rsi"] = rsi(d["Close"])
    d["vol20"] = d["Volume"].rolling(20).mean()

    r = d.iloc[-1]
    prev = d.iloc[-2]
    chg = float((r["Close"] / prev["Close"] - 1) * 100)
    rvol = float(r["Volume"] / r["vol20"]) if r["vol20"] > 0 else 1.0
    wick_ok, _ = check_wick_health(r)

    checks = [
        ("Orta ve uzun vadeli trend yukarı yönlü (Fiyat > EMA20 > EMA50)", bool(r["Close"] > r["ema20"] > r["ema50"])),
        (f"Yükseliş gücü sağlıklı ve dengeli (RSI {r['rsi']:.0f})", bool(50 <= r["rsi"] <= 68)),
        ("Alıcı hacmi ortalamanın üzerinde", bool(rvol >= 1.1)),
        (f"Fiyat henüz tavan/aşırı prim yapmamış (%{chg:+.1f})", bool(-1.0 <= chg <= 6.5)),
        ("Gün sonu kapanışı güçlü, satış baskısı yok", wick_ok),
    ]

    entry = float(r["Close"])
    recent_low = float(d["Low"].tail(5).min())
    raw_support = max(float(r["ema20"]), recent_low)
    puan = sum(ok for _, ok in checks)
    toplam = len(checks)
    tek = puan / toplam * 100

    plan = compute_smart_plan(entry, raw_support, daily_atr, p["capital"], p["risk_pct"], temel_puan, tek, is_swing=True)

    return {
        "Hisse": t, "Fiyat": entry, "Değişim": chg, "Kontroller": checks,
        "Puan": puan, "Toplam": toplam, "Tek": tek,
        "Stop": plan["stop"], "Risk": plan["risk"], "Hedef1": plan["target1"], "Hedef2": plan["target2"],
        "Lot": plan["lot"], "IsWide": plan["is_wide"], "PlanNote": plan["plan_note"],
        "IsLive": is_live, "_ts": d.index[-1], "RVOL": rvol
    }


def run_backtest(daily_dict, universe_list, hold_days=5, rr_target=2.0):
    trades = []
    for t in universe_list:
        d = daily_dict.get(t)
        if d is None or len(d) < 80:
            continue
        df = d.copy()
        df["ema20"] = ema(df["Close"], 20)
        df["ema50"] = ema(df["Close"], 50)
        df["rsi"] = rsi(df["Close"])
        df["atr"] = atr(df)
        df["vol20"] = df["Volume"].rolling(20).mean()
        df["tl_vol"] = (df["Close"] * df["Volume"]).rolling(20).mean() / 1e6

        in_trade = False
        entry_p = stop_p = target_p = 0.0
        days_held = 0

        for i in range(55, len(df)):
            row = df.iloc[i]
            if in_trade:
                days_held += 1
                if row["Low"] <= stop_p:
                    trades.append({"Hisse": t, "Sonuç": "Zarar Kes", "R": -1.0})
                    in_trade = False
                elif row["High"] >= target_p:
                    trades.append({"Hisse": t, "Sonuç": "Kâr Hedefi", "R": rr_target})
                    in_trade = False
                elif days_held >= hold_days:
                    risk_unit = entry_p - stop_p
                    pnl_r = (row["Close"] - entry_p) / risk_unit if risk_unit > 0 else 0
                    trades.append({"Hisse": t, "Sonuç": "Süre Sonu", "R": pnl_r})
                    in_trade = False
            else:
                wick_ok, _ = check_wick_health(row)
                cond = (
                    row["tl_vol"] >= 50
                    and row["Close"] > row["ema20"] > row["ema50"]
                    and 50 <= row["rsi"] <= 68
                    and row["Volume"] >= 1.1 * row["vol20"]
                    and wick_ok
                )
                if cond and row["atr"] > 0:
                    entry_p = float(row["Close"])
                    stop_p = float(row["ema20"]) - 0.5 * float(row["atr"])
                    if stop_p >= entry_p:
                        stop_p = entry_p - float(row["atr"])
                    risk_unit = entry_p - stop_p
                    if 0 < risk_unit <= 1.8 * float(row["atr"]):
                        target_p = entry_p + rr_target * risk_unit
                        in_trade = True
                        days_held = 0

    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["R"] > 0]
    losses = tdf[tdf["R"] <= 0]
    win_rate = len(wins) / len(tdf) * 100
    gross_profit = wins["R"].sum()
    gross_loss = abs(losses["R"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    tdf["Kümülatif Kazanç (R)"] = tdf["R"].cumsum()
    return {
        "total": len(tdf),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_r": tdf["R"].sum(),
        "df": tdf
    }


def market_status():
    now = pd.Timestamp.now(TZ)
    hm = (now.hour, now.minute)
    if now.weekday() >= 5 or hm < (9, 30) or hm > (18, 10):
        return "closed", "Borsa şu an kapalı. Görünen rakamlar son işlem gününe aittir."
    if hm < (10, 15):
        return "early", "Seans yeni açıldı, fiyatlar çok oynak. Saat 10:15'i beklemek daha sağlıklıdır."
    if hm >= (17, 30):
        return "late", "Borsa kapanmak üzere. Günlük al-sat için yeni işlem açmayın."
    return "open", "Borsa açık. Fırsatlar her 15 dakikada bir otomatik yenilenir."


def bar(label, v):
    return f'<div class="bar"><span>{label}</span><div class="track"><div class="fill" style="width:{v:.0f}%;background:{color(v)}"></div></div><b>{v:.0f}</b></div>'


def build_plain_reason(r, is_swing):
    parts = [f"Hisse <b>%{r['Değişim']:+.1f}</b> hareketle yükseliş eğiliminde"]
    if r.get("Para", 0) >= 66:
        parts.append("hisseye belirgin bir <b>para girişi</b> var")
    if is_swing and r.get("Temel") is not None:
        if r["Temel"] >= 60:
            parts.append("şirketin <b>bilanço ve kârlılık verileri sağlam</b>")
        elif r["Temel"] < 45:
            parts.append("<b>bilanço verileri zayıf</b> olduğu için temkinli hedef belirlendi")
    return "💡 <b>Durum Özeti:</b> " + ", ".join(parts) + "."


def card(r, info, frows, mode_name, key_prefix="opp"):
    label, col = verdict(r["Genel"])
    is_swing = "Birkaç Günlük" in mode_name
    sector = info.get("sector") or info.get("industry") or ("Gün İçi Fırsat" if not is_swing else "Orta Vadeli Trend")
    is_live = r.get("IsLive", False)
    card_cls = "card-live" if is_live else "card"
    live_tag = '<span class="live-badge">⚡ ANLIK FİYATLA GÜNCELLENDİ</span>' if is_live else ""
    price_label = "Anlık Girilen Fiyat" if is_live else "Gecikmeli Fiyat (~15 dk)"

    html = (f'<div class="{card_cls}"><div class="top"><div><div class="tk">{r["Hisse"]}{live_tag}</div>'
            f'<div class="sub">{sector} • {price_label}: <b>{tl(r["Fiyat"])} TL</b> (%{r["Değişim"]:+.1f})</div></div>'
            f'<div class="pill" style="background:{col}">{label} • {r["Genel"]:.0f}/100</div></div>'
            f'<div class="summary-box">{build_plain_reason(r, is_swing)}<br><small style="opacity:.8">📐 <b>Plan Mantığı:</b> {r["PlanNote"]}</small></div>')

    html += ('<div class="grid">'
             f'<div class="box"><small>1. Alış Fiyatı</small><b>{tl(r["Fiyat"])} TL</b></div>'
             f'<div class="box"><small>2. Zarar Kes (Stop)</small><b>{tl(r["Stop"])} TL</b></div>'
             f'<div class="box"><small>3. İlk Kâr Hedefi</small><b>{tl(r["Hedef1"])} TL</b></div>'
             f'<div class="box"><small>4. İkinci Kâr Hedefi</small><b>{tl(r["Hedef2"])} TL</b></div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # --- CANLI FİYAT GİRİŞİ + TAKİBE AL SATIRI ---
    live_dict = st.session_state.setdefault("live_prices", {})
    c_live_in, c_live_btn, c_track_btn = st.columns([2.2, 1.3, 1.3])
    with c_live_in:
        entered_live = st.number_input(
            f"{r['Hisse']} Anlık Fiyat (TL)",
            min_value=0.0,
            value=float(live_dict.get(r["Hisse"], 0.0)),
            step=0.05,
            format="%.2f",
            key=f"inp_live_{key_prefix}_{mode_name}_{r['Hisse']}",
            label_visibility="collapsed",
            help="Aracı kurumundan gördüğün anlık fiyatı yazıp 'Fiyatı Uygula'ya bas; tüm teknik puan, stop, hedef ve adet anında yeniden hesaplansın."
        )
    with c_live_btn:
        if st.button("⚡ Anlık Fiyatı Uygula", key=f"btn_live_{key_prefix}_{mode_name}_{r['Hisse']}", use_container_width=True):
            if entered_live > 0:
                live_dict[r["Hisse"]] = entered_live
            else:
                live_dict.pop(r["Hisse"], None)
            st.rerun()
    with c_track_btn:
        if st.button("📝 Takibe Al", key=f"log_{key_prefix}_{mode_name}_{r['Hisse']}", use_container_width=True, type="primary"):
            add_trade(mode_name, r["Hisse"], r["Fiyat"], r["Stop"], r["Hedef1"], r["Hedef2"], r["Lot"])
            st.toast(f"✅ {r['Hisse']} ({tl(r['Fiyat'])} TL) takip listene eklendi!", icon="📌")

    # Uyarı / Özet Kutusu
    total_cost = r["Lot"] * r["Fiyat"]
    max_loss = r["Lot"] * r["Risk"]
    est_gain = r["Lot"] * (r["Hedef1"] - r["Fiyat"])

    if r["Puan"] < r["Toplam"] and is_live:
        failed_checks = [l for l, ok in r["Kontroller"] if not ok]
        st.error(
            f"🚨 **Anlık Fiyat Uyarısı:** Girdiğin **{tl(r['Fiyat'])} TL** fiyatıyla bu hisse **{len(failed_checks)} teknik kuralı bozdu!** "
            f"(Bozulan: *{failed_checks[0]}*). Alım yapmadan önce tekrar düşün."
        )
    elif r.get("IsWide"):
        st.warning(
            f"⚠️ **Oynaklık Yüksek:** Önerilen **{r['Lot']} adet** (~**{tl(total_cost)} TL**) • "
            f"Stop kaybı: **{tl(max_loss)} TL** • 1. hedef kârı: **{tl(est_gain)} TL**."
        )
    else:
        st.success(
            f"📌 **Önerilen Alım:** **{r['Lot']} adet** (Yaklaşık **{tl(total_cost)} TL**) • "
            f"Stop olursa kayıp: **{tl(max_loss)} TL** • "
            f"1. hedefte kâr: **{tl(est_gain)} TL**."
        )

    with st.expander(f"🔍 {r['Hisse']} Teknik ve Bilanço Detaylarını Gör"):
        bars_html = bar("Grafik Gücü", r["Tek"]) + bar("Para Girişi", r["Para"])
        if is_swing and r.get("Temel") is not None:
            bars_html += bar("Bilanço Sağlamlığı", r["Temel"])
        st.markdown(bars_html, unsafe_allow_html=True)
        st.markdown("**Kontrol Listesi (Anlık fiyata göre güncel):**")
        for l, ok in r["Kontroller"]:
            st.write(("✅ " if ok else "❌ ") + l)
        if is_swing and frows:
            st.markdown("**Şirket Bilanço Özeti:**")
            for n, v, rule, good in frows:
                st.write(f"{'✅' if good else '⬜'} {n}: **{v}** (İdeal: {rule})")


# ---------- Arayüz ----------
st.markdown(
    '<div class="hdr"><h1>📈 BIST Kolay Hisse Tarayıcı</h1>'
    '<p>Filtrelenen hisselerin aracı kurumundaki anlık fiyatını gir; sistem tüm teknik göstergeleri, puanı, stop ve hedefleri anında yeniden hesaplasın.</p></div>',
    unsafe_allow_html=True
)
st_autorefresh(interval=15 * 60 * 1000, key="auto")

live_prices = st.session_state.setdefault("live_prices", {})

# Ana Ekranda Sade 3 Adımlı Ayar Alanı
with st.container(border=True):
    st.markdown("#### ⚙️ 1. Adım: Kendi Planını Seç")
    col1, col2, col3 = st.columns(3)
    with col1:
        capital = st.number_input("Toplam Yatırım Bütçen (TL)", min_value=5_000, max_value=100_000_000, value=50_000, step=5_000)
    with col2:
        mode = st.selectbox("Ne Kadar Süre Tutacaksın?", [
            "⚡ Gün İçi Al-Sat (Aynı gün kapat)",
            "🌊 Birkaç Günlük / Haftalık (Swing)"
        ], index=1)
    with col3:
        prof = st.selectbox("Risk Tercihin Nedir?", list(PROFILES), index=1)

    with st.expander("🛠️ Gelişmiş Ayarlar & Anlık Fiyatları Sıfırla"):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            risk_pct = st.slider("Tek işlemde göze alınan maks. zarar (%)", 0.5, 2.5, 1.0, 0.25)
        with adv2:
            regime_strict = st.checkbox("Borsa düşerken uyarı ver (Endeks Koruması)", value=True)
        with adv3:
            tf = st.selectbox("Bilanço Filtresi (Çok günlük modda)", ["Kapalı", "Orta (Önerilen)", "Sıkı"], index=1)

        banned_input = st.text_input("Yasaklı / Brüt Takas Hisseleri (Otomatik hariç tutulur)",
                                     value=", ".join(sorted(DEFAULT_BANNED)))
        banned_set = {x.strip().upper() for x in banned_input.replace(",", " ").split() if x.strip()}

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🔄 Borsa Verilerini Yenile", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with b_col2:
            if st.button("🧹 Girilen Anlık Fiyatları Temizle", use_container_width=True):
                st.session_state["live_prices"] = {}
                st.rerun()

rv, c_lo, c_hi, r_lo, r_hi, min_tl, min_atr = PROFILES[prof]
params = dict(capital=capital, risk_pct=risk_pct, rvol=rv, chg_lo=c_lo, chg_hi=c_hi, rsi_lo=r_lo, rsi_hi=r_hi)
min_temel = {"Kapalı": 0, "Orta (Önerilen)": 50, "Sıkı": 65}[tf] if "Birkaç Günlük" in mode else 0

state, msg = market_status()
xu15, xu1d = get_xu100()
regime = check_market_regime(xu15, xu1d)

s1, s2 = st.columns(2)
with s1:
    {"open": st.success, "early": st.warning, "late": st.warning, "closed": st.info}[state]("🕒 " + msg)
with s2:
    if regime["ok"]:
        st.success("📊 " + regime["msg"])
    else:
        st.error("🛑 " + regime["msg"])

daily = get_daily(tuple(UNIVERSE))
dtab = daily_table(daily, min_tl, min_atr, banned_set) if daily else pd.DataFrame()
watch = dtab[dtab["Geçti"]]["Hisse"].tolist() if not dtab.empty else []

if "Birkaç Günlük" in mode and watch:
    top_watch = dtab[dtab["Geçti"]].sort_values("İşlem (M TL)", ascending=False)["Hisse"].tolist()[:40]
    fund = get_fund(tuple(top_watch))
    fs = {t: fund_score(fund.get(t, {})) for t in watch}
else:
    fund = {}
    fs = {t: (None, 0, []) for t in watch}

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Günün Fırsatları",
    "📒 Takip Listem & İşlemlerim",
    "📖 Nasıl Kullanılır?",
    "🧪 Geçmiş Başarı Testi"
])

PARA_TXT = {0: "Para girişi zayıf", 1: "Para girişi zayıf", 2: "Para girişi orta", 3: "Para girişi güçlü"}

with tab1:
    # --- İSTEDİĞİN HİSSEYİ SOR & ANLIK FİYATLA HESAPLA ---
    with st.container(border=True):
        st.markdown("#### 🔎 İstediğin Hisse İçin Teknik & Temel Bazlı Plan Oluştur")
        q1, q2, q3 = st.columns([2.2, 1.8, 1])
        with q1:
            custom_ticker = st.text_input("Hisse Kodu (Örn: THYAO)", placeholder="Örn: ASELS").strip().upper()
        with q2:
            custom_live_price = st.number_input("Anlık Fiyat (TL) - İsteğe Bağlı", min_value=0.0, value=0.0, step=0.05, format="%.2f",
                                                help="0 bırakırsan gecikmeli veriyi kullanır; anlık fiyat yazarsan doğrudan o fiyattan hesaplar.")
        with q3:
            st.write("")
            st.write("")
            analyze_custom_btn = st.button("🎯 Plan Oluştur", use_container_width=True)

        if custom_ticker or analyze_custom_btn:
            if custom_ticker:
                if custom_live_price > 0:
                    live_prices[custom_ticker] = custom_live_price
                effective_live = live_prices.get(custom_ticker)

                with st.spinner(f"{custom_ticker} analiz ediliyor..."):
                    t_clean, d1d_c, d15_c, info_c = get_single_stock_data(custom_ticker)
                if d1d_c.empty or len(d1d_c) < 55:
                    st.error(f"❌ '{custom_ticker}' kodu için yeterli BIST verisi bulunamadı.")
                else:
                    daily_atr_c = float(atr(d1d_c).iloc[-1])
                    sc_c, _, frows_c = fund_score(info_c)
                    fp_c, _, _, _ = flow(d1d_c)
                    if "Gün İçi" in mode and not d15_c.empty:
                        item_c = analyze_intraday(t_clean, d15_c, params, daily_atr_c, live_price=effective_live)
                    else:
                        item_c = analyze_swing(t_clean, d1d_c, params, daily_atr_c, temel_puan=sc_c, live_price=effective_live)

                    if item_c:
                        item_c["Temel"] = sc_c
                        item_c["Para"] = fp_c / 3 * 100
                        item_c["ParaTxt"] = PARA_TXT[fp_c]
                        item_c["Genel"] = overall_intraday(item_c["Tek"], item_c["Para"]) if "Gün İçi" in mode else overall_swing(item_c["Tek"], item_c["Temel"], item_c["Para"])
                        card(item_c, info_c, frows_c, mode, key_prefix="custom")

    st.divider()

    if regime_strict and not regime["ok"]:
        st.warning("⚠️ **Dikkat:** Borsa İstanbul genelinde düşüş baskısı var. Piyasa yatışana kadar alım yapmamak paranızı korur.")

    if not watch:
        st.info("Şu an otomatik taramada kriterlere uyan hisse bulunamadı. Yukarıdaki kutudan kendi istediğin hisseyi yazıp analiz edebilirsin.")
    else:
        drow = dtab.set_index("Hisse")
        res = []
        if "Gün İçi" in mode:
            intra = get_intraday(tuple(watch))
            for t, d in intra.items():
                lp = live_prices.get(t)
                item = analyze_intraday(t, d, params, float(drow.loc[t, "DailyATR"]), live_price=lp)
                if item:
                    res.append(item)
        else:
            for t in watch:
                d = daily.get(t)
                if d is not None:
                    lp = live_prices.get(t)
                    sc_t = fs[t][0]
                    item = analyze_swing(t, d, params, float(drow.loc[t, "DailyATR"]), temel_puan=sc_t, live_price=lp)
                    if item:
                        res.append(item)

        for x in res:
            t = x["Hisse"]
            fpt = int(drow.loc[t, "ParaP"])
            x["Temel"] = fs[t][0]
            x["Para"] = fpt / 3 * 100
            x["ParaTxt"] = PARA_TXT[fpt]
            x["Genel"] = overall_intraday(x["Tek"], x["Para"]) if "Gün İçi" in mode else overall_swing(x["Tek"], x["Temel"], x["Para"])

        okt = lambda x: min_temel == 0 or (x["Temel"] is not None and x["Temel"] >= min_temel)

        # Anlık fiyat girilmiş hisseleri (puanı düşse bile) kullanıcının görebilmesi için listede tut!
        opp = sorted([x for x in res if (x["Puan"] == x["Toplam"] or x["IsLive"]) and okt(x)], key=lambda x: (-int(x["IsLive"]), -x["Genel"]))
        near = sorted([x for x in res if x["Puan"] == x["Toplam"] - 1 and not x["IsLive"] and okt(x)], key=lambda x: -x["Genel"])

        if not opp:
            st.info("Bugün otomatik taramada tüm şartları eksiksiz karşılayan fırsat çıkmadı. Yukarıdan kendi hisseni sorgulayabilir veya aşağıdaki izleme listesine bakabilirsin.")
        else:
            st.markdown(f"#### ✨ Öne Çıkan Fırsatlar (Anlık Fiyat Girip Yeniden Hesaplatabilirsin)")
            st.caption("💡 Kartın altındaki kutuya aracı kurumundan gördüğün **anlık fiyatı** yazıp **'⚡ Anlık Fiyatı Uygula'** butonuna basarsan; RSI, ortalamalar, puan, stop, hedef ve adet anında güncellenir.")
            for x in opp[:6]:
                t = x["Hisse"]
                card(x, fund.get(t, {}), fs[t][2], mode, key_prefix="opp")

        if near:
            st.markdown(f"#### 👀 İzleme Listesi ({len(near)} hisse - 1 şartı eksik, anlık fiyatını girip test edebilirsin)")
            for x in near[:5]:
                t = x["Hisse"]
                card(x, fund.get(t, {}), fs[t][2], mode, key_prefix="near")

with tab2:
    st.subheader("📒 Takibe Aldığım Hisseler & İşlemlerim")
    st.write("Günün Fırsatları sekmesinden **'📝 Takibe Al'** dediğinde hisseler, teknik & temel bazlı Stop/Hedef seviyeleriyle buraya kaydedilir.")

    with st.container(border=True):
        st.markdown("**➕ Takip Listene Doğrudan Hisse Ekle (Otomatik Teknik & Temel Stop/Hedef Hesaplar)**")
        a1, a2, a3 = st.columns([2, 2, 1])
        with a1:
            add_tk = st.text_input("Hisse Kodu (Örn: GARAN)", key="add_tk_input").strip().upper()
        with a2:
            custom_entry = st.number_input("Anlık / Alış Fiyatın (TL) (0 = gecikmeli son fiyat)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
        with a3:
            st.write("")
            st.write("")
            if st.button("➕ Hesapla ve Ekle", use_container_width=True, type="primary"):
                if add_tk:
                    with st.spinner(f"{add_tk} için teknik ve temel destek/hedefler hesaplanıyor..."):
                        t_clean, d1d_a, _, info_a = get_single_stock_data(add_tk)
                    if d1d_a.empty or len(d1d_a) < 30:
                        st.error("Hisse verisi bulunamadı. Kodu kontrol edin.")
                    else:
                        if custom_entry > 0:
                            d1d_a = apply_live_price_to_df(d1d_a, custom_entry)
                        daily_atr_a = float(atr(d1d_a).iloc[-1])
                        sc_a, _, _ = fund_score(info_a)
                        entry_a = custom_entry if custom_entry > 0 else float(d1d_a["Close"].iloc[-1])
                        ema20_a = float(ema(d1d_a["Close"], 20).iloc[-1])
                        recent_low_a = float(d1d_a["Low"].tail(5).min())
                        raw_sup_a = max(ema20_a, recent_low_a)
                        if raw_sup_a >= entry_a:
                            raw_sup_a = entry_a - 0.5 * daily_atr_a
                        plan_a = compute_smart_plan(entry_a, raw_sup_a, daily_atr_a, capital, risk_pct, sc_a, 75, is_swing=True)
                        add_trade(mode, t_clean, entry_a, plan_a["stop"], plan_a["target1"], plan_a["target2"], plan_a["lot"])
                        st.success(f"✅ {t_clean} eklendi! Alış: {tl(entry_a)} TL | Stop: {tl(plan_a['stop'])} TL | 1. Hedef: {tl(plan_a['target1'])} TL")
                        st.rerun()

    tdf = get_trades()
    if tdf.empty:
        st.info("Henüz takip listende bir hisse yok.")
    else:
        closed = tdf[tdf["status"] != "AÇIK"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Takip", len(tdf))
        c2.metric("Devam Eden", len(tdf[tdf["status"] == "AÇIK"]))
        if not closed.empty and closed["pnl"].notna().any():
            wins = len(closed[closed["pnl"] > 0])
            c3.metric("Başarı Oranı", f"%{wins / len(closed) * 100:.0f}")
            c4.metric("Toplam Kâr / Zarar", f"{tl(closed['pnl'].sum())} TL")
        else:
            c3.metric("Başarı Oranı", "-")
            c4.metric("Toplam Kâr / Zarar", "0,00 TL")

        show_tdf = tdf[["id", "created_at", "ticker", "entry", "stop", "target1", "target2", "lot", "status", "exit_price", "pnl"]].copy()
        show_tdf.columns = ["No", "Tarih", "Hisse", "Alış", "Zarar Kes (Stop)", "1. Hedef", "2. Hedef", "Adet", "Durum", "Satış", "Net Kâr/Zarar (TL)"]
        st.dataframe(show_tdf, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.markdown("**İşlemi Güncelle / Kapat / Sil**")
            u1, u2, u3, u4, u5 = st.columns([1, 1.5, 1.3, 1, 1])
            with u1:
                sel_id = st.selectbox("İşlem No", tdf["id"].tolist())
            with u2:
                new_status = st.selectbox("Sonuç", ["KÂR İLE KAPATILDI", "ZARAR KES OLDU", "AÇIK"])
            with u3:
                exit_p = st.number_input("Sattığın Fiyat (TL)", min_value=0.0, value=0.0, step=0.1)
            with u4:
                st.write("")
                st.write("")
                if st.button("Kaydet", use_container_width=True):
                    update_trade_status(sel_id, new_status, exit_p if exit_p > 0 else None)
                    st.rerun()
            with u5:
                st.write("")
                st.write("")
                if st.button("🗑️ Sil", use_container_width=True):
                    delete_trade(sel_id)
                    st.rerun()

with tab3:
    st.markdown("""
### 👋 Hiç Bilmeyenler İçin 4 Adımda Kullanım Rehberi

1. **Bütçeni ve Süreni Seç:** Üst kutudan borsaya ayırdığın parayı yaz. Yeni başlıyorsan **"Birkaç Günlük / Haftalık"** modunu seç.
2. **Anlık Fiyatı Gir ve Yeniden Hesaplat:** Ücretsiz veriler 15 dakika gecikmelidir. Sistemin bulduğu hissenin altındaki kutuya banka/aracı kurum uygulamanda gördüğün **anlık fiyatı** yaz ve **"⚡ Anlık Fiyatı Uygula"** butonuna bas. Sistem yeni fiyata göre:
   * Hissenin aşırı yükselip yükselmediğini (`RSI` ve günlük `%` değişimi) tekrar kontrol eder,
   * Ortalamaların (`EMA`, `VWAP`) üzerinde kalıp kalmadığını yeniden ölçer,
   * Yeni **Stop**, **1. Hedef**, **2. Hedef** ve **Adet (Lot)** miktarını anında günceller.
3. **'📝 Takibe Al' Butonuna Bas:** Şartlar hâlâ uygunsa tek tuşla **Takip Listem & İşlemlerim** sekmesine kaydet.
4. **Önce Stop, Sonra Hedef:** Alım yaptıktan sonra mutlaka tabloda yazan **Zarar Kes (Stop)** seviyesine satış emri gir.
""")

with tab4:
    st.subheader("🧪 Bu Strateji Geçmişte Ne Kazandırdı?")
    b1, b2 = st.columns(2)
    with b1:
        bt_sample = st.slider("Test edilecek büyük hisse sayısı", 15, 60, 30, 5)
    with b2:
        bt_days = st.slider("Ortalama elde tutma süresi (iş günü)", 3, 10, 5)

    if st.button("🚀 Son 1 Yılı Test Et"):
        if dtab.empty:
            st.error("Önce borsa verilerinin yüklenmesi gerekiyor.")
        else:
            liquid_tickers = dtab.sort_values("İşlem (M TL)", ascending=False)["Hisse"].tolist()[:bt_sample]
            with st.spinner("Son 1 yılın grafikleri tek tek hesaplanıyor..."):
                bt_res = run_backtest(daily, liquid_tickers, hold_days=bt_days, rr_target=2.0)
            if not bt_res:
                st.warning("Bu kriterlerle geçmişte işlem sinyali oluşmadı.")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Toplam İşlem Sayısı", bt_res["total"])
                m2.metric("Kârlı İşlem Oranı", f"%{bt_res['win_rate']:.1f}")
                m3.metric("Kazanç / Kayıp Gücü", f"{bt_res['profit_factor']:.2f}x")
                m4.metric("Net Performans (Risk Birimi)", f"{bt_res['net_r']:+.1f} R")
                st.line_chart(bt_res["df"]["Kümülatif Kazanç (R)"])

st.divider()
st.caption("⚠️ Bu uygulama yalnızca eğitim ve karar destek amaçlıdır, kesin yatırım tavsiyesi değildir.")
