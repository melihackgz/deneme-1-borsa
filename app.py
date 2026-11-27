import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

TZ = "Europe/Istanbul"

# BIST 50 ağırlıklı örnek evren. Kodlar değişirse buradan güncelle.
UNIVERSE = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "CCOLA", "CIMSA", "DOHOL",
    "EKGYO", "ENJSA", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HALKB", "HEKTS",
    "ISCTR", "KCHOL", "KONTR", "KRDMD", "MAVI", "MGROS", "ODAS", "OYAKC", "PETKM",
    "PGSUS", "SAHOL", "SASA", "SISE", "SOKM", "TAVHL", "TCELL", "THYAO", "TOASO",
    "TSKB", "TTKOM", "TUPRS", "ULKER", "VAKBN", "YKBNK", "AEFES", "AKSEN", "EUPWR",
    "GESAN", "SMRTG", "TKFEN", "ISMEN", "BRSAN",
]

st.set_page_config(page_title="BIST Gün İçi Tarayıcı", layout="wide")


# ---------- Göstergeler ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------- Veri ----------
def _download(tickers, period, interval):
    raw = yf.download([t + ".IS" for t in tickers], period=period, interval=interval,
                      group_by="ticker", progress=False, threads=True, auto_adjust=False)
    out = {}
    for t in tickers:
        try:
            d = raw[t + ".IS"].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(d) < 30:
            continue
        d.index = d.index.tz_localize("UTC").tz_convert(TZ) if d.index.tz is None else d.index.tz_convert(TZ)
        out[t] = d
    return out


@st.cache_data(ttl=3600)
def get_daily(tickers):
    return _download(list(tickers), "6mo", "1d")


@st.cache_data(ttl=240)
def get_intraday(tickers):
    return _download(list(tickers), "5d", "15m")


# ---------- Ana liste (günlük filtreler) ----------
def daily_table(daily, min_tl_m, min_atr):
    today = pd.Timestamp.now(TZ).date()
    rows = []
    for t, d in daily.items():
        d = d[d.index.date < today]  # sadece tamamlanmış günler
        if len(d) < 25:
            continue
        c = d["Close"].iloc[-1]
        tl = (d["Close"] * d["Volume"]).tail(20).mean() / 1e6
        a = atr(d).iloc[-1] / c * 100
        up = c > ema(d["Close"], 20).iloc[-1]
        rows.append({"Hisse": t, "Kapanış": c, "Ort. Hacim (M TL)": tl, "ATR %": a,
                     "Fiyat>EMA20": up, "Geçti": bool(tl >= min_tl_m and a >= min_atr and up)})
    return pd.DataFrame(rows)


# ---------- Gün içi analiz ----------
def analyze(t, d, p):
    d = d.copy()
    d["day"] = d.index.date
    last_day = d["day"].iloc[-1]
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    d["pv"] = tp * d["Volume"]
    d["vwap"] = d.groupby("day")["pv"].cumsum() / d.groupby("day")["Volume"].cumsum()
    d["cumv"] = d.groupby("day")["Volume"].cumsum()
    d["ema9"], d["ema20"] = ema(d["Close"], 9), ema(d["Close"], 20)
    d["rsi"], d["atr"] = rsi(d["Close"]), atr(d)

    today = d[d["day"] == last_day]
    prev_days = d[d["day"] < last_day]
    if prev_days.empty:
        return None
    n = len(today)
    prev_cum = [g["cumv"].iloc[n - 1] for _, g in prev_days.groupby("day") if len(g) >= n]
    rvol = today["cumv"].iloc[-1] / np.mean(prev_cum) if prev_cum else np.nan
    prev_close = prev_days["Close"].iloc[-1]

    r = d.iloc[-1]
    chg = (r["Close"] / prev_close - 1) * 100
    checks = {
        "Fiyat>VWAP": r["Close"] > r["vwap"],
        "Fiyat>EMA20": r["Close"] > r["ema20"],
        "EMA9>EMA20": r["ema9"] > r["ema20"],
        f"RSI {p['rsi_lo']}-{p['rsi_hi']}": p["rsi_lo"] <= r["rsi"] <= p["rsi_hi"],
        f"Değişim {p['chg_lo']}-{p['chg_hi']}%": p["chg_lo"] <= chg <= p["chg_hi"],
        f"RVOL≥{p['rvol']}": bool(rvol >= p["rvol"]),
    }
    missing = [k for k, v in checks.items() if not v]

    entry = r["Close"]
    stop = min(r["vwap"], r["ema20"], today["Low"].iloc[-1]) - 0.25 * r["atr"]
    risk = entry - stop
    stop_ok = 0 < risk <= r["atr"]
    lot = 0
    if risk > 0:
        lot = int(min(p["capital"] * p["risk_pct"] / 100 / risk, p["capital"] / entry))
    return {
        "Hisse": t, "Son bar": d.index[-1].strftime("%d.%m %H:%M"), "Fiyat": entry,
        "Değişim %": chg, "RVOL": rvol, "RSI": r["rsi"], "VWAP": r["vwap"], "EMA20": r["ema20"],
        "Stop": stop, "Hedef 1.5R": entry + 1.5 * risk, "Hedef 2.5R": entry + 2.5 * risk,
        "Lot": lot, "Stop uygun": stop_ok, "Eksik": ", ".join(missing),
        "Sinyal": not missing, "_ts": d.index[-1],
    }


def data_warning(ts):
    now = pd.Timestamp.now(TZ)
    in_session = now.weekday() < 5 and (10, 0) <= (now.hour, now.minute) <= (18, 10)
    if not in_session:
        return "Seans dışı: veriler son seansa aittir."
    if ts.date() != now.date():
        return "Bugüne ait veri yok (tatil veya veri kopması olabilir)."
    age = (now - ts).total_seconds() / 60
    if age > 45:
        return f"Veri bayat: son bar {age:.0f} dk önce. Sinyallere güvenme."
    return None


# ---------- Arayüz ----------
st.title("📈 BIST Gün İçi Tarayıcı")
st.caption("Veri: yfinance (~15 dk gecikmeli). Aday bulur; girişi canlı fiyatı teyit ederek sen yaparsın. Yatırım tavsiyesi değildir.")
st_autorefresh(interval=15 * 60 * 1000, key="auto")

with st.sidebar:
    st.header("Ayarlar")
    capital = st.number_input("Sermaye (TL)", 10_000, 100_000_000, 200_000, 10_000)
    risk_pct = st.number_input("İşlem başı risk (%)", 0.1, 5.0, 1.0, 0.1)
    st.subheader("Günlük filtre")
    min_tl = st.number_input("Min ort. hacim (M TL)", 10, 1000, 100, 10)
    min_atr = st.number_input("Min ATR/Fiyat (%)", 0.5, 10.0, 2.5, 0.1)
    st.subheader("Gün içi filtre")
    rvol_min = st.number_input("Min RVOL", 1.0, 5.0, 1.5, 0.1)
    chg_lo, chg_hi = st.slider("Günlük değişim (%)", 0.0, 10.0, (1.5, 6.0), 0.5)
    rsi_lo, rsi_hi = st.slider("RSI aralığı", 30, 90, (50, 70))
    if st.button("Şimdi yenile"):
        st.cache_data.clear()
        st.rerun()

params = dict(capital=capital, risk_pct=risk_pct, rvol=rvol_min, chg_lo=chg_lo, chg_hi=chg_hi,
              rsi_lo=rsi_lo, rsi_hi=rsi_hi)

now = pd.Timestamp.now(TZ)
st.write(f"Şu an: **{now.strftime('%d.%m.%Y %H:%M')}** (İstanbul)")
if now.weekday() < 5 and (9, 0) <= (now.hour, now.minute) < (10, 15):
    st.info("Açılış volatilitesi: taramayı 10:15-10:30 sonrası değerlendir.")
if now.weekday() < 5 and (now.hour, now.minute) >= (17, 30):
    st.warning("17:30 geçti: açık pozisyonları kapatmayı düşün, kapanışa taşıma.")

daily = get_daily(tuple(UNIVERSE))
dt_ = daily_table(daily, min_tl, min_atr) if daily else pd.DataFrame()
watch = dt_[dt_["Geçti"]]["Hisse"].tolist() if not dt_.empty else []

tab1, tab2 = st.tabs(["Canlı tarama", "Ana liste"])

with tab2:
    if dt_.empty:
        st.error("Günlük veri alınamadı.")
    else:
        st.write(f"Ana listeye giren: **{len(watch)}** / {len(dt_)}")
        st.dataframe(dt_.sort_values("ATR %", ascending=False).round(2), use_container_width=True, hide_index=True)

with tab1:
    if not watch:
        st.info("Ana liste boş. Günlük filtreleri gevşetmeyi dene.")
    else:
        intra = get_intraday(tuple(watch))
        res = [x for t, d in intra.items() if (x := analyze(t, d, params))]
        if not res:
            st.error("Gün içi veri alınamadı.")
        else:
            df = pd.DataFrame(res)
            warn = data_warning(df["_ts"].max())
            if warn:
                st.error(warn)
            sig = df[df["Sinyal"]].sort_values("RVOL", ascending=False)

            seen = st.session_state.setdefault("seen", set())
            for t in sig["Hisse"]:
                if t not in seen:
                    st.toast(f"Yeni aday: {t}", icon="🔔")
                    seen.add(t)

            cols = ["Hisse", "Fiyat", "Değişim %", "RVOL", "RSI", "VWAP", "Stop", "Hedef 1.5R", "Hedef 2.5R", "Lot", "Stop uygun", "Son bar"]
            st.subheader(f"Adaylar ({len(sig)})")
            if sig.empty:
                st.write("Şu an tüm filtreleri geçen hisse yok.")
            else:
                st.dataframe(sig[cols].round(2), use_container_width=True, hide_index=True)
                st.caption("Stop uygun = stop mesafesi ≤ 1 ATR(15dk). Değilse girme. Girişte canlı fiyat bu seviyeden %0.5-1'den fazla uzaksa kovalama.")

            with st.expander("Filtreyi geçemeyenler (neden?)"):
                st.dataframe(df[~df["Sinyal"]][["Hisse", "Fiyat", "Değişim %", "RVOL", "RSI", "Eksik"]].round(2),
                             use_container_width=True, hide_index=True)
