from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

TZ = "Europe/Istanbul"
UNIVERSE = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "CCOLA", "CIMSA", "DOHOL",
    "EKGYO", "ENJSA", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HALKB", "HEKTS",
    "ISCTR", "KCHOL", "KONTR", "KRDMD", "MAVI", "MGROS", "ODAS", "OYAKC", "PETKM",
    "PGSUS", "SAHOL", "SASA", "SISE", "SOKM", "TAVHL", "TCELL", "THYAO", "TOASO",
    "TSKB", "TTKOM", "TUPRS", "ULKER", "VAKBN", "YKBNK", "AEFES", "AKSEN", "EUPWR",
    "GESAN", "SMRTG", "TKFEN", "ISMEN", "BRSAN",
]
try:
    _p = Path(__file__).parent / "hisseler.txt"
    _all = sorted(set(_p.read_text(encoding="utf-8").upper().split()))
    if len(_all) > 50:
        UNIVERSE = _all
except Exception:
    pass  # dosya yoksa BIST 50 listesiyle devam eder

# Profil: (min RVOL, değişim alt, değişim üst, RSI alt, RSI üst, min günlük hacim M TL, min oynaklık %)
PROFILES = {
    "🛡️ Temkinli": (2.0, 1.5, 4.0, 50, 65, 100, 2.5),
    "⚖️ Dengeli": (1.5, 1.5, 6.0, 50, 70, 100, 2.5),
    "🔥 Atak": (1.2, 1.0, 7.0, 50, 75, 50, 2.0),
}

st.set_page_config(page_title="BIST Fırsat Tarayıcı", page_icon="📈", layout="centered")


def tl(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
                d = raw[t + ".IS"].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(d) < 30:
                continue
            d.index = d.index.tz_localize("UTC").tz_convert(TZ) if d.index.tz is None else d.index.tz_convert(TZ)
            out[t] = d
    return out


@st.cache_data(ttl=3600, show_spinner="Tüm BIST hisseleri taranıyor (ilk açılışta 1-2 dk sürebilir)...")
def get_daily(tickers):
    return _download(list(tickers), "6mo", "1d")


@st.cache_data(ttl=240)
def get_intraday(tickers):
    return _download(list(tickers), "5d", "15m")


# ---------- 1. Aşama: Günlük eleme ----------
def daily_table(daily, min_tl_m, min_atr):
    today = pd.Timestamp.now(TZ).date()
    rows = []
    for t, d in daily.items():
        d = d[d.index.date < today]
        if len(d) < 25:
            continue
        c = d["Close"].iloc[-1]
        vol = (d["Close"] * d["Volume"]).tail(20).mean() / 1e6
        a = atr(d).iloc[-1] / c * 100
        up = c > ema(d["Close"], 20).iloc[-1]
        rows.append({"Hisse": t, "Kapanış (TL)": c, "Günlük işlem (M TL)": vol, "Günlük hareket %": a,
                     "Geçti": bool(vol >= min_tl_m and a >= min_atr and up and c >= 5)})
    return pd.DataFrame(rows)


# ---------- 2. Aşama: Gün içi analiz ----------
def analyze(t, d, p):
    d = d.copy()
    d["day"] = d.index.date
    last_day = d["day"].iloc[-1]
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    d["vwap"] = (tp * d["Volume"]).groupby(d["day"]).cumsum() / d.groupby("day")["Volume"].cumsum()
    d["cumv"] = d.groupby("day")["Volume"].cumsum()
    d["ema9"], d["ema20"] = ema(d["Close"], 9), ema(d["Close"], 20)
    d["rsi"], d["atr"] = rsi(d["Close"]), atr(d)

    today, prev = d[d["day"] == last_day], d[d["day"] < last_day]
    if prev.empty:
        return None
    n = len(today)
    pc = [g["cumv"].iloc[n - 1] for _, g in prev.groupby("day") if len(g) >= n]
    rvol = today["cumv"].iloc[-1] / np.mean(pc) if pc else np.nan
    chg = (d["Close"].iloc[-1] / prev["Close"].iloc[-1] - 1) * 100
    r = d.iloc[-1]

    checks = [
        (f"Gün ortalamasının üstünde ({tl(r['Close'])} > {tl(r['vwap'])})", r["Close"] > r["vwap"]),
        ("Kısa vadeli trend yukarı yönlü", r["ema9"] > r["ema20"] and r["Close"] > r["ema20"]),
        (f"Yükseliş tükenmemiş (bugün %{chg:+.1f})", p["chg_lo"] <= chg <= p["chg_hi"]),
        (f"Alıcı gücü sağlıklı (RSI {r['rsi']:.0f})", p["rsi_lo"] <= r["rsi"] <= p["rsi_hi"]),
        ("Hacim normalin üstünde" + (f" ({rvol:.1f} kat)" if not np.isnan(rvol) else ""),
         bool(rvol >= p["rvol"])),
    ]
    entry = r["Close"]
    stop = min(r["vwap"], r["ema20"], today["Low"].iloc[-1]) - 0.25 * r["atr"]
    risk = entry - stop
    lot = 0
    if risk > 0:
        lot = int(min(p["capital"] * p["risk_pct"] / 100 / risk, p["capital"] * 0.4 / entry))
    return {
        "Hisse": t, "Fiyat": entry, "Değişim": chg, "Son bar": d.index[-1].strftime("%d.%m %H:%M"),
        "Kontroller": checks, "Puan": sum(ok for _, ok in checks), "Toplam": len(checks),
        "Stop": stop, "Risk": risk, "Hedef1": entry + 1.5 * risk, "Hedef2": entry + 2.5 * risk,
        "Lot": lot, "Stop uygun": 0 < risk <= r["atr"], "_ts": d.index[-1], "RVOL": rvol,
    }


def market_status():
    now = pd.Timestamp.now(TZ)
    hm = (now.hour, now.minute)
    if now.weekday() >= 5 or hm < (9, 30) or hm > (18, 10):
        return "closed", "Borsa şu an kapalı. Ekrandaki bilgiler son seansa aittir."
    if hm < (10, 15):
        return "early", "Açılış saatleri çok dalgalıdır. 10:15'ten sonra tarama daha güvenilir."
    if hm >= (17, 30):
        return "late", "Kapanışa yaklaşıldı. Yeni işlem açma, açık pozisyonları kapatmayı düşün."
    return "open", "Tarama zamanı. Fırsatlar 15 dakikada bir güncellenir."


def data_problem(ts):
    now = pd.Timestamp.now(TZ)
    if market_status()[0] == "closed":
        return None
    if ts.date() != now.date():
        return "Bugüne ait veri gelmiyor. Sinyallere güvenme."
    age = (now - ts).total_seconds() / 60
    return f"Veri {age:.0f} dk önceye ait, bayat. Sinyallere güvenme." if age > 45 else None


def card(r, p):
    with st.container(border=True):
        a, b = st.columns([3, 2])
        a.subheader(r["Hisse"])
        b.markdown(f"**Uyum: {r['Puan']}/{r['Toplam']}**")
        st.caption(f"Son fiyat {tl(r['Fiyat'])} TL • Bugün %{r['Değişim']:+.1f} • Veri: {r['Son bar']}")
        for label, ok in r["Kontroller"]:
            st.write(("✅ " if ok else "⬜ ") + label)
        if not r["Stop uygun"] or r["Lot"] < 1:
            st.warning("Zarar durdurma mesafesi bu hisse için fazla geniş. Risk yüksek, bu fırsatı atla.")
            return
        c1, c2 = st.columns(2)
        c1.metric("Alım fiyatı (yaklaşık)", f"{tl(r['Fiyat'])} TL")
        c2.metric("Zarar durdur (stop)", f"{tl(r['Stop'])} TL")
        c3, c4 = st.columns(2)
        c3.metric("1. kâr hedefi", f"{tl(r['Hedef1'])} TL")
        c4.metric("2. kâr hedefi", f"{tl(r['Hedef2'])} TL")
        kayip = r["Lot"] * r["Risk"]
        kazanc = r["Lot"] / 2 * 1.5 * r["Risk"]
        st.info(
            f"**Alınacak adet: {r['Lot']}**\n\n"
            f"• Stop'a gelirse tahmini kaybın: **{tl(kayip)} TL**\n\n"
            f"• 1. hedefte yarısını satarsan tahmini kazancın: **{tl(kazanc)} TL**\n\n"
            f"• Güncel fiyat {tl(r['Fiyat'] * 1.01)} TL'nin üstündeyse **girme**, kovalama."
        )


# ---------- Arayüz ----------
st.title("📈 BIST Fırsat Tarayıcı")
st_autorefresh(interval=15 * 60 * 1000, key="auto")

with st.sidebar:
    st.header("Ayarların")
    capital = st.number_input("Kullanacağın sermaye (TL)", 10_000, 100_000_000, 200_000, 10_000)
    risk_pct = st.number_input("Tek işlemde kaybetmeyi göze aldığın tutar (sermayenin %'si)", 0.25, 3.0, 1.0, 0.25,
                               help="Yeni başlayanlar için %0,5-1 önerilir. Örn. 200.000 TL ve %1 = en fazla 2.000 TL kayıp.")
    prof = st.radio("Tarama profili", list(PROFILES), index=1,
                    help="Temkinli: az ama güçlü fırsat. Dengeli: standart. Atak: daha çok fırsat, daha çok risk.")
    if st.button("🔄 Şimdi yenile"):
        st.cache_data.clear()
        st.rerun()

rv, c_lo, c_hi, r_lo, r_hi, min_tl, min_atr = PROFILES[prof]
params = dict(capital=capital, risk_pct=risk_pct, rvol=rv, chg_lo=c_lo, chg_hi=c_hi, rsi_lo=r_lo, rsi_hi=r_hi)

state, msg = market_status()
{"open": st.success, "early": st.warning, "late": st.warning, "closed": st.info}[state](msg)
st.caption("⏱️ Fiyatlar yaklaşık 15 dk gecikmelidir. Alım yapmadan önce aracı kurum uygulamandan güncel fiyatı kontrol et.")

daily = get_daily(tuple(UNIVERSE))
dtab = daily_table(daily, min_tl, min_atr) if daily else pd.DataFrame()
watch = dtab[dtab["Geçti"]]["Hisse"].tolist() if not dtab.empty else []

st.caption(f"📊 {len(dtab)} / {len(UNIVERSE)} hissenin verisi alındı • {len(watch)} hisse bugünkü listeye girdi")

tab1, tab2, tab3 = st.tabs(["🎯 Fırsatlar", "📋 Listem", "📖 Nasıl kullanılır"])

with tab1:
    if not watch:
        st.info("Bugün için uygun hisse listesi oluşmadı veya veri alınamadı. Biraz sonra yenile.")
    else:
        intra = get_intraday(tuple(watch))
        res = [x for t, d in intra.items() if (x := analyze(t, d, params))]
        if not res:
            st.error("Gün içi veri alınamadı. Biraz sonra yenile.")
        else:
            prob = data_problem(max(x["_ts"] for x in res))
            if prob:
                st.error("⚠️ " + prob)
            opp = sorted([x for x in res if x["Puan"] == x["Toplam"]], key=lambda x: -(x["RVOL"] if x["RVOL"] == x["RVOL"] else 0))
            near = sorted([x for x in res if x["Puan"] == x["Toplam"] - 1], key=lambda x: -x["Puan"])
            seen = st.session_state.setdefault("seen", set())
            for x in opp:
                if x["Hisse"] not in seen:
                    st.toast(f"Yeni fırsat: {x['Hisse']}", icon="🔔")
                    seen.add(x["Hisse"])

            st.markdown(f"**{len(res)}** hisse tarandı • **{len(opp)}** fırsat • **{len(near)}** yakın takip")
            if state == "closed":
                st.caption("Borsa kapalı olduğu için aşağıdakiler son seansın görüntüsüdür, işlem için kullanma.")
            if not opp:
                st.info("Şu an tüm koşulları sağlayan hisse yok. Acele etme, işlem yapmamak da bir karardır.")
            for x in opp[:5]:
                card(x, params)
            if near:
                with st.expander(f"👀 Yakın takip ({len(near)}): 1 koşul eksik"):
                    for x in near[:6]:
                        eksik = [l for l, ok in x["Kontroller"] if not ok]
                        st.write(f"**{x['Hisse']}** → eksik: {eksik[0]}")

with tab2:
    st.write("Bugünün taramasına giren hisseler: yeterince işlem gören ve gün içinde hareket edecek kadar oynak olanlar.")
    if dtab.empty:
        st.error("Günlük veri alınamadı.")
    else:
        show = dtab[dtab["Geçti"]].drop(columns="Geçti").sort_values("Günlük hareket %", ascending=False)
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("""
**Bu uygulama ne yapar?** Her 15 dakikada yüzlerce veriyi senin yerine tarar, kısa vadeli işlem için uygun görünen hisseleri ve hazır bir plan (nereden al, nerede zararı kes, nerede kâr al, kaç adet) önerir.

**Adım adım kullanım**
1. Yan menüden (sol üstteki **>**) sermayeni ve riskini gir.
2. 10:15'ten sonra **Fırsatlar** sekmesine bak.
3. Bir fırsat çıkarsa aracı kurum uygulamandan **güncel fiyatı** kontrol et.
4. Fiyat kartta yazan sınırın içindeyse planı uygula. Önce **zarar durdur (stop) emrini** gir.
5. 1. hedefe gelince pozisyonun yarısını sat, kalanı 2. hedefe kadar taşı.
6. Gün sonunda (en geç 17:45) kalan pozisyonu kapat.

**Altın kurallar**
- Fırsat yoksa işlem yok.
- Stop'u asla geriye çekme veya kaldırma.
- Günde en fazla 2-3 işlem. Günlük kayıp sermayenin %2-3'ünü bulursa o gün dur.
- İlk 2-4 hafta gerçek para koymadan sadece izle veya çok küçük adetle dene.
""")
    for title, text in {
        "Zarar durdur (stop) nedir?": "Fiyat belirlediğin seviyeye düşerse zararı sınırlamak için hisseyi satma emridir. Kaybını önceden bilmeni sağlar.",
        "Gün ortalaması (VWAP) nedir?": "Gün içinde hissenin hacimle ağırlıklandırılmış ortalama fiyatıdır. Fiyat bunun üstündeyse alıcılar güçlüdür.",
        "Trend (EMA) nedir?": "Son fiyatların ortalamasından çizilen yön çizgisidir. Kısa çizgi uzun çizginin üstündeyse hisse yükseliş eğilimindedir.",
        "Alıcı gücü (RSI) nedir?": "0-100 arası bir güç ölçüsüdür. 50-70 sağlıklı yükselişi gösterir, 70 üstü aşırı ısınma olabilir.",
        "Hacim kat sayısı (RVOL) nedir?": "Hissenin bugünkü işlem hacminin normal günlere göre kaç kat olduğudur. 1,5 kat ve üstü ilgi arttığını gösterir.",
        "Kâr hedefi neye göre?": "Hedefler, stop mesafen kadar (1,5 ve 2,5 katı) hesaplanır. Böylece kazanç, riskinden büyük olur.",
    }.items():
        with st.expander(title):
            st.write(text)

st.divider()
st.caption("Bu uygulama eğitim ve karar destek aracıdır, yatırım tavsiyesi değildir. Kayıp riski her zaman vardır. Veriler hatalı veya gecikmeli olabilir.")
