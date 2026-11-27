from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

TZ = "Europe/Istanbul"
UNIVERSE = ["AKBNK", "ARCLK", "ASELS", "BIMAS", "EREGL", "FROTO", "GARAN", "ISCTR", "KCHOL", "SAHOL",
            "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"]
try:
    _all = sorted(set((Path(__file__).parent / "hisseler.txt").read_text(encoding="utf-8").upper().split()))
    if len(_all) > 50:
        UNIVERSE = _all
except Exception:
    pass

# Profil: (min RVOL, değişim alt, üst, RSI alt, üst, min günlük hacim M TL, min günlük hareket %)
PROFILES = {"🛡️ Temkinli": (2.0, 1.5, 4.0, 50, 65, 100, 2.5),
            "⚖️ Dengeli": (1.5, 1.5, 6.0, 50, 70, 100, 2.5),
            "🔥 Atak": (1.2, 1.0, 7.0, 50, 75, 50, 2.0)}

st.set_page_config(page_title="BIST Pro Tarayıcı", page_icon="📈", layout="wide")
st.markdown("""<style>
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1100px;padding-top:1.2rem}
.hdr{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;padding:18px 22px;border-radius:14px;margin-bottom:12px}
.hdr h1{margin:0;font-size:1.5rem;color:#fff;padding:0}.hdr p{margin:4px 0 0;opacity:.8;font-size:.85rem}
.card{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:14px 16px;margin:10px 0 4px;background:rgba(128,128,128,.06)}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.tk{font-size:1.45rem;font-weight:800}.sub{opacity:.7;font-size:.8rem}
.pill{padding:4px 12px;border-radius:999px;font-weight:700;font-size:.78rem;color:#fff;white-space:nowrap}
.tag{display:inline-block;padding:2px 9px;border-radius:8px;font-size:.72rem;margin:6px 6px 0 0;background:rgba(128,128,128,.18)}
.bar{display:flex;align-items:center;gap:8px;font-size:.8rem;margin:6px 0}.bar span{width:96px;opacity:.85}
.track{flex:1;height:8px;border-radius:6px;background:rgba(128,128,128,.25);overflow:hidden}.fill{height:100%;border-radius:6px}
.bar b{width:30px;text-align:right}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:10px}
@media(min-width:700px){.grid{grid-template-columns:repeat(4,1fr)}}
.box{border-radius:10px;padding:8px 10px;background:rgba(128,128,128,.12)}
.box small{opacity:.7;display:block;font-size:.7rem}.box b{font-size:1rem}
</style>""", unsafe_allow_html=True)


def tl(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def ok_num(x):
    return x is not None and isinstance(x, (int, float)) and x == x


def color(v):
    return "#16a34a" if v >= 70 else "#d97706" if v >= 50 else "#dc2626"


# ---------- Göstergeler ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    return 100 - 100 / (1 + up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean())


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def flow(d):
    """Para girişi göstergesi (tahmini): CMF, OBV, hacimli alım/satım günleri."""
    d = d.tail(60)
    hl = (d["High"] - d["Low"]).replace(0, np.nan)
    mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / hl
    cmf = float((mfm * d["Volume"]).tail(20).sum() / d["Volume"].tail(20).sum())
    obv = (np.sign(d["Close"].diff()).fillna(0) * d["Volume"]).cumsum()
    obv_up = bool(obv.iloc[-1] > obv.rolling(20).mean().iloc[-1])
    big = d["Volume"] > 1.5 * d["Volume"].rolling(20).mean().shift()
    acc = int((big & (d["Close"] > d["Open"])).tail(10).sum())
    dist = int((big & (d["Close"] < d["Open"])).tail(10).sum())
    pts = int(cmf > 0.05) + int(obv_up) + int(acc > dist)
    return pts, cmf, acc, dist


def fund_score(i):
    """Temel puan 0-100. Veri olmayan kalemler hesaba katılmaz."""
    items = [
        ("F/K", i.get("trailingPE"), 20, lambda v: 0 < v <= 20, lambda v: f"{v:.1f}", "≤ 20"),
        ("PD/DD", i.get("priceToBook"), 15, lambda v: 0 < v <= 3, lambda v: f"{v:.2f}", "≤ 3"),
        ("Özkaynak kârlılığı (ROE)", i.get("returnOnEquity"), 25, lambda v: v >= 0.15, lambda v: f"%{v*100:.0f}", "≥ %15"),
        ("Net kâr marjı", i.get("profitMargins"), 15, lambda v: v >= 0.08, lambda v: f"%{v*100:.0f}", "≥ %8"),
        ("Kâr büyümesi", i.get("earningsGrowth"), 10, lambda v: v > 0, lambda v: f"%{v*100:.0f}", "> 0"),
        ("Borç/Özkaynak", i.get("debtToEquity"), 15, lambda v: v <= 100, lambda v: f"%{v:.0f}", "≤ %100"),
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


def overall(tek, temel, para):
    w = {"t": 0.40, "f": 0.35, "p": 0.25}
    parts = {"t": tek, "p": para}
    if temel is not None:
        parts["f"] = temel
    tot = sum(w[k] for k in parts)
    return sum(w[k] * v for k, v in parts.items()) / tot


def verdict(g):
    return ("GÜÇLÜ", "#16a34a") if g >= 75 else ("İYİ", "#2563eb") if g >= 60 else ("DİKKAT", "#d97706")


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


@st.cache_data(ttl=43200, show_spinner="Temel veriler alınıyor...")
def get_fund(tickers):
    def one(t):
        try:
            return t, yf.Ticker(t + ".IS").info or {}
        except Exception:
            return t, {}
    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, tickers))


def daily_table(daily, min_tl_m, min_atr):
    today = pd.Timestamp.now(TZ).date()
    rows = []
    for t, d in daily.items():
        d = d[d.index.date < today]
        if len(d) < 30:
            continue
        c = d["Close"].iloc[-1]
        vol = (d["Close"] * d["Volume"]).tail(20).mean() / 1e6
        a = atr(d).iloc[-1] / c * 100
        up = c > ema(d["Close"], 20).iloc[-1]
        fp, cmf, acc, dist = flow(d)
        rows.append({"Hisse": t, "Kapanış": c, "İşlem (M TL)": vol, "Hareket %": a, "ParaP": fp, "CMF": cmf,
                     "Alım günü": acc, "Satım günü": dist,
                     "Geçti": bool(vol >= min_tl_m and a >= min_atr and up and c >= 5)})
    return pd.DataFrame(rows)


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
        "Hacim normalin üstünde" + (f" ({rvol:.1f} kat)" if rvol == rvol else ""),
    ]
    checks[4] = (checks[4], bool(rvol >= p["rvol"]))
    entry = r["Close"]
    stop = min(r["vwap"], r["ema20"], today["Low"].iloc[-1]) - 0.25 * r["atr"]
    risk = entry - stop
    lot = int(min(p["capital"] * p["risk_pct"] / 100 / risk, p["capital"] * 0.4 / entry)) if risk > 0 else 0
    return {"Hisse": t, "Fiyat": entry, "Değişim": chg, "Son bar": d.index[-1].strftime("%d.%m %H:%M"),
            "Kontroller": checks, "Puan": sum(ok for _, ok in checks), "Toplam": len(checks), "Stop": stop,
            "Risk": risk, "Hedef1": entry + 1.5 * risk, "Hedef2": entry + 2.5 * risk, "Lot": lot,
            "Stop uygun": 0 < risk <= r["atr"], "_ts": d.index[-1], "RVOL": rvol}


def market_status():
    now = pd.Timestamp.now(TZ)
    hm = (now.hour, now.minute)
    if now.weekday() >= 5 or hm < (9, 30) or hm > (18, 10):
        return "closed", "Borsa kapalı. Ekrandaki bilgiler son seansa aittir."
    if hm < (10, 15):
        return "early", "Açılış çok dalgalıdır. 10:15'ten sonra tarama daha güvenilir."
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


def bar(label, v):
    return f'<div class="bar"><span>{label}</span><div class="track"><div class="fill" style="width:{v:.0f}%;background:{color(v)}"></div></div><b>{v:.0f}</b></div>'


def card(r, info, frows, kurum):
    label, col = verdict(r["Genel"])
    sector = info.get("sector") or info.get("industry") or ""
    tags = ""
    if kurum is not None:
        tags += f'<span class="tag">🏦 Kurum {"net alıcı" if kurum > 0 else "net satıcı"}</span>'
    ip = info.get("heldPercentInstitutions")
    if ok_num(ip):
        tags += f'<span class="tag">Kurumsal pay %{ip*100:.0f}</span>'
    tags += f'<span class="tag">{r["ParaTxt"]}</span>'
    html = (f'<div class="card"><div class="top"><div><div class="tk">{r["Hisse"]}</div>'
            f'<div class="sub">{sector} • {tl(r["Fiyat"])} TL • bugün %{r["Değişim"]:+.1f}</div></div>'
            f'<div class="pill" style="background:{col}">{label} {r["Genel"]:.0f}</div></div>{tags}'
            + bar("Teknik", r["Tek"])
            + (bar("Temel", r["Temel"]) if r["Temel"] is not None else '<div class="sub">Temel veri bulunamadı</div>')
            + bar("Para girişi", r["Para"]))
    if r["Stop uygun"] and r["Lot"] >= 1:
        html += ('<div class="grid">'
                 f'<div class="box"><small>Alım (yaklaşık)</small><b>{tl(r["Fiyat"])}</b></div>'
                 f'<div class="box"><small>Zarar durdur</small><b>{tl(r["Stop"])}</b></div>'
                 f'<div class="box"><small>1. hedef</small><b>{tl(r["Hedef1"])}</b></div>'
                 f'<div class="box"><small>2. hedef</small><b>{tl(r["Hedef2"])}</b></div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    if r["Stop uygun"] and r["Lot"] >= 1:
        st.info(f"**Adet: {r['Lot']}** • Stop'ta tahmini kayıp **{tl(r['Lot'] * r['Risk'])} TL** • "
                f"1. hedefte yarısını satarsan **{tl(r['Lot'] / 2 * 1.5 * r['Risk'])} TL** kazanç • "
                f"Fiyat {tl(r['Fiyat'] * 1.01)} TL üstündeyse girme.")
    else:
        st.warning("Zarar durdurma mesafesi bu hisse için fazla geniş. Risk yüksek, bu fırsatı atla.")
    with st.expander(f"{r['Hisse']} detayları"):
        st.markdown("**Teknik kontroller**")
        for l, ok in r["Kontroller"]:
            st.write(("✅ " if ok else "⬜ ") + l)
        st.markdown("**Temel göstergeler**")
        if frows:
            for n, v, rule, good in frows:
                st.write(f"{'✅' if good else '⬜'} {n}: **{v}** (hedef {rule})")
        else:
            st.write("Bu hisse için yeterli temel veri alınamadı.")
        st.markdown("**Para girişi (tahmini)**")
        st.write(r["ParaDetay"])


# ---------- Arayüz ----------
st.markdown('<div class="hdr"><h1>📈 BIST Pro Tarayıcı</h1><p>Teknik + Temel + Para girişi tek ekranda • tüm BIST</p></div>',
            unsafe_allow_html=True)
st_autorefresh(interval=15 * 60 * 1000, key="auto")

with st.sidebar:
    st.header("Ayarların")
    capital = st.number_input("Sermaye (TL)", 10_000, 100_000_000, 200_000, 10_000)
    risk_pct = st.number_input("Tek işlemde göze aldığın kayıp (%)", 0.25, 3.0, 1.0, 0.25,
                               help="Yeni başlayanlar için %0,5-1. Örn. 200.000 TL ve %1 = en fazla 2.000 TL kayıp.")
    prof = st.radio("Tarama profili", list(PROFILES), index=1)
    tf = st.radio("Temel filtre", ["Kapalı", "Orta (≥50)", "Sıkı (≥65)"], index=1,
                  help="Temel puanı düşük hisseleri fırsat listesinden çıkarır.")
    up = st.file_uploader("Kurum/yabancı net alım dosyası (CSV, isteğe bağlı)", type=["csv"],
                          help="Aracı kurumundan aldığın veriyi 'Hisse,NetAlim' sütunlarıyla yükle. Pozitif = net alım.")
    if st.button("🔄 Şimdi yenile"):
        st.cache_data.clear()
        st.rerun()

kurum_map = {}
if up is not None:
    try:
        kd = pd.read_csv(up, sep=None, engine="python")
        kd.columns = [c.strip().lower() for c in kd.columns]
        kurum_map = {str(h).strip().upper(): float(v) for h, v in zip(kd.iloc[:, 0], kd.iloc[:, 1])}
    except Exception:
        st.sidebar.error("Dosya okunamadı. 'Hisse,NetAlim' sütunları olmalı.")

rv, c_lo, c_hi, r_lo, r_hi, min_tl, min_atr = PROFILES[prof]
params = dict(capital=capital, risk_pct=risk_pct, rvol=rv, chg_lo=c_lo, chg_hi=c_hi, rsi_lo=r_lo, rsi_hi=r_hi)
min_temel = {"Kapalı": 0, "Orta (≥50)": 50, "Sıkı (≥65)": 65}[tf]

state, msg = market_status()
{"open": st.success, "early": st.warning, "late": st.warning, "closed": st.info}[state](msg)
st.caption("⏱️ Fiyatlar ~15 dk gecikmelidir. Alım öncesi aracı kurum uygulamandan güncel fiyatı kontrol et.")

daily = get_daily(tuple(UNIVERSE))
dtab = daily_table(daily, min_tl, min_atr) if daily else pd.DataFrame()
watch = dtab[dtab["Geçti"]]["Hisse"].tolist() if not dtab.empty else []
fund = get_fund(tuple(watch)) if watch else {}
fs = {t: fund_score(fund.get(t, {})) for t in watch}
st.caption(f"📊 {len(dtab)} / {len(UNIVERSE)} hissenin verisi alındı • {len(watch)} hisse bugünkü listeye girdi")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Fırsatlar", "🏛️ Temel & Para Akışı", "📋 Liste", "📖 Rehber"])

PARA_TXT = {0: "Para girişi zayıf", 1: "Para girişi zayıf", 2: "Para girişi orta", 3: "Para girişi güçlü"}
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
            drow = dtab.set_index("Hisse")
            for x in res:
                t = x["Hisse"]
                fpt = int(drow.loc[t, "ParaP"])
                x["Tek"] = x["Puan"] / x["Toplam"] * 100
                x["Temel"] = fs[t][0]
                x["Para"] = fpt / 3 * 100
                x["ParaTxt"] = PARA_TXT[fpt]
                x["ParaDetay"] = (f"Para akışı göstergesi (CMF) {drow.loc[t, 'CMF']:+.2f}; son 10 günde hacimli "
                                  f"yükseliş günü {int(drow.loc[t, 'Alım günü'])}, hacimli düşüş günü {int(drow.loc[t, 'Satım günü'])}. "
                                  "Bu veri fiyat ve hacimden tahmin edilir, gerçek fon/kurum alımı değildir.")
                x["Genel"] = overall(x["Tek"], x["Temel"], x["Para"])
            okt = lambda x: min_temel == 0 or (x["Temel"] is not None and x["Temel"] >= min_temel)
            opp = sorted([x for x in res if x["Puan"] == x["Toplam"] and okt(x)], key=lambda x: -x["Genel"])
            near = [x for x in res if x["Puan"] == x["Toplam"] - 1 and okt(x)]
            seen = st.session_state.setdefault("seen", set())
            for x in opp:
                if x["Hisse"] not in seen:
                    st.toast(f"Yeni fırsat: {x['Hisse']}", icon="🔔")
                    seen.add(x["Hisse"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Taranan", len(res))
            m2.metric("Fırsat", len(opp))
            m3.metric("Yakın takip", len(near))
            if state == "closed":
                st.caption("Borsa kapalı: aşağıdakiler son seansın görüntüsüdür, işlem için kullanma.")
            if not opp:
                st.info("Şu an teknik, temel ve filtre koşullarını birlikte sağlayan hisse yok. İşlem yapmamak da bir karardır.")
            for x in opp[:5]:
                t = x["Hisse"]
                card(x, fund.get(t, {}), fs[t][2], kurum_map.get(t))
            if near:
                with st.expander(f"👀 Yakın takip ({len(near)}): 1 teknik koşul eksik"):
                    for x in near[:8]:
                        st.write(f"**{x['Hisse']}** (Genel {x['Genel']:.0f}) → eksik: {[l for l, ok in x['Kontroller'] if not ok][0]}")

with tab2:
    st.write("Bugünkü listedeki hisselerin temel ve para girişi özeti. Sütun başlığına dokunup sıralayabilirsin.")
    if not watch:
        st.info("Liste boş.")
    else:
        rows = []
        for t in watch:
            i, (sc, cov, _) = fund.get(t, {}), fs[t]
            drow = dtab.set_index("Hisse").loc[t]
            kv = kurum_map.get(t)
            rows.append({"Hisse": t, "Temel puan": sc, "F/K": i.get("trailingPE"), "PD/DD": i.get("priceToBook"),
                         "ROE %": i.get("returnOnEquity") * 100 if ok_num(i.get("returnOnEquity")) else None,
                         "Net marj %": i.get("profitMargins") * 100 if ok_num(i.get("profitMargins")) else None,
                         "Para girişi": PARA_TXT[int(drow["ParaP"])].replace("Para girişi ", ""),
                         "CMF": drow["CMF"],
                         "Kurumsal pay %": i.get("heldPercentInstitutions") * 100 if ok_num(i.get("heldPercentInstitutions")) else None,
                         "Kurum notu": None if kv is None else ("Net alıcı" if kv > 0 else "Net satıcı")})
        tdf = pd.DataFrame(rows).sort_values("Temel puan", ascending=False, na_position="last")
        st.dataframe(tdf.round(2), use_container_width=True, hide_index=True)
        st.caption("Enflasyon muhasebesi ve banka/holding yapıları nedeniyle F/K ve büyüme rakamları yanıltıcı olabilir. Temel puanı tek başına karar için kullanma.")

with tab3:
    if dtab.empty:
        st.error("Günlük veri alınamadı.")
    else:
        show = dtab[dtab["Geçti"]].drop(columns=["Geçti", "ParaP"]).sort_values("Hareket %", ascending=False)
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)

with tab4:
    st.markdown("""
**Üç puan nasıl okunur?**
- **Teknik:** Fiyatın bugünkü gücü (gün ortalaması, trend, hacim, momentum). Fırsat olması için hepsi sağlanmalıdır.
- **Temel:** Şirketin değerlemesi, kârlılığı ve borçluluğu (F/K, PD/DD, ROE, net marj, kâr büyümesi, borç). 100 üzerinden puanlanır.
- **Para girişi:** Hisseye para girip girmediğinin **tahmini**. Fiyat ve hacimden hesaplanır.

**Genel puan** = %40 teknik + %35 temel + %25 para girişi. 75+ güçlü, 60-75 iyi, altı dikkat.

**Veri sınırı (önemli):** Ücretsiz veri kaynağı fonların ve kurumların gerçek alım-satımını vermez. Gerçek veri için aracı kurum uygulamandaki **AKD (aracı kurum dağılımı)**, **yabancı takas payı** ve KAP'taki **fon portföy bildirimlerine** bak. Elindeki listeyi sol menüden CSV olarak yüklersen kartlarda "Kurum net alıcı/satıcı" etiketi görünür.

**Adım adım**
1. Sermayeni ve riskini gir. 2. 10:15'ten sonra Fırsatlar'a bak. 3. Güncel fiyatı aracı kurum uygulamandan doğrula. 4. Önce stop emrini gir. 5. 1. hedefte yarısını sat. 6. En geç 17:45'te kapat.

**Altın kurallar:** Fırsat yoksa işlem yok. Stop'u geriye çekme. Günde en fazla 2-3 işlem. Günlük kayıp %2-3'ü bulursa dur.
""")

st.divider()
st.caption("Eğitim ve karar destek aracıdır, yatırım tavsiyesi değildir. Veriler gecikmeli veya hatalı olabilir. Kayıp riski her zaman vardır.")
