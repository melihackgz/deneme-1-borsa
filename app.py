from __future__ import annotations
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from backtest.engine import BacktestConfig, run_walk_forward_backtest
from config import (
    DEFAULT_BSMV_RATE, DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE,
    MAX_HEAVY_ANALYSIS_CANDIDATES, PROFILES, TZ
)
from data.bist_rules import bist_market_status, fetch_banned_tickers_with_fallback
from data.sources import (
    HAS_TV_FEED, HAS_TV_TA, get_daily_with_health,
    get_fundamentals_with_health, get_intraday_with_health, get_tv_realtime_snapshot
)
from data.universe import load_universe, refresh_universe_from_web
from exceptions import ValidationError
from indicators.technical import atr, ema
from risk.portfolio import evaluate_portfolio_risk
from risk.notifier import send_telegram_alert
from storage.repository import (
    add_trade, delete_trade, export_df_to_excel_bytes, get_trades, init_db, update_trade_status
)
from strategy.analyzer import analyze_stock_intraday, analyze_stock_swing, build_daily_summary_table
from strategy.fundamental import compute_fundamental_score
from strategy.planner import compute_smart_plan
from ui.components import render_legal_disclaimer, render_signal_card, render_source_health_panel, tl

st.set_page_config(page_title="BIST Karar Destek Tarayıcı", page_icon="📈", layout="wide")
st.markdown("""<style>
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1040px;padding-top:1.1rem}
.hdr{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;padding:18px 22px;border-radius:16px;margin-bottom:12px}
.hdr h1{margin:0;font-size:1.5rem;color:#fff}
.hdr p{margin:6px 0 0;opacity:.85;font-size:.88rem}
.card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:16px 18px;margin:12px 0 6px;background:rgba(128,128,128,.05)}
.card-live{border:2px solid #16a34a;border-radius:16px;padding:16px 18px;margin:12px 0 6px;background:rgba(22,163,74,.05)}
.top{display:flex;justify-content:space-between;align-items:center;gap:10px}
.tk{font-size:1.5rem;font-weight:800}
.sub{opacity:.78;font-size:.84rem;margin-top:2px}
.pill{padding:6px 14px;border-radius:999px;font-weight:700;font-size:.8rem;color:#fff;white-space:nowrap}
.live-badge{display:inline-block;padding:2px 8px;border-radius:6px;background:#16a34a;color:#fff;font-size:.7rem;font-weight:700;margin-left:8px}
.src-badge{display:inline-block;padding:2px 8px;border-radius:6px;background:#0284c7;color:#fff;font-size:.7rem;font-weight:700;margin-left:6px}
.summary-box{margin:10px 0;padding:8px 12px;border-radius:10px;background:rgba(37,99,235,.08);font-size:.85rem}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:10px}
@media(min-width:650px){.grid{grid-template-columns:repeat(4,1fr)}}
.box{border-radius:12px;padding:10px 12px;background:rgba(128,128,128,.1)}
.box small{opacity:.75;display:block;font-size:.74rem;margin-bottom:2px}
.box b{font-size:1.05rem}
</style>""", unsafe_allow_html=True)

init_db()

if "last_toast" in st.session_state:
    st.toast(st.session_state.pop("last_toast"), icon="📌")

st.markdown(
    '<div class="hdr"><h1>📈 BIST Nicel Karar Destek & Tarama Platformu</h1>'
    '<p>Eğitim ve Karar Destek Amaçlı Teknik, Bilanço ve Risk Yönetimi Analiz Motoru</p></div>',
    unsafe_allow_html=True,
)
render_legal_disclaimer()
st_autorefresh(interval=10 * 60 * 1000, key="auto_refresh_timer")

universe_res = load_universe()
if universe_res.warning:
    st.warning(f"⚠️ **Evren Uyarısı:** {universe_res.warning}")

live_prices = st.session_state.setdefault("live_prices", {})
live_times = st.session_state.setdefault("live_times", {})

with st.container(border=True):
    st.markdown("#### ⚙️ 1. Adım: Strateji, Bütçe ve Maliyet Parametreleri")
    c1, c2, c3 = st.columns(3)
    with c1:
        capital = st.number_input("Toplam Portföy Bütçesi (TL)", min_value=5_000, max_value=100_000_000, value=50_000, step=5_000)
    with c2:
        mode = st.selectbox("Analiz Ufku", ["⚡ Gün İçi (Aynı Gün Kapanış)", "🌊 Birkaç Günlük / Haftalık (Swing)"], index=1)
    with c3:
        prof = st.selectbox("Filtre Profili", list(PROFILES), index=1)

    with st.expander("🛠️ Gelişmiş Ayarlar (Komisyon, BSMV, Kayma, Tedbirli Hisseler, Evren)"):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            risk_pct = st.slider("İşlem Başı Maks. Risk (%)", 0.25, 2.5, 1.0, 0.25)
        with a2:
            comm_bps = st.number_input("Komisyon (On Binde)", min_value=0.0, max_value=50.0, value=DEFAULT_COMMISSION_RATE * 10000, step=0.5)
        with a3:
            slip_bps = st.number_input("Tahmini Kayma (On Binde)", min_value=0.0, max_value=100.0, value=DEFAULT_SLIPPAGE_RATE * 10000, step=1.0)
        with a4:
            tf = st.selectbox("Bilanço Eşiği (Swing)", ["Kapalı", "Orta (50+)", "Sıkı (65+)"], index=1)

        default_banned, banned_src = fetch_banned_tickers_with_fallback()
        banned_input = st.text_input(
            f"VBTS / Brüt Takas / Yasaklı Listesi ({banned_src})",
            value=", ".join(sorted(default_banned)),
        )
        banned_set = {x.strip().upper() for x in banned_input.replace(",", " ").split() if x.strip()}

        u1, u2, u3 = st.columns(3)
        with u1:
            if st.button("🔄 Verileri & Önbelleği Yenile", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with u2:
            if st.button("🌐 BIST Evrenini Güncelle (hisseler.txt)", use_container_width=True):
                ok_u, msg_u = refresh_universe_from_web()
                (st.success if ok_u else st.error)(msg_u)
        with u3:
            if st.button("🧹 Manuel Fiyatları Sıfırla", use_container_width=True):
                st.session_state["live_prices"] = {}
                st.session_state["live_times"] = {}
                st.rerun()

rv, c_lo, c_hi, r_lo, r_hi, min_tl, min_atr = PROFILES[prof]
params = {
    "capital": capital,
    "risk_pct": risk_pct,
    "rvol": rv,
    "chg_lo": c_lo,
    "chg_hi": c_hi,
    "rsi_lo": r_lo,
    "rsi_hi": r_hi,
    "comm": comm_bps / 10000.0,
    "bsmv": DEFAULT_BSMV_RATE,
    "slip": slip_bps / 10000.0,
}
min_temel = {"Kapalı": 0, "Orta (50+)": 50, "Sıkı (65+)": 65}[tf] if "Birkaç Günlük" in mode else 0

state, m_msg = bist_market_status()
{"open": st.success, "early": st.warning, "late": st.warning, "closed": st.info}[state]("🕒 " + m_msg)

all_query_tickers = tuple(sorted(set(universe_res.tickers + ["XU100"])))
daily_all, health_list = get_daily_with_health(all_query_tickers)
xu100_df = daily_all.get("XU100", pd.DataFrame())
daily_stocks = {k: v for k, v in daily_all.items() if k != "XU100"}

dtab = build_daily_summary_table(daily_stocks, min_tl, min_atr, banned_set) if daily_stocks else pd.DataFrame()
passed_df = dtab[dtab["Geçti"]].sort_values("İşlem (M TL)", ascending=False) if not dtab.empty else pd.DataFrame()
all_passed_tickers = passed_df["Hisse"].tolist()
analyzed_tickers = all_passed_tickers[:MAX_HEAVY_ANALYSIS_CANDIDATES]
truncated_count = max(0, len(all_passed_tickers) - len(analyzed_tickers))

render_source_health_panel(
    health_list,
    {
        "universe": len(universe_res.tickers),
        "loaded": len(daily_stocks),
        "passed_pre": len(all_passed_tickers),
        "analyzed": len(analyzed_tickers),
        "truncated": truncated_count,
    },
)

if "Birkaç Günlük" in mode and analyzed_tickers:
    fund_map, fund_health = get_fundamentals_with_health(tuple(analyzed_tickers))
    fs_map = {t: compute_fundamental_score(fund_map.get(t, {})) for t in analyzed_tickers}
else:
    fund_map = {}
    fs_map = {t: (None, 0, [], "SANAYI") for t in analyzed_tickers}

tab1, tab2, tab3 = st.tabs([
    "🎯 Tarama Sonuçları & Tekil Analiz",
    "📒 İşlem Günlüğü & Portföy Riski",
    "🧪 Geçmiş Performans Simülasyonu (Backtest)",
])

with tab1:
    with st.container(border=True):
        st.markdown("#### 🔎 Tekil Hisse Analizi")
        with st.form("single_stock_form"):
            q1, q2, q3 = st.columns([2.2, 1.8, 1.0])
            with q1:
                q_ticker = st.text_input("BIST Sembolü (Örn: THYAO)", placeholder="ASELS").strip().upper()
            with q2:
                q_price = st.number_input("Manuel Fiyat (TL - İsteğe Bağlı)", min_value=0.0, value=0.0, step=0.05)
            with q3:
                st.write("")
                st.write("")
                q_submit = st.form_submit_button("Analiz Et", use_container_width=True, type="primary")

        if q_submit and q_ticker:
            st.session_state["active_custom_ticker"] = q_ticker
            if q_price > 0:
                live_prices[q_ticker] = q_price
                live_times[q_ticker] = pd.Timestamp.now(TZ).strftime("%H:%M")

        act_t = st.session_state.get("active_custom_ticker", "")
        if act_t:
            d1d_s = daily_stocks.get(act_t, pd.DataFrame())
            if d1d_s.empty or len(d1d_s) < 55:
                st.error(f"❌ '{act_t}' için yeterli veri bulunamadı. Veri kaynağını kontrol edin.")
            else:
                f_single, _ = get_fundamentals_with_health((act_t,))
                info_s = f_single.get(act_t, {})
                sc_s, _, frows_s, _ = compute_fundamental_score(info_s)
                d_atr_s = float(atr(d1d_s).iloc[-1])
                man_p = live_prices.get(act_t)
                if "Gün İçi" in mode:
                    intra_s, _ = get_intraday_with_health((act_t,))
                    d15_s = intra_s.get(act_t, pd.DataFrame())
                    if not d15_s.empty:
                        item_s = analyze_stock_intraday(act_t, d15_s, d1d_s, params, d_atr_s, live_price=man_p, updated_at=live_times.get(act_t))
                        chart_s = d15_s
                    else:
                        item_s = analyze_stock_swing(act_t, d1d_s, params, d_atr_s, temel_puan=sc_s, live_price=man_p, updated_at=live_times.get(act_t))
                        chart_s = d1d_s
                else:
                    item_s = analyze_stock_swing(act_t, d1d_s, params, d_atr_s, temel_puan=sc_s, live_price=man_p, updated_at=live_times.get(act_t))
                    chart_s = d1d_s

                if item_s:
                    render_signal_card(item_s, info_s, frows_s, mode, chart_df=chart_s, key_prefix="single")

    st.divider()
    if not analyzed_tickers:
        st.info("Mevcut filtrelerle ön elemeyi geçen hisse bulunamadı.")
    else:
        drow = dtab.set_index("Hisse")
        tv_snaps, tv_time, _ = get_tv_realtime_snapshot(tuple(analyzed_tickers))
        results = []

        if "Gün İçi" in mode:
            intra_map, _ = get_intraday_with_health(tuple(analyzed_tickers))
            for t in analyzed_tickers:
                d15_t = intra_map.get(t)
                if d15_t is None or d15_t.empty:
                    continue
                man_p = live_prices.get(t)
                eff_p = man_p or tv_snaps.get(t)
                src_l = "TradingView Canlı" if (not man_p and tv_snaps.get(t)) else "Piyasa Verisi"
                upd_l = live_times.get(t) if man_p else (tv_time if tv_snaps.get(t) else None)
                it = analyze_stock_intraday(t, d15_t, daily_stocks[t], params, float(drow.loc[t, "DailyATR"]), live_price=eff_p, price_source=src_l, updated_at=upd_l)
                if it:
                    it["IsLive"] = bool(man_p)
                    it["_chart_df"] = d15_t
                    results.append(it)
        else:
            for t in analyzed_tickers:
                d1d_t = daily_stocks.get(t)
                if d1d_t is None or d1d_t.empty:
                    continue
                man_p = live_prices.get(t)
                eff_p = man_p or tv_snaps.get(t)
                src_l = "TradingView Canlı" if (not man_p and tv_snaps.get(t)) else "Piyasa Verisi"
                upd_l = live_times.get(t) if man_p else (tv_time if tv_snaps.get(t) else None)
                it = analyze_stock_swing(t, d1d_t, params, float(drow.loc[t, "DailyATR"]), temel_puan=fs_map[t][0], live_price=eff_p, price_source=src_l, updated_at=upd_l)
                if it:
                    it["IsLive"] = bool(man_p)
                    it["_chart_df"] = d1d_t
                    results.append(it)

        ok_temel = lambda x: min_temel == 0 or (x["Temel"] is not None and x["Temel"] >= min_temel)
        # Manuel fiyat sıralamayı DEĞİŞTİRMEZ; yalnızca objektif puan sıralaması kullanılır
        full_match = sorted([x for x in results if x["Puan"] == x["Toplam"] and ok_temel(x)], key=lambda x: -x["Genel"])
        near_match = sorted([x for x in results if x["Puan"] == x["Toplam"] - 1 and ok_temel(x)], key=lambda x: -x["Genel"])

        st.markdown(f"#### ✨ Tüm Kriterleri Karşılayanlar ({len(full_match)} hisse)")
        for x in full_match[:12]:
            t = x["Hisse"]
            render_signal_card(x, fund_map.get(t, {}), fs_map[t][2], mode, chart_df=x["_chart_df"], key_prefix="opp")

        if near_match:
            st.markdown(f"#### 👀 1 Kriteri Eksik İzleme Listesi ({len(near_match)} hisse)")
            for x in near_match[:8]:
                t = x["Hisse"]
                render_signal_card(x, fund_map.get(t, {}), fs_map[t][2], mode, chart_df=x["_chart_df"], key_prefix="near")

with tab2:
    st.subheader("📒 İşlem Günlüğü ve Portföy Risk Kontrolü")
    tdf = get_trades()
    open_df = tdf[tdf["status"] == "AÇIK"].copy() if not tdf.empty else pd.DataFrame()
    sector_lookup = {k: (v.get("sector") or v.get("industry") or "") for k, v in fund_map.items()}
    port_risk = evaluate_portfolio_risk(open_df, capital, sector_lookup)

    for w in port_risk["warnings"]:
        st.warning(f"⚠️ **Portföy Uyarısı:** {w}")

    if "Gün İçi" in mode and not open_df.empty and pd.Timestamp.now(TZ).hour >= 17 and pd.Timestamp.now(TZ).minute >= 30:
        st.error("⏰ **Gün Sonu Kapanış Uyarısı (17:30+):** Gün içi modda açık pozisyonlarınızı seans kapanmadan kapatmanız önerilir.")

    if not tdf.empty:
        # Açık pozisyonlar için anlık fiyat ile otomatik stop/hedef ve K/Z takibi
        live_rows = []
        for _, row in open_df.iterrows():
            tk = row["ticker"]
            cur_p = live_prices.get(tk) or (float(daily_stocks[tk]["Close"].iloc[-1]) if tk in daily_stocks else float(row["entry"]))
            unrealized = round((cur_p - float(row["entry"])) * int(row["lot"]), 2)
            alert_txt = "Normal"
            if cur_p <= float(row["stop"]):
                alert_txt = "🛑 STOP SEVİYESİ VURULDU!"
            elif cur_p >= float(row["target2"]):
                alert_txt = "🎯 2. HEDEF VURULDU!"
            elif cur_p >= float(row["target1"]):
                alert_txt = "✅ 1. HEDEF VURULDU (Kısmi Kâr / Stop Girişe Çek)"
            live_rows.append({"No": row["id"], "Hisse": tk, "Alış": row["entry"], "Güncel": cur_p, "Stop": row["stop"], "Hedef1": row["target1"], "Anlık K/Z (TL)": unrealized, "Durum Uyarısı": alert_txt})

        if live_rows:
            st.markdown("**📡 Açık Pozisyonların Canlı Durumu**")
            st.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True)
            if st.button("📲 Açık Pozisyon Özetini Telegram'a Gönder"):
                msg_lines = ["<b>BIST Açık Pozisyon Özeti</b>"] + [f"• {r['Hisse']}: {r['Güncel']} TL ({r['Anlık K/Z (TL)']:+.2f} TL) - {r['Durum Uyarısı']}" for r in live_rows]
                ok_tg, msg_tg = send_telegram_alert("\n".join(msg_lines))
                (st.success if ok_tg else st.warning)(msg_tg)

        d_exp1, d_exp2 = st.columns(2)
        with d_exp1:
            st.download_button("⬇️ İşlem Günlüğünü CSV İndir", data=tdf.to_csv(index=False).encode("utf-8-sig"), file_name="bist_islem_gunlugu.csv", mime="text/csv", use_container_width=True)
        with d_exp2:
            st.download_button("⬇️ İşlem Günlüğünü Excel (.xlsx) İndir", data=export_df_to_excel_bytes(tdf), file_name="bist_islem_gunlugu.xlsx", use_container_width=True)

        st.dataframe(tdf, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.markdown("**✏️ İşlem Güncelle / Kapat / Sil**")
            sel_id = st.selectbox("Kayıt Seç", tdf["id"].tolist())
            cur_r = tdf[tdf["id"] == sel_id].iloc[0]
            with st.form(f"edit_form_{sel_id}"):
                e1, e2, e3, e4 = st.columns(4)
                with e1:
                    n_entry = st.number_input("Alış Fiyatı", min_value=0.01, value=float(cur_r["entry"]), step=0.05)
                with e2:
                    n_lot = st.number_input("Lot", min_value=1, value=int(cur_r["lot"]), step=1)
                with e3:
                    st_opts = ["AÇIK", "KÂR İLE KAPATILDI", "KISMİ KÂR (T1)", "ZARAR KES OLDU"]
                    n_stat = st.selectbox("Durum", st_opts, index=st_opts.index(cur_r["status"]) if cur_r["status"] in st_opts else 0)
                with e4:
                    n_exit = st.number_input("Satış Fiyatı (Açık ise 0)", min_value=0.0, value=float(cur_r["exit_price"] or 0.0), step=0.05)
                b_save, b_del = st.columns([3, 1])
                s_click = b_save.form_submit_button("💾 Kaydet", use_container_width=True, type="primary")
                d_click = b_del.form_submit_button("🗑️ Sil", use_container_width=True)

            if s_click:
                try:
                    update_trade_status(sel_id, n_stat, n_exit if n_exit > 0 else None, new_entry=n_entry, new_lot=n_lot, partial_t1_taken=(n_stat == "KISMİ KÂR (T1)"))
                    st.session_state["last_toast"] = f"✅ #{sel_id} güncellendi."
                    st.rerun()
                except ValidationError as exc:
                    st.error(f"❌ Doğrulama Hatası: {exc}")
            if d_click:
                delete_trade(sel_id)
                st.rerun()

with tab3:
    st.subheader("🧪 Walk-Forward & Maliyet Dahil Strateji Simülasyonu")
    st.caption("Survivorship bias önlemek için her tarihte o günün en likit hisseleri dinamik seçilir; gap'li stoplar açılış fiyatından doldurulur.")
    b1, b2, b3 = st.columns(3)
    with b1:
        bt_top_n = st.slider("Günlük dinamik likit evren sayısı", 20, 100, 40, 10)
    with b2:
        bt_days = st.slider("Maksimum elde tutma (iş günü)", 3, 15, 5)
    with b3:
        bt_regime = st.checkbox("XU100 Endeks Rejimi Filtresi Aktif", value=True)

    if st.button("🚀 Simülasyonu Çalıştır", type="primary"):
        if not daily_stocks:
            st.error("Önce günlük veriler yüklenmelidir.")
        else:
            bt_cfg = BacktestConfig(
                hold_days=bt_days,
                top_n_liquid_per_day=bt_top_n,
                commission_rate=params["comm"],
                bsmv_rate=params["bsmv"],
                slippage_rate=params["slip"],
                use_regime_filter=bt_regime,
            )
            f_scores_bt = {t: fs_map[t][0] for t in fs_map}
            with st.spinner("2 yıllık walk-forward simülasyon hesaplanıyor..."):
                st.session_state["bt_result"] = run_walk_forward_backtest(daily_stocks, xu100_df=xu100_df, cfg=bt_cfg, fund_scores=f_scores_bt)

    bt_res = st.session_state.get("bt_result")
    if bt_res:
        ov = bt_res["overall"]
        oos = bt_res["out_of_sample"]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Toplam İşlem", ov["count"])
        m2.metric("Kazanma Oranı", f"%{ov['win_rate']}")
        m3.metric("Profit Factor", f"{ov['pf']}x")
        m4.metric("Beklenti (Expectancy)", f"{ov['expectancy']:+.3f} R")
        m5.metric("Max Drawdown", f"{ov['max_dd']} R")
        m6.metric("Sharpe / Sortino", f"{ov['sharpe']} / {ov['sortino']}")

        st.info(f"📊 **Out-of-Sample (Son %30 Test Dönemi) Performansı:** {oos['count']} işlem • Kazanma: **%{oos['win_rate']}** • Profit Factor: **{oos['pf']}x** • Net: **{oos['net_r']:+.2f} R**")
        st.line_chart(bt_res["df"].set_index("Tarih")["Kümülatif Kazanç (R)"])
        st.dataframe(bt_res["monthly"], use_container_width=True, hide_index=True)
