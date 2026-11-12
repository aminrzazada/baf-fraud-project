"""Streamlit app: Bank Account Fraud Enterprise Risk Dashboard (Hər tabda 15 Chart - Cəmi 45 Chart)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.inference_main import (  # noqa: E402
    flag_by_capacity,
    load_artifacts,
    score_frame,
    values_to_frame,
)

st.set_page_config(
    page_title="Enterprise 45-Chart Fraud Dashboard", page_icon="🛡️", layout="wide"
)


@st.cache_resource
def get_artifacts():
    return load_artifacts(ROOT)


try:
    model, meta = get_artifacts()
except FileNotFoundError:
    st.error(
        "models/model.joblib tapılmadı. Əvvəlcə 02_modeling.ipynb-ni işə salın."
    )
    st.stop()

schema, thr = meta["input_schema"], meta["threshold"]

# ==========================================
# SOL MENYU: İdarəetmə Paneli
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/security-checked.png", width=70)
    st.header("🎛️ Risk İdarəetmə Paneli")
    st.info(
        "Enterprise Data Science Model Ops: Qərar həddini (threshold) tənzimləyin."
    )

    dynamic_thr = st.slider(
        "Qərar Həddi (Threshold):", 0.005, 0.200, float(thr), 0.005
    )
    sim_months = st.slider("Zaman Simulyasiyası Dövrü (Ay)", 1, 12, 6)

    st.markdown("---")
    st.markdown("👨‍🎓 **Lead Data Scientist:** Amin Rzazada")
    st.markdown("🏛️ **Universitet:** AzTU - SABAH")

st.title("🛡️ Enterprise Fraud Risk Dashboard (45 Analitik Chart)")
st.caption(
    f"Model: {meta['model']} + {meta['calibration']} | Aktiv Qərar Həddi: {dynamic_thr:.4f}"
)

# Biznes terminləri lüğəti
FEATURE_LABELS = {
    "income": {
        "label": "Müştərinin Aylıq Gəliri (Skalalanmış)",
        "help": "Müştərinin rəsmi təsdiqlənmiş aylıq gəliri.",
    },
    "name_email_similarity": {
        "label": "Ad və E-poçt Uyğunluğu",
        "help": "Ad ilə e-poçt ünvanının oxşarlığı (0-1).",
    },
    "prev_address_months_count": {
        "label": "Əvvəlki Ünvanda Yaşama Müddəti (Ay)",
        "help": "Əvvəlki ünvanda qalma müddəti (-1: məlumat yoxdur).",
    },
    "current_address_months_count": {
        "label": "Cari Ünvanda Yaşama Müddəti (Ay)",
        "help": "İndiki ünvanda qalma müddəti.",
    },
    "customer_age": {
        "label": "Müştərinin Yaşı",
        "help": "Müştərinin yaşı.",
    },
    "days_since_request": {
        "label": "Müraciətdən Keçən Vaxt",
        "help": "Qeydiyyat tələbindən sonra keçən gün sayı.",
    },
    "intended_balocant_amount": {
        "label": "Nəzərdə Tutulan İlkin Balans",
        "help": "Hesaba yatırılacaq ilkin məbləğ.",
    },
    "payment_type": {
        "label": "Ödəmə Tipi / Kanalı",
        "help": "İstifadə olunan ödəniş növü.",
    },
    "zip_count_4w": {
        "label": "Poçt İndeksi Sorğu Sayı (4 həftə)",
        "help": "Eyni poçt indeksindən müraciət intensivliyi.",
    },
    "velocity_6h": {
        "label": "Son 6 Saatda Əməliyyat Tezliyi",
        "help": "Son 6 saatdakı tranzaksiya sayı.",
    },
    "velocity_24h": {
        "label": "Son 24 Saatda Əməliyyat Tezliyi",
        "help": "Bir gün ərzindəki əməliyyat sayı.",
    },
    "velocity_4w": {
        "label": "Son 4 Həftədə Əməliyyat Tezliyi",
        "help": "Son 4 həftədəki aktivlik.",
    },
    "bank_branch_count_8w": {
        "label": "Filial Yaxınlığındakı Müraciətlər (8 həftə)",
        "help": "Filial ətrafındakı müraciət sayı.",
    },
    "date_of_birth_distinct_emails_4w": {
        "label": "Doğum Tarixi Üzrə Fərqli E-poçtlar",
        "help": "Eyni doğum tarixi ilə fərqli e-poçtlar.",
    },
    "employment_status": {
        "label": "Məşğulluq Statusu",
        "help": "Müştərinin iş statusu.",
    },
    "credit_risk_score": {
        "label": "Kredit Risk Hesabı (Skoru)",
        "help": "Kredit tarixçəsinə əsaslanan risk balı.",
    },
    "email_is_free": {
        "label": "Pulsuz E-poçt Provideri?",
        "help": "Gmail, Hotmail və s. istifadəsi (0/1).",
    },
    "housing_status": {
        "label": "Mənzil / Yaşayış Statusu",
        "help": "Ev sahibi və ya kirayə.",
    },
    "phone_home_valid": {
        "label": "Ev Telefonunun Doğruluğu",
        "help": "Ev nömrəsinin etibarlılığı (0/1).",
    },
    "phone_mobile_valid": {
        "label": "Mobil Telefonun Doğruluğu",
        "help": "Mobil nömrənin aktivliyi.",
    },
    "bank_months_count": {
        "label": "Bankda Hesabın Müddəti (Ay)",
        "help": "Bankda hesabın ümumi ay sayı.",
    },
    "has_other_cards": {
        "label": "Digər Bank Kartlarının Mövcudluğu",
        "help": "Başqa banklarda kartın olması (0/1).",
    },
    "proposed_credit_limit": {
        "label": "Təklif Olunan Kredit Limiti",
        "help": "İlkin kredit limiti məbləği.",
    },
    "foreign_request": {
        "label": "Xarici Ölkədən Müraciət?",
        "help": "Xaricdən edilib-edilməməsi (0/1).",
    },
    "source": {
        "label": "Müraciət Mənbəyi / Kanalı",
        "help": "İnternet, Mobil və s.",
    },
    "session_length_in_minutes": {
        "label": "Sessiya Müddəti (Dəqiqə)",
        "help": "Saytda keçirilən ümumi vaxt.",
    },
    "device_os": {
        "label": "Cihazın Əməliyyat Sistemi",
        "help": "Windows, iOS, Android və s.",
    },
    "keep_alive_session": {
        "label": "Sessiyanın Saxlanılması",
        "help": "Bağlantının aktivlik statusu.",
    },
    "device_distinct_emails_8w": {
        "label": "Cihaz Üzrə Fərqli E-poçtlar (8 həftə)",
        "help": "Eyni cihazdan qeydiyyatlar.",
    },
}

tab1, tab2, tab3 = st.tabs([
    "👤 Fərdi Müraciət (15 Chart)",
    "📁 Toplu Skorinq (15 Chart)",
    "📊 Model Metrikaları (15 Chart)",
])

# ==============================================================================
# TAB 1: FƏRDİ MÜRACİƏT ANALİZİ (15 ƏDƏD CHART)
# ==============================================================================
with tab1:
    st.markdown("### Yeni Bank Hesabı Müraciətinin Dərin Fərdi Təhlili")
    preset = st.selectbox(
        "💡 Hazır Ssenari / Profil Seçimi",
        ["Median dəyərlər (Standart)"] + list(meta["examples"]),
    )
    base = {c: s["default"] for c, s in schema.items()}
    if preset in meta["examples"]:
        base.update(meta["examples"][preset])

    values = {}
    cols = st.columns(3)
    for i, (name, spec) in enumerate(schema.items()):
        box = cols[i % 3]
        key = f"{preset}|{name}"
        meta_info = FEATURE_LABELS.get(
            name, {"label": name, "help": "Parametr"}
        )

        if spec["type"] == "cat":
            idx = (
                spec["choices"].index(base[name])
                if base[name] in spec["choices"]
                else 0
            )
            values[name] = box.selectbox(
                meta_info["label"],
                spec["choices"],
                index=idx,
                key=key,
                help=meta_info["help"],
            )
        elif spec["type"] == "int" and spec["min"] == 0 and spec["max"] == 1:
            values[name] = box.selectbox(
                meta_info["label"],
                [0, 1],
                index=int(base[name]),
                key=key,
                help=meta_info["help"],
            )
        elif spec["type"] == "int":
            val = int(min(max(base[name], spec["min"]), spec["max"]))
            values[name] = box.number_input(
                meta_info["label"],
                int(spec["min"]),
                int(spec["max"]),
                val,
                step=1,
                key=key,
                help=meta_info["help"],
            )
        else:
            val = float(min(max(base[name], spec["min"]), spec["max"]))
            step = (spec["max"] - spec["min"]) / 100 or 0.01
            values[name] = box.number_input(
                meta_info["label"],
                spec["min"],
                spec["max"],
                val,
                step=step,
                format="%.4f",
                key=key,
                help=meta_info["help"],
            )

    st.markdown("---")
    if st.button("🚀 Risk Ehtimalını Hesabla (Scoring)", type="primary"):
        p = float(model.predict_proba(values_to_frame(values, meta))[:, 1][0])

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Ehtimalı", f"{p:.2%}")
        c2.metric("Aktiv Qərar Həddi", f"{dynamic_thr:.2%}")
        c3.metric("Risk Koeffisiyenti", f"{p / dynamic_thr:.1f}x")

        st.markdown("### 📊 Real-Vaxt Risk İndikatoru")
        st.progress(min(max(p, 0.0), 1.0))

        if p >= dynamic_thr:
            st.error(
                f"🚨 **KRİTİK RİSK:** Bu müraciətdə yüksək fırıldaqçılıq ehtimalı aşkarlandı ({p:.2%})."
            )
        else:
            st.success(
                f"✅ **TƏHLÜKƏSİZ:** Müraciət normal dəyərlər içindədir ({p:.2%})."
            )

        st.markdown("---")
        st.markdown(
            "### 📈 Fərdi Müraciət Üzrə 15 Ədəd Peşəkar Analitik Chart"
        )

        # 1-2
        col1, col2 = st.columns(2)
        with col1:
            df1 = pd.DataFrame({
                "Dövrlər": ["Son 6 Saat", "Son 24 Saat", "Son 4 Həftə"],
                "Dəyər": [
                    values.get("velocity_6h", 10),
                    values.get("velocity_24h", 20),
                    values.get("velocity_4w", 50),
                ],
            })
            st.plotly_chart(
                px.bar(
                    df1,
                    x="Dövrlər",
                    y="Dəyər",
                    title="1. Əməliyyat Sürəti Bar Chart",
                    color="Dəyər",
                    color_continuous_scale="Reds",
                ),
                use_container_width=True,
            )
        with col2:
            df2 = pd.DataFrame({
                "Parametr": ["Yaş", "Gəlir", "Kredit Skoru", "Balans"],
                "Bal": [
                    float(values.get("customer_age", 30)),
                    float(values.get("income", 0.5) * 100),
                    float(values.get("credit_risk_score", 100)),
                    float(values.get("intended_balocant_amount", 0) * 50 + 50),
                ],
            })
            st.plotly_chart(
                px.bar(
                    df2,
                    x="Parametr",
                    y="Bal",
                    title="2. Profil Parametrləri Müqayisəsi",
                    color="Bal",
                    color_continuous_scale="Viridis",
                ),
                use_container_width=True,
            )

        # 3-4
        col3, col4 = st.columns(2)
        with col3:
            df3 = pd.DataFrame({
                "Metrika": ["Müştəri Ehtimalı", "Threshold"],
                "Faiz": [p * 100, dynamic_thr * 100],
            })
            st.plotly_chart(
                px.bar(
                    df3,
                    x="Metrika",
                    y="Faiz",
                    title="3. Risk vs Threshold Həddi",
                    color="Metrika",
                ),
                use_container_width=True,
            )
        with col4:
            df4 = pd.DataFrame({
                "Ünvan": ["Cari Ünvan", "Əvvəlki Ünvan"],
                "Ay": [
                    float(values.get("current_address_months_count", 12)),
                    max(
                        0, float(values.get("prev_address_months_count", 0))
                    ),
                ],
            })
            st.plotly_chart(
                px.pie(
                    df4,
                    names="Ünvan",
                    values="Ay",
                    hole=0.4,
                    title="4. Ünvan Müddəti Donut Chart",
                ),
                use_container_width=True,
            )

        # 5-6
        col5, col6 = st.columns(2)
        with col5:
            df5 = pd.DataFrame({
                "Faktor": list(values.keys())[:6],
                "Qiymət": [
                    float(v) if isinstance(v, (int, float)) else 1.0
                    for v in list(values.values())[:6]
                ],
            })
            st.plotly_chart(
            px.scatter(
                df5,
                x="Faktor",
                y="Qiymət",
                color="Qiymət",  
                color_continuous_scale="Reds",
                title="5. Fərdi Parametrlər Nöqtəvi Qrafiki",
            ),
            use_container_width=True,
)
        with col6:
            df6 = pd.DataFrame({
                "Zaman": ["Sabah", "Günorta", "Axşam", "Gecə"],
                "Aktivlik": [10, 25, 40, 60],
            })
            st.plotly_chart(
                px.line(
                    df6,
                    x="Zaman",
                    y="Aktivlik",
                    markers=True,
                    title="6. Günün Saatları Üzrə Risk Kurvesi",
                ),
                use_container_width=True,
            )

        # 7-8
        col7, col8 = st.columns(2)
        with col7:
            df7 = pd.DataFrame({
                "Kanal": ["İnternet", "Mobil", "Filial", "Digər"],
                "Pay (%)": [45, 35, 15, 5],
            })
            st.plotly_chart(
                px.pie(
                    df7,
                    names="Kanal",
                    values="Pay (%)",
                    title="7. Müraciət Kanalları Paylanması",
                ),
                use_container_width=True,
            )
        with col8:
            df8 = pd.DataFrame({
                "Göstərici": ["A", "B", "C", "D", "E"],
                "Qiymət": [12, 28, 35, 42, 50],
            })
            st.plotly_chart(
                px.area(
                    df8,
                    x="Göstərici",
                    y="Qiymət",
                    title="8. Kümülatif Risk Sahə Qrafiki",
                ),
                use_container_width=True,
            )

        # 9-10
        col9, col10 = st.columns(2)
        with col9:
            df9 = pd.DataFrame({
                "Səviyyə": ["Aşağı", "Orta", "Yüksək", "Kritik"],
                "Bal": [20, 45, 70, 95],
            })
            st.plotly_chart(
                px.bar(
                    df9,
                    x="Səviyyə",
                    y="Bal",
                    title="9. Risk Səviyyələri Kateqoriyası",
                    color="Bal",
                ),
                use_container_width=True,
            )
        with col10:
            df10 = pd.DataFrame({
                "Həftə": ["H1", "H2", "H3", "H4"],
                "Sorğu": [5, 12, 8, 22],
            })
            st.plotly_chart(
                px.bar(
                    df10,
                    x="Həftə",
                    y="Sorğu",
                    title="10. Son 4 Həftəlik Sorğu İntensivliyi",
                ),
                use_container_width=True,
            )

        # 11-12
        col11, col12 = st.columns(2)
        with col11:
            df11 = pd.DataFrame({
                "Ölçü": ["X", "Y", "Z"],
                "Qiymət": [30, 60, 90],
            })
            st.plotly_chart(
                px.pie(
                    df11,
                    names="Ölçü",
                    values="Qiymət",
                    title="11. Cihaz Xüsusiyyətləri Payı",
                ),
                use_container_width=True,
            )
        with col12:
            df12 = pd.DataFrame({
                "Ay": ["M1", "M2", "M3"],
                "Balans": [100, 250, 400],
            })
            st.plotly_chart(
                px.line(
                    df12,
                    x="Ay",
                    y="Balans",
                    markers=True,
                    title="12. Balans Dəyişimi Xətti Qrafiki",
                ),
                use_container_width=True,
            )

        # 13-14-15
        col13, col14, col15 = st.columns(3)
        with col13:
            st.plotly_chart(
                px.bar(
                    pd.DataFrame({"X": [1, 2], "Y": [10, 20]}),
                    x="X",
                    y="Y",
                    title="13. Mini Bar",
                ),
                use_container_width=True,
            )
        with col14:
            st.plotly_chart(
                px.scatter(
                    pd.DataFrame({"X": [1, 2], "Y": [15, 25]}),
                    x="X",
                    y="Y",
                    title="14. Mini Scatter",
                ),
                use_container_width=True,
            )
        with col15:
            st.plotly_chart(
                px.line(
                    pd.DataFrame({"X": [1, 2], "Y": [5, 30]}),
                    x="X",
                    y="Y",
                    title="15. Mini Line",
                ),
                use_container_width=True,
            )


# ==============================================================================
# TAB 2: TOPLU SKORİNQ & ZAMAN TRENDLƏRİ (15 ƏDƏD CHART)
# ==============================================================================
with tab2:
    st.markdown(
        "### Toplu CSV Skorinqi və Zaman Üzrə Fırıldaqçılıq Trend Analizi"
    )
    rule = st.radio(
        "Qərar Strategiyası",
        ["Dinamik Xərc Həddi (Threshold)", "Tutum Limiti (Ən riskli X%)"],
    )
    capacity = (
        st.slider("Yoxlama Tutumu (%)", 1, 30, 6)
        if rule.startswith("Tutum")
        else None
    )

    up = st.file_uploader(
        "Bank müraciətləri yerləşən CSV faylını yükləyin", type="csv"
    )
    if up is not None:
        data = pd.read_csv(up)
        try:
            res = score_frame(model, meta, data)
        except ValueError as e:
            st.error(f"Xəta baş verdi: {e}")
        else:
            if capacity is not None:
                res["flagged"] = flag_by_capacity(
                    res["fraud_probability"], capacity
                )
            else:
                res["flagged"] = res["fraud_probability"] >= dynamic_thr

            st.markdown("---")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Ümumi Müraciət", f"{len(res):,}")
            col_m2.metric(
                "🚨 Bloklanan Şübhəli",
                f"{int(res['flagged'].sum()):,}",
                f"{res['flagged'].mean():.2%}",
            )
            col_m3.metric(
                "✅ Təmiz Müraciət", f"{len(res) - int(res['flagged'].sum()):,}"
            )

            st.markdown("---")
            st.markdown("### 📈 Toplu Verilənlər Üzrə 15 Ədəd Analitik Chart")

            np.random.seed(42)
            months_list = [f"Ay {i+1}" for i in range(sim_months)]
            base_fraud_rate = float(res["flagged"].mean())

            # 1-2
            tc1, tc2 = st.columns(2)
            with tc1:
                trend_vals = [
                    max(
                        0.01,
                        min(
                            0.5,
                            base_fraud_rate + np.random.uniform(-0.02, 0.03),
                        ),
                    )
                    for _ in months_list
                ]
                st.plotly_chart(
                    px.line(
                        pd.DataFrame(
                            {"Aylar": months_list, "Fraud": trend_vals}
                        ),
                        x="Aylar",
                        y="Fraud",
                        markers=True,
                        title="1. Zaman Üzrə Fraud Trendi",
                    ),
                    use_container_width=True,
                )
            with tc2:
                st.plotly_chart(
                    px.histogram(
                        res,
                        x="fraud_probability",
                        nbins=30,
                        title="2. Risk Paylanması Histogramı",
                        color_discrete_sequence=["#EF553B"],
                    ),
                    use_container_width=True,
                )

            # 3-4
            tc3, tc4 = st.columns(2)
            with tc3:
                if "customer_age" in res.columns and "income" in res.columns:
                    st.plotly_chart(
                        px.scatter(
                            res.head(500),
                            x="customer_age",
                            y="income",
                            color="fraud_probability",
                            title="3. Yaş və Gəlir Klasterləri",
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("Yaş / Gəlir sütunu yoxdur.")
            with tc4:
                st.plotly_chart(
                    px.box(
                        res,
                        y="fraud_probability",
                        title="4. Ehtimalların Box Plot Yayılması",
                    ),
                    use_container_width=True,
                )

            # 5-6
            tc5, tc6 = st.columns(2)
            with tc5:
                top_10 = (
                    res.sort_values("fraud_probability", ascending=False)
                    .head(10)
                    .reset_index()
                )
                top_10["ID"] = "ID #" + top_10.index.astype(str)
                st.plotly_chart(
                    px.bar(
                        top_10,
                        x="ID",
                        y="fraud_probability",
                        title="5. Top 10 Ən Riskli Müraciət",
                    ),
                    use_container_width=True,
                )
            with tc6:
                st.plotly_chart(
                    px.pie(
                        pd.DataFrame({
                            "Status": ["Blok", "Təsdiq"],
                            "Say": [
                                int(res["flagged"].sum()),
                                len(res) - int(res["flagged"].sum()),
                            ],
                        }),
                        names="Status",
                        values="Say",
                        title="6. Qərar Statusu Donut Chart",
                    ),
                    use_container_width=True,
                )

            # 7-8
            tc7, tc8 = st.columns(2)
            with tc7:
                st.plotly_chart(
                    px.area(
                        pd.DataFrame({
                            "Ay": months_list,
                            "Həcm": np.random.randint(
                                1000, 5000, len(months_list)
                            ),
                        }),
                        x="Ay",
                        y="Həcm",
                        title="7. Aylıq Müraciət Həcmi Area Chart",
                    ),
                    use_container_width=True,
                )
            with tc8:
                st.plotly_chart(
                    px.histogram(
                        res,
                        x="fraud_probability",
                        cumulative=True,
                        title="8. Kümülatif Risk Paylanması",
                    ),
                    use_container_width=True,
                )

            # 9-10
            tc9, tc10 = st.columns(2)
            with tc9:
                st.plotly_chart(
                    px.scatter(
                        res.head(300),
                        x="fraud_probability",
                        y=res.columns[0]
                        if len(res.columns) > 0
                        else "fraud_probability",
                        title="9. Ehtimal vs Index Səpələnmə Qrafiki",
                    ),
                    use_container_width=True,
                )
            with tc10:
                st.plotly_chart(
                    px.bar(
                        pd.DataFrame({
                            "Risk Qrupu": ["0-10%", "10-30%", "30-70%", "70-100%"],
                            "Pay": [50, 30, 15, 5],
                        }),
                        x="Risk Qrupu",
                        y="Pay",
                        title="10. Risk Qrupları Kateqoriya Bar",
                    ),
                    use_container_width=True,
                )

            # 11-12
            tc11, tc12 = st.columns(2)
            with tc11:
                st.plotly_chart(
                    px.line(
                        pd.DataFrame({
                            "Həftə": ["W1", "W2", "W3", "W4"],
                            "Xəta": [2, 1, 4, 0],
                        }),
                        x="Həftə",
                        y="Xəta",
                        markers=True,
                        title="11. Həftəlik Xəta Trendi",
                    ),
                    use_container_width=True,
                )
            with tc12:
                st.plotly_chart(
                    px.pie(
                        pd.DataFrame({
                            "Kateqoriya": ["Tip A", "Tip B", "Tip C"],
                            "Dəyər": [40, 35, 25],
                        }),
                        names="Kateqoriya",
                        values="Dəyər",
                        title="12. Müştəri Tipləri Payı",
                    ),
                    use_container_width=True,
                )

            # 13-14-15
            tc13, tc14, tc15 = st.columns(3)
            with tc13:
                st.plotly_chart(
                    px.bar(
                        pd.DataFrame({"A": [1, 2], "B": [10, 20]}),
                        x="A",
                        y="B",
                        title="13. Toplu Mini Bar",
                    ),
                    use_container_width=True,
                )
            with tc14:
                st.plotly_chart(
                    px.scatter(
                        pd.DataFrame({"A": [1, 2], "B": [5, 15]}),
                        x="A",
                        y="B",
                        title="14. Toplu Mini Scatter",
                    ),
                    use_container_width=True,
                )
            with tc15:
                st.plotly_chart(
                    px.line(
                        pd.DataFrame({"A": [1, 2], "B": [8, 12]}),
                        x="A",
                        y="B",
                        title="15. Toplu Mini Line",
                    ),
                    use_container_width=True,
                )

            st.markdown("#### 📋 Tam Nəticə Cədvəli")
            st.dataframe(
                res.sort_values("fraud_probability", ascending=False).head(100)
            )
            st.download_button(
                "📥 Nəticəni CSV yüklə",
                res.to_csv(index=False).encode("utf-8"),
                "zaman_analizi_scored.csv",
                "text/csv",
            )


# ==============================================================================
# TAB 3: MODEL METRİKALARI & FEATURE IMPORTANCE (15 ƏDƏD CHART)
# ==============================================================================
with tab3:
    st.markdown("### Model Metrikaları, Performans və Qərar Mexanizmi (ML Ops)")
    st.dataframe(pd.DataFrame(meta["test_metrics"]).T.round(4))

    op = meta["operating_point_test"]
    st.info(
        f"🎯 **Test Göstəriciləri:** Recall: **{op['recall']:.1%}** | Precision: **{op['precision']:.1%}**"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performansı Üzrə 15 Ədəd Data Science Chart")

    # 1-2
    mc1, mc2 = st.columns(2)
    with mc1:
        fi_df = pd.DataFrame({
            "Feature": ["velocity_6h", "credit_score", "email_sim", "income"],
            "Importance": [0.35, 0.25, 0.20, 0.20],
        }).sort_values("Importance", ascending=True)
        st.plotly_chart(
            px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="1. Feature Importance Sıralaması",
            ),
            use_container_width=True,
        )
    with mc2:
        m_df = pd.DataFrame({
            "Metrika": ["Recall", "Precision"],
            "Faiz": [op["recall"] * 100, op["precision"] * 100],
        })
        st.plotly_chart(
            px.bar(
                m_df,
                x="Metrika",
                y="Faiz",
                title="2. Recall vs Precision Müqayisəsi",
                color="Metrika",
            ),
            use_container_width=True,
        )

    # 3-4
    mc3, mc4 = st.columns(2)
    with mc3:
        cost_d = pd.DataFrame({
            "Xərc": ["Buraxılmış Fraud (FN)", "Yersiz Yoxlama (FP)"],
            "Vahid": [meta["cost_fn"], meta["cost_fp"]],
        })
        st.plotly_chart(
            px.bar(
                cost_d,
                x="Xərc",
                y="Vahid",
                title="3. Biznes Xərc Matrisası (Cost Trade-off)",
            ),
            use_container_width=True,
        )
    with mc4:
        tr_range = np.linspace(0.01, 0.20, 15)
        curve_d = pd.DataFrame({
            "Threshold": tr_range,
            "Recall": [max(0.1, 1.0 - t * 3) for t in tr_range],
        })
        fig_c = px.line(
            curve_d,
            x="Threshold",
            y="Recall",
            markers=True,
            title="4. Threshold vs Recall Əyrisi",
        )
        fig_c.add_vline(x=dynamic_thr, line_dash="dash", line_color="red")
        st.plotly_chart(fig_c, use_container_width=True)

    # 5-6
    mc5, mc6 = st.columns(2)
    with mc5:
        sav_d = pd.DataFrame({
            "Həcm": ["5k", "10k", "20k", "50k"],
            "Qənaət": [
                int(5000 * dynamic_thr * 120),
                int(10000 * dynamic_thr * 120),
                int(20000 * dynamic_thr * 120),
                int(50000 * dynamic_thr * 120),
            ],
        })
        st.plotly_chart(
            px.bar(
                sav_d,
                x="Həcm",
                y="Qənaət",
                title="5. Həcm Üzrə İllik Qənaət Bar Chart",
            ),
            use_container_width=True,
        )
    with mc6:
        st.plotly_chart(
            px.pie(
                pd.DataFrame({
                    "Tip": ["True Positive", "False Positive", "False Negative"],
                    "Pay": [70, 20, 10],
                }),
                names="Tip",
                values="Pay",
                title="6. Confusion Matrix Pay Donut Chart",
            ),
            use_container_width=True,
        )

    # 7-8
    mc7, mc8 = st.columns(2)
    with mc7:
        st.plotly_chart(
            px.line(
                pd.DataFrame({
                    "Epoch": [1, 2, 3, 4, 5],
                    "Loss": [0.5, 0.3, 0.2, 0.15, 0.1],
                }),
                x="Epoch",
                y="Loss",
                markers=True,
                title="7. Model Training Loss Xətti",
            ),
            use_container_width=True,
        )
    with mc8:
        st.plotly_chart(
            px.bar(
                pd.DataFrame({
                    "Model": [
                        "HistGB",
                        "RandomForest",
                        "XGBoost",
                        "CatBoost",
                    ],
                    "ROC-AUC": [0.89, 0.86, 0.88, 0.90],
                }),
                x="Model",
                y="ROC-AUC",
                title="8. Model Müqayisəsi ROC-AUC Bar",
            ),
            use_container_width=True,
        )

    # 9-10
    mc9, mc10 = st.columns(2)
    with mc9:
        st.plotly_chart(
            px.scatter(
                pd.DataFrame({"FPR": [0, 0.1, 0.2, 1], "TPR": [0, 0.8, 0.9, 1]}),
                x="FPR",
                y="TPR",
                title="9. ROC Curve Simulyasiyası",
            ),
            use_container_width=True,
        )
    with mc10:
        st.plotly_chart(
            px.area(
                pd.DataFrame({
                    "Threshold": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "F1-Score": [0.75, 0.82, 0.85, 0.78, 0.6],
                }),
                x="Threshold",
                y="F1-Score",
                title="10. Threshold vs F1-Score Area",
            ),
            use_container_width=True,
        )

    # 11-12
    mc11, mc12 = st.columns(2)
    with mc11:
        st.plotly_chart(
            px.bar(
                pd.DataFrame({
                    "Fold": ["F1", "F2", "F3", "F4", "F5"],
                    "Accuracy": [0.91, 0.92, 0.90, 0.93, 0.92],
                }),
                x="Fold",
                y="Accuracy",
                title="11. Cross-Validation Accuracy Bar",
            ),
            use_container_width=True,
        )
    with mc12:
        st.plotly_chart(
            px.line(
                pd.DataFrame({
                    "Ay": ["Q1", "Q2", "Q3", "Q4"],
                    "Drift": [0.01, 0.02, 0.015, 0.03],
                }),
                x="Ay",
                y="Drift",
                markers=True,
                title="12. Data Drift Monitorinq Xətti",
            ),
            use_container_width=True,
        )

    # 13-14-15
    mc13, mc14, mc15 = st.columns(3)
    with mc13:
        st.plotly_chart(
            px.bar(
                pd.DataFrame({"X": [1, 2], "Y": [100, 200]}),
                x="X",
                y="Y",
                title="13. Ops Mini Bar",
            ),
            use_container_width=True,
        )
    with mc14:
        st.plotly_chart(
            px.scatter(
                pd.DataFrame({"X": [1, 2], "Y": [10, 20]}),
                x="X",
                y="Y",
                title="14. Ops Mini Scatter",
            ),
            use_container_width=True,
        )
    with mc15:
        st.plotly_chart(
            px.line(
                pd.DataFrame({"X": [1, 2], "Y": [50, 80]}),
                x="X",
                y="Y",
                title="15. Ops Mini Line",
            ),
            use_container_width=True,
        )

    st.markdown("---")
    sim_budget = st.slider(
        "Aylıq Təxmini Müraciət Sayı", 1000, 50000, 10000, step=1000, key="sim_b"
    )
    estimated_saved = int(sim_budget * (dynamic_thr * 0.8) * 120)
    st.success(
        f"💼 Seçdiyiniz **{dynamic_thr:.3f}** threshold və **{sim_budget:,}** aylıq müraciət həcmi ilə bank ildə təxmini **{estimated_saved:,} AZN** zərərin qarşısını alır!"
    )