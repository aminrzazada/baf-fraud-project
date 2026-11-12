# 🛡️ Bank Account Fraud Detection

Bank hesabı açılışı zamanı müraciətin fraud olub-olmadığını proqnozlaşdıran end-to-end Data Science layihəsi: EDA → feature engineering → model müqayisəsi → calibration → xərcə əsaslanan threshold → Streamlit app.

## Problem və dataset
- **Məsələ:** binary classification (`fraud_bool`). Bank saniyələr içində müraciəti təsdiqləməli, yoxsa bloklamalıdır.
- **Dataset:** Bank Account Fraud (BAF) Dataset Suite, NeurIPS 2022 (Feedzai), `Base.csv` variantı. Təxminən 1M sətir, 30+ feature, fraud nisbəti ~1%. Real bank datası əsasında generasiya olunmuş, `month` sütunu zaman dinamikasını saxlayır.
- **Çətinliklər:** güclü imbalance, gizli missing kodları (`-1`), zamanla dəyişən paylanma, həssas atribut (`customer_age`).

## Yanaşma
```
raw data → BAFCleaner (-1/mənfi → NaN + indikator, drop)
         → FeatureEngineer (16 yeni feature; BAFCleaner-in 6 missing indikatoru ilə cəmi 22)
         → ColumnTransformer (median impute + StandardScaler | most_frequent + OneHot)
         → FeatureSelector (best features)
         → Model → Calibration (isotonic/sigmoid) → Threshold (xərcə görə)
```
- **Split:** zamana görə: train = ay 0-4, valid = ay 5, test = ay 6-7. Valid tuning/calibration/threshold üçün, test yalnız final yoxlama üçün.
- **Modellər:** Logistic Regression (baseline), Random Forest, HistGradientBoosting.
- **Əsas metrik:** PR-AUC (imbalance səbəbilə). Əlavə: ROC-AUC, TPR@5%FPR.
- **Overfitting yoxlaması:** train / valid / test PR-AUC cədvəli və qrafiki.
- **Feature selection:** permutation importance, kumulyativ 95% qaydası, seçilmiş feature-lərlə modelin **yenidən qurulması** və müqayisəsi.
- **Calibration:** sigmoid vs isotonic, Brier/ECE/reliability curve. Qayda: Brier ən kiçik olan, amma sıralamanı (PR-AUC) pozmayan üsul seçilir.
- **Threshold:** biznes xərci (FN vs FP) ilə optimallaşdırma, həssaslıq analizi, F1/FPR strategiyaları ilə müqayisə.
- **Əlavə:** yaş qrupları üzrə fairness yoxlaması.

## Nəticələr
<!-- RESULTS_START -->
| Göstərici | Dəyər |
|---|---|
| Seçilən model | HistGradientBoosting (72 feature) |
| Test PR-AUC | 0.193 |
| Test ROC-AUC | 0.894 |
| Test TPR @ 5% FPR | 0.542 |
| Calibration | sigmoid (test ECE 0.0016, əvvəl 0.2263) |
| Threshold (xərcə görə) | 0.0130 |
| Threshold-da recall / precision | 0.801 / 0.063 |
| Alarm nisbəti | 17.72% |
<!-- RESULTS_END -->

### Əsas tapıntılar (test = ay 6-7)
- **Model seçimi:** HistGradientBoosting qalib gəldi. Valid PR-AUC: Logistic Regression 0.162, Random Forest 0.149, HistGradientBoosting 0.182 (tuning ilə 0.191).
- **Overfitting:** ROC-AUC-də train-test fərqi Logistic Regression 0.005, Random Forest 0.081, tuning olunmamış HGB 0.063, tuning olunmuş HGB 0.018. Tuning overfitting-i xeyli azaldıb. PR-AUC splitlər arası müqayisə olunmur, çünki fraud nisbəti 1.0% (train) → 1.4% (test) artır.
- **Feature selection:** 72 feature-dən 26 seçildi və model yenidən quruldu. Paired bootstrap göstərdi ki, bu model valid-də statistik olaraq pisdir (ΔPR-AUC −0.0057, 95% CI [−0.0105, −0.0015]), ona görə final modeldə 72 feature saxlanıldı. Test-də fərq çox kiçikdir (0.1919 vs 0.1930).
- **Feature engineering ablation:** FE olmadan 0.1946, FE ilə 0.1930 test PR-AUC. Yəni yazdığımız əlavə feature-lər bu datasetdə ölçülə bilən qazanc vermədi (`results/fe_ablation.csv`). Missing indikatorları isə EDA-da güclü siqnal göstərdi. Logistic Regression ilə eyni ablation-da FE kiçik müsbət qazanc verdi (test PR-AUC 0.1597 → 0.1650), yəni xətti model FE-dən faydalanır, boosting isə kombinasiyaları özü öyrənir.
- **Booster müqayisəsi:** LightGBM, XGBoost, CatBoost və HGB bu datada praktik olaraq eynidir (valid PR-AUC 0.188-0.194, paired bootstrap-da fərq əhəmiyyətli deyil, XGBoost bir qədər pis). HGB sadəliyə görə saxlanıldı (`scripts/compare_boosters.py`).
- **Calibration:** `class_weight` ehtimalları şişirmişdi (ECE 0.226). Sigmoid calibration ECE-ni 0.0016-ya, Brier-i 0.107-dən 0.012-yə endirdi, PR-AUC dəyişmədi.
- **Threshold:** xərc fərziyyəsi (FN=500, FP=10) ilə optimal threshold 0.013. Test-də fraudun 80%-i tutulur, 1000 müraciətə xərc 3057 (heç kəsi bloklamamaqla 7019). Bunun qiyməti: müraciətlərin 17.7%-i işarələnir, precision 6.3%. Daha az alarm üçün FPR≤5% strategiyası (recall 55%, alarm 5.9%) və həssaslıq cədvəli notebook-dadır.
- **Fairness:** 50+ yaşlılarda real fraud nisbəti 2.34%, 50-dən aşağıda 0.83% (2.8 dəfə yüksək), buna görə model yaşlıları daha çox işarələyir. Amma yanlış alarm nisbəti (FPR) yalnız legit müştərilər üzərində hesablanır və yaşlılarda 32% vs 14%-dir, yəni baza nisbəti bunu birbaşa izah etmir; partial dependence modelin yaşdan əhəmiyyətli istifadə etdiyini göstərir. Qrup üzrə threshold FPR-ı bərabərləşdirir (5.1% / 5.8%), amma qərarda yaşdan istifadə deməkdir (hüquqi/etik məsələ). Yaşsız model sınanıb (`scripts/age_ablation.py`): FPR fərqi 9.1 → 6.6 faiz bəndi azalır (ümumi FPR≤5% nöqtəsində), PR-AUC 0.1938 → 0.1906, amma fərq tam aradan qalxmır. Final modeldə yaş saxlanıb.


- **Sabitlik (03):** ROC-AUC valid/test aylarında 0.893-0.897, seçilən threshold-da recall 78-82%, alarm nisbəti 16-19%.
- **Data drift (03):** velocity feature-lərində güclü drift (PSI: `velocity_4w` 3.7, `velocity_24h` 1.9, `velocity_6h` 1.1), lakin model skorunun PSI-si yalnız 0.016. Bu feature-lər permutation importance top-20-də deyil (səbəb əlaqəsi yoxlanılmayıb).
- **Lift (03):** ilk decile fraudun 67%-ni tutur (lift 6.7x), KS = 0.637.
- **İnam intervalları (test bootstrap):** PR-AUC 0.193 [0.180, 0.210], ROC-AUC 0.894 [0.888, 0.901], TPR@5%FPR 0.542 [0.523, 0.559].
- **Zamana görə CV (5 fold):** ROC-AUC 0.902 ± 0.009, PR-AUC 0.186 ± 0.019.

## Repo strukturu
```
├── 01_eda.ipynb              # EDA
├── 02_modeling.ipynb         # Pipeline, modellər, calibration, threshold, artefaktlar
├── 03_extended_analysis.ipynb # Sabitlik, drift (PSI), lift/KS, CI, xəta analizi, PDP, zamana görə CV
├── app.py                    # Streamlit app
├── src/
│   ├── config.py             # sabitlər (split aylar, xərc fərziyyələri)
│   ├── features.py           # BAFCleaner, FeatureEngineer, FeatureSelector, pipeline
│   ├── evaluation.py         # metrikalar, ECE, threshold funksiyaları
│   └── inference.py          # app üçün skorinq məntiqi
├── models/                   # model.joblib + meta.json (notebook yaradır)
├── scripts/make_synthetic_baf.py   # yalnız kodu yoxlamaq üçün sintetik data
├── scripts/fe_ablation.py          # FE-nin faydasını ölçən ablation
├── scripts/compare_boosters.py     # LightGBM/XGBoost/CatBoost vs HGB (pip install -r requirements-extra.txt)
├── scripts/fe_ablation_linear.py   # FE ablation Logistic Regression ilə
├── scripts/age_ablation.py         # yaşsız model: performans və qruplar üzrə FPR
├── scripts/wider_tuning.py         # genişləndirilmiş HGB tuning
├── results/fe_ablation.csv
├── DECISIONS.md              # hər qərarın səbəbi
├── DEFENSE_QA.md             # müdafiə sual-cavabları
└── requirements.txt
```

## İşə salma
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # Python >= 3.11
# Base.csv-ni Kaggle-dan yükləyib data/ qovluğuna qoyun
jupyter notebook          # 01_eda.ipynb, sonra 02_modeling.ipynb (Run All)
streamlit run app.py
```
`02_modeling.ipynb` bitəndə `models/model.joblib`, `models/meta.json` yaranır və bu README-nin nəticə cədvəli yenilənir.

**Qeyd:** repo-dakı notebook-lar real `Base.csv` ilə işlədilib və çıxışlarla saxlanılıb. `models/` artıq real modeli saxlayır.

**Yaddaş:** `03_extended_analysis.ipynb` 1M sətirlik datanı və 5 fold CV-ni işlədir, təxminən 4 GB RAM tələb edə bilər.

**Sürətli yoxlama (real data olmadan):** `python scripts/make_synthetic_baf.py` və `BAF_DATA_PATH=data/Base_synthetic.csv` ilə notebook-u işlədin. Sintetik datanın nəticələri real deyil.

## Deploy (Streamlit Community Cloud)
1. Repo-nu GitHub-a push edin (`models/` daxil olmaqla).
2. share.streamlit.io → New app → repo və `app.py` seçin.
3. **Vacib:** `requirements.txt`-də `scikit-learn==1.9.1` pinlənib, çünki `model.joblib` bu versiya ilə hazırlanıb. Deploy-da model yüklənməsə (pandas/numpy versiya fərqi mümkündür), `02_modeling.ipynb`-ni öz mühitinizdə işlədib yeni `models/model.joblib` commit edin.
4. `models/model.joblib` GitHub-un 100 MB limitindən kiçik olmalıdır (notebook ölçünü çap edir).

## Məhdudiyyətlər
- Xərc fərziyyələri (`COST_FN`, `COST_FP`) illüstrativdir; real tətbiqdə bankın rəqəmləri ilə əvəz olunmalıdır.
- Dataset sintetik/anonimləşdirilmişdir; real production performansı fərqli ola bilər.
- Fraud davranışı zamanla dəyişir: production-da drift monitorinqi və dövri retraining lazımdır.
