# Müdafiə sual-cavab (real nəticələrlə)

Bütün rəqəmlər `02_modeling.ipynb` çıxışlarındandır (test = ay 6-7, əgər başqa qeyd olunmayıbsa).

## 1. Dataset və məsələ
**Niyə bu dataset?** BAF (NeurIPS 2022, Feedzai): real bank hesabı açılışı datası əsasında generasiya olunmuş, 1,000,000 sətir, fraud nisbəti 1.10%, 8 aylıq zaman ölçüsü var. Güclü imbalance, zaman driftı və həssaslıq atributu (yaş) ona real iş problemi xarakteri verir. Titanic/Telco kimi klassik datasetlərdə bu ölçülər yoxdur.

**Dataset sintetikdirmi?** Anonimləşdirilmiş real datadan generasiya olunub (privacy üçün). Nəticələr real bankın performansına zəmanət deyil.

**Məsələnin növü?** Binary classification: `fraud_bool`.

## 2. Preprocessing
**Niyə `-1` NaN edildi?** `-1` ölçü deyil, "məlumat yoxdur" kodudur. Rəqəm kimi saxlasaq model onu "ən kiçik dəyər" kimi şərh edər. Missing indikator saxladıq, çünki EDA göstərdi ki, missing özü siqnaldır: `prev_address_months_count` yoxdursa fraud 1.42%, varsa 0.31%; `intended_balcon_amount` üçün 1.31% vs 0.50%.

**Niyə sütun drop etdin?** Yalnız iki hal: `device_fraud_count` (real datada sabit, variansı 0, kodla yoxlanıb) və `month` (split açarıdır, feature olsa model zamana yapışardı). Başqa sütun atılmayıb.

**Gizli missing nə qədərdir?** `intended_balcon_amount` ~74%, `prev_address_months_count` ~71%, `bank_months_count` ~25%. Bu qədər yüksək olduğu üçün sütunları silmədik, indikator əlavə etdik.

**Median imputation niyə?** Çəp paylanmalarda ortalamadan dayanıqlıdır. KNN/iterative imputer 1M sətirdə bahalıdır və missing indikator məlumatı saxlayır.

**Outlier niyə silinmədi?** Fraud kənar dəyərlərdə cəmləşə bilər, silmək siqnalı atardı. Xətti modeldə scaling, ağac modellərində dayanıqlılıq kifayət edir.

**StandardScaler / OneHot niyə?** Scaling Logistic Regression üçün vacibdir, ağac modellərə zərəri yoxdur və tək preprocessing yolu pipeline-ı sadələşdirir. Kateqoriklər az səviyyəli və nominaldır, label encoding saxta sıra yaradardı.

## 3. Split
**Niyə random split yox?** Zaman sütunu var. Random split gələcəyi train-ə qatıb nəticəni şişirər. Zaman driftinin real olduğunu EDA göstərdi: aylıq fraud nisbəti 0.87%-dən 1.47%-ə qədər dəyişir və son aylarda yüksəkdir.

**Niyə train 0-4 / valid 5 / test 6-7?** Valid ayrıca ay olmalıdır ki, tuning, calibration, threshold test-i çirkləndirməsin. Split-lərdə fraud nisbəti: 1.00% / 1.18% / 1.40%.

## 4. Metrik
**Niyə PR-AUC?** Fraud ~1% olduğu üçün accuracy yanıltıcıdır, ROC-AUC isə çoxlu true negative səbəbilə yüksək görünür. PR-AUC fraud sinfinin precision-recall mübadiləsini ölçür. İkinci metrik TPR@5%FPR: BAF məqaləsinin əməliyyat metrikası ("5% yanlış alarmla fraudun neçə faizi tutulur").

**PR-AUC-in zəifliyi nədir?** Baza nisbətindən asılıdır. Fraud nisbəti train-dən test-ə artdığı üçün PR-AUC splitlər arası birbaşa müqayisə olunmur. Buna görə overfitting yoxlamasında ROC-AUC-ə də baxdıq.

## 5. Modellər, overfitting, tuning
**Üç model, nəticələr (valid PR-AUC / test PR-AUC):** Logistic Regression 0.162 / 0.165; Random Forest 0.149 / 0.158; HistGradientBoosting 0.182 / 0.184. Seçim valid üzrədir.

**Niyə məhz bu üç model?** Üç fərqli ailə: xətti baseline, bagging, boosting. Baseline-ı mürəkkəb modelin keçməli olduğu mərtəbə kimi saxladıq.

**XGBoost/LightGBM niyə yox?** Onları da yoxladım (`scripts/compare_boosters.py`, eyni split və pipeline, boosterlər sabit parametrlərlə, HGB tuning olunmuş). Valid PR-AUC: LightGBM 0.1942, CatBoost 0.1915, HGB tuned 0.1913, XGBoost 0.1881. Test PR-AUC: LightGBM 0.1975, CatBoost 0.1958, XGBoost 0.1951, HGB tuned 0.1938. HGB-yə qarşı paired bootstrap (valid, Δ PR-AUC): LightGBM +0.0027 [-0.0024, +0.0072], CatBoost +0.0003 [-0.0041, +0.0057], yəni fərq əhəmiyyətli deyil; XGBoost -0.0031 [-0.0061, -0.0001], yəni HGB-dən bir qədər pis. Test fərqləri ən çoxu 0.004-dür, test inam intervalının (±0.015) daxilindədir. Nəticə: dörd boosting kitabxanası bu datada praktik olaraq eyni nəticə verir. HGB-ni saxladım, çünki performans eynidir, əlavə asılılıq tələb etmir və `model.joblib` yalnız scikit-learn-dən asılıdır (deploy sadədir). Dürüst qeyd: yalnız valid PR-AUC-ə baxsaydıq LightGBM ən yüksək çıxırdı, amma fərq statistik əhəmiyyətli deyil. Tələb 2-3 fərqli model müqayisəsidir: xətti, bagging, boosting.

**Overfitting varmı?** Train-test ROC-AUC fərqi: Logistic Regression 0.005, Random Forest 0.081, tuning olunmamış HGB 0.063, **tuning olunmuş HGB 0.018** (train 0.912, test 0.894). Yəni Random Forest və baseline HGB aydın overfit edirdi, tuning bunu xeyli azaltdı.

**Tuning olunmuş modeldə test PR-AUC train-dən yüksəkdir (0.193 vs 0.165), bu nədir?** Overfitting əlaməti deyil, baza nisbəti effektidir: test-də fraud 1.4%, train-də 1.0%. Buna görə ROC-AUC fərqinə baxırıq.

**Tuning necə aparıldı, nə qədər qazandı?** Valid ayında 6 təsadüfi sınaq (random CV zaman sızıntısı verərdi). Valid PR-AUC 0.182 → 0.191, test 0.184 → 0.193. Ən yaxşı: max_depth=3, min_samples_leaf=20, l2=0, learning_rate=0.1. **Zəiflik:** bu dəyərlər axtarış şəbəkəsinin kənarında idi. Bunu yoxlamaq üçün `scripts/wider_tuning.py` ilə genişləndirilmiş şəbəkədə 10 təsadüfi sınaq (+ mövcud parametrlər) işlətdim: ən yaxşı valid PR-AUC 0.1919, mövcud parametrlərdə 0.1913 (fərq +0.0006, səs-küy daxilində). 11 sınaqda valid PR-AUC 0.164-0.192 aralığındadır, yəni səth düzdür və mövcud parametrlər onun yuxarı hissəsindədir. Test-də ən yaxşı konfiqurasiya 0.1984, mövcud 0.1938 verdi, amma test seçimdə iştirak etmir, ona görə parametrləri dəyişmədim. Qeyd: 10 sınaq azdır və ən yaxşı konfiqurasiyanın 3 parametri (learning_rate=0.15, max_depth=2, l2=100) yenə yeni şəbəkənin kənarındadır. Tuning-in əsas faydası overfitting-i azaltmaqdır (train-test ROC-AUC fərqi 0.063 → 0.018): valid-də tuning-siz və tuning olunmuş HGB arasında Δ PR-AUC = -0.0047, 95% CI [-0.0100, +0.0015], yəni PR-AUC fərqi statistik əhəmiyyətli deyil.

**Random Forest niyə zəif çıxdı?** Sübut: valid PR-AUC 0.149 (üç modelin ən aşağısı) və train-test ROC-AUC fərqi 0.081 (Logistic Regression-da 0.005). Yəni RF train-ə uyğunlaşıb, ümumiləşdirmə zəifdir. İzah: ağaclar bir-birindən asılı olmadan qurulur və nadir fraud sinfində dərin yarpaqlar konkret nümunələri əzbərləyə bilir. Boosting isə hər addımda əvvəlki səhvləri düzəldir və `max_depth`, `min_samples_leaf`, `l2` ilə nəzarət olunur. Qeyd: RF konfiqurasiyası yüngüldür (100 ağac, hər ağac datanın 20%-i, sürət üçün), daha ağır konfiqurasiya ayrıca sınanmayıb. Ona görə nəticəni "bu konfiqurasiyada, bu büdcədə" kimi ifadə edirəm. Seçim valid PR-AUC-ə görə edildiyi üçün bu qərarı dəyişmir.

**Hər parametr niyə belə seçildi?**
| Model / parametr | Dəyər | Səbəb |
|---|---|---|
| Hamısı: `class_weight` | `balanced` (RF: `balanced_subsample`) | Fraud ~1%, class weight azlıq sinfinə çəki verir. SMOTE sintetik sətir yaradıb real paylanmanı pozur, calibration-ı çətinləşdirir. RF-də `balanced_subsample` çəkini hər ağacın öz bootstrap nümunəsində hesablayır |
| LogReg `C` | 0.1 | `C` regularizasiyanın tərsidir (kiçik = güclü). 72 feature və korrelyasiyalı qruplar (velocity) var, güclü L2 əmsalları sabitləşdirir. Qeyd: bu baseline-dır, tuning yalnız qalib model üçün aparılıb, `C` üçün ayrıca müqayisə nəticəsi yoxdur |
| LogReg `max_iter` | 1000 | Standart 100 iterasiya 1M sətirdə konvergensiya xəbərdarlığı verə bilər |
| RF `n_estimators` | 100 | Ağac sayı artdıqca variasiya azalır, amma qazanc tədricən azalır. 1M sətirdə 100 vaxt və keyfiyyət kompromisidir |
| RF `min_samples_leaf` | 30 | Yarpaqda ən azı 30 nümunə: nadir fraud-u tək nümunəyə uyğunlaşmanın qarşısı |
| RF `max_samples` | 0.2 | Hər ağac datanın 20%-i ilə öyrənir: sürət (1M sətir) və ağaclar arası korrelyasiyanın azalması |
| HGB `learning_rate` | 0.1 | Standart sürət/dəqiqlik balansı. Tuning grid-ində 0.03 və 0.05 də sınanıb |
| HGB `max_iter` | 200 | Ağac sayı. `early_stopping=False` ilə sabit saxlanılır |
| HGB `early_stopping` | `False` | Daxili early stopping validasiya hissəsini təsadüfi seçir, bu isə zaman sızıntısı verərdi. Zaman validasiyası bizim ayrıca valid ayımızdadır |
| HGB `max_depth` | 6 → tuning-dən sonra 3 | Dərinlik kapasiteni idarə edir. Tuning dayaz ağacı seçdi və ROC-AUC fərqini 0.063-dən 0.018-ə endirdi (overfitting azaldı) |
| HGB `min_samples_leaf` | tuning grid: 20/100/300 | Nadir sinifdə yarpağın kiçik qrupa uyğunlaşmasının qarşısı. Seçilən 20 şəbəkənin aşağı kənarındadır |
| HGB `l2_regularization` | tuning grid: 0/1/10 | Yarpaq çəkilərini kiçildir. Seçilən 0 şəbəkənin kənarındadır, yəni əlavə regularizasiya valid-də fayda vermədi |

## 6. Feature engineering və selection
**Feature engineering faydalı oldu?** Dürüst cavab: bu datasetdə ölçülə bilən qazanc vermədi. Ablation (eyni tuned model): FE olmadan 56 feature, test PR-AUC 0.1946; FE ilə 72 feature, 0.1930. Valid-də 0.1876 vs 0.1906. Fərq kiçikdir və istiqaməti dəyişir, ROC-AUC eynidir. Səbəb: boosting ağacları bölgülər vasitəsilə feature kombinasiyalarını və nisbətləri özü öyrənir, BAF feature-ləri isə artıq mühəndislik olunmuş formada gəlir. Dolayı sübut: permutation importance top-20-də yalnız 3 feature-im var və ablation-da ROC-AUC eynidir. Bu izahı Logistic Regression ilə eyni ablation-la yoxladım (`scripts/fe_ablation_linear.py`): xətti model interaksiyanı özü öyrənə bilmir və FE-dən faydalandı. Valid PR-AUC 0.1552 → 0.1615, test 0.1597 → 0.1650, valid ROC-AUC 0.8829 → 0.8860. Yəni FE xətti modeldə kiçik müsbət qazanc verir, HGB-də vermir, bu izahı dəstəkləyir. Qeyd: tək split, inam intervalı hesablanmayıb, ona görə 'sübut edir' yox, 'dəstəkləyir' deyirəm. Missing indikatorlar isə EDA-dakı fraud fərqinə (1.42% vs 0.31%) əsaslanır və məntiqli idi.

**Top feature-lər hansılardır?** Permutation importance-a görə: `housing_status_BA`, `has_other_cards`, `device_os_windows`, `phone_home_valid`, `keep_alive_session`, `current_address_months_count`, `proposed_credit_limit`, `name_email_similarity`. Mənim yaratdığım feature-lərdən top-20-də yalnız `email_risk` və iki missing indikator var.

**Niyə permutation importance?** Model-agnostikdir və korrelyasiyalı feature-lərdə impurity importance-dən etibarlıdır.

**Best features ilə model yenidən quruldu?** Bəli: 72 feature-dən 26-sı seçildi (müsbət importance-ın 95% kumulyativ payı), model yenidən quruldu. Test: 0.1919 vs 0.1930 PR-AUC, TPR@5%FPR 0.538 vs 0.542. Valid-də paired bootstrap: ΔPR-AUC = −0.0057, 95% CI [−0.0105, −0.0015], yəni az feature-li model statistik olaraq pisdir. **Ona görə final modeldə 72 feature qalıb.** Praktik fərq çox kiçikdir, 26 feature-li model də qəbul edilə bilən alternativdir.

**Seçim qaydasını nəticədən sonra dəyişmisiniz?** Bəli, açıq deyirəm: ilk qayda "valid PR-AUC-in ≥99%-i" idi, sonra bootstrap testi ilə əvəz etdim, çünki 1% həddi özbaşına idi. Hər iki qayda eyni nəticə verdi (72 feature).

**Importance valid-də hesablanıb, qərar da valid-də: sızıntı?** Kiçik optimizm riski var. Test seçimdə iştirak etmir və test-də də fərq göstərilib.

## 7. Calibration
**Niyə calibration?** `class_weight='balanced'` ehtimalları şişirir: ECE 0.226. Sigmoid calibration ilə ECE 0.0016, Brier 0.107 → 0.012, log-loss 0.347 → 0.054 (test). PR-AUC dəyişmədi (0.193), çünki sigmoid monotondur və sıralamanı saxlayır.

**Niyə sigmoid, isotonic yox?** İkisi valid-in ayrıca hissəsində Brier ilə müqayisə olunur: 0.01042 (sigmoid) vs 0.01044 (isotonic), yəni praktik olaraq bərabər. Bərabərlikdə sıralamanı saxlayan seçilir: sigmoid monotondur, isotonic isə pilləli funksiyadır və bərabər skorlar yaradır. Bunu notebook-u yenidən işlədəndə gördüm: isotonic seçiləndə test PR-AUC 0.194 → 0.177, TPR@5%FPR 0.543 → 0.472 düşdü. Ona görə qayda belədir: Brier ən kiçik olan, amma PR-AUC-i pozmayan üsul. **Qayda ilk nəticədən sonra əlavə olunub** (DECISIONS.md-də açıq yazılıb).

## 8. Threshold
**Niyə 0.5 yox?** Default 0.5 testdə fraudun cəmi 3.6%-ni tutur (müraciətlərin 0.09%-i işarələnir), xərc 6772, heç kəsi bloklamamaqdan (7019) demək olar fərqlənmir.

**Seçilən threshold?** 0.0130 (xərcə görə). Test: recall 80.1%, precision 6.3%, alarm 17.7%, xərc 3057/1000 müraciət. Müqayisə: heç kəsi bloklamamaq 7019, hamını bloklamaq 9860, FPR≤5% strategiyası 3677 (recall 55%, alarm 5.9%), F1-optimal 5344 (recall 25%). Xərc 56% azalır.

**Xərc rəqəmləri haradandır?** Illüstrativ fərziyyələrdir (FN=500, FP=10), real bank rəqəmləri ilə əvəz olunmalıdır. Həssaslıq (FN/FP nisbəti → threshold, recall, alarm): 10 → 0.089, 40%, 2.5%; 25 → 0.030, 63%, 8.4%; 50 → 0.013, 80%, 17.2%; 100 → 0.008, 87%, 24.6%; 250 → 0.004, 95%, 41.8%.

**17.7% alarm real həyatda çoxdur, deyilmi?** Bəli, bu seçilən xərc nisbətinin nəticəsidir (buraxılmış fraud 50 dəfə bahadır). Yoxlama komandasının tutumu məhduddursa, FPR≤5% strategiyası (alarm 5.9%, recall 55%) və ya həssaslıq cədvəlindən daha aşağı nisbət seçilməlidir. Streamlit app-da bunun üçün "Tutum limiti" qaydası var: CSV-də ən riskli X% müraciət işarələnir.

**Nəzəri və empirik threshold uyğundurmu?** Nəzəri C_FP/(C_FP+C_FN) = 0.0196, empirik 0.0130. Eyni tərtibdədir, amma bərabər deyil. Fərq calibration-ın mükəmməl olmamasından və valid ayının sonlu ölçüsündən gələ bilər.

## 9. Fairness
**Yaşa görə ayrı-seçkilik varmı?** Var. Seçilən threshold-da 50+ yaşlılarda yanlış alarm nisbəti (FPR) 32.3%, 50-dən aşağıda 13.9%; alarm nisbəti 34.0% vs 14.5%; recall 90.9% vs 74.4%. Yəni model yaşlıları həm daha çox tutur, həm də daha çox səhv bloklayır.

**Nə etmək olar?** Qrup üzrə threshold sınadıq: FPR 5.1% / 5.8%-ə bərabərləşir, recall 53.4% / 52.9% olur. Amma bu qərarda həssas atributdan istifadə deməkdir (hüquqi və etik məsələ). **Səbəb:** 50+ yaşlılarda real fraud nisbəti 2.34%, 50-dən aşağıda 0.83% (2.8 dəfə; yaş artdıqca monoton artır, EDA-da göstərilib) və partial dependence modelin yaşdan əhəmiyyətli istifadə etdiyini göstərir (orta ehtimal yaş artdıqca qalxır). Yüksək alarm nisbəti qismən bununla izah olunur. Amma FPR yalnız legit müştərilər üzərində hesablanır, ona görə baza nisbəti fərqi onu birbaşa izah etmir: yaşlı legit müştərilər 2.3 dəfə çox səhv bloklanır. Yaşı feature-dən çıxarmağı sınadım (`scripts/age_ablation.py`: `customer_age` və ondan törəyən `current_addr_share_of_life`, 72 → 70 feature; threshold valid-də ümumi FPR≤5% şərti ilə seçilib, test-də qruplar müqayisə olunur). Nəticə: test PR-AUC 0.1938 → 0.1906, TPR@5%FPR 0.5427 → 0.5372. 50+ yaşlılarda FPR 13.0% → 10.9%, 50-dən aşağıda 3.9% → 4.4%, fərq 9.1 → 6.6 faiz bəndi (təxminən 27% azalma). Recall 50+ üçün 71.0% → 65.9%, <50 üçün 48.0% → 49.7%. Yəni yaşı çıxarmaq fərqi qismən azaldır və kiçik performans qiyməti var, amma problemi həll etmir: 50+ yaşlılarda FPR hələ də təxminən 2.5 dəfə yüksəkdir (10.9% vs 4.4%). Səbəb: `income`, `employment_status`, `housing_status` kimi feature-lər yaşı qismən bərpa edir (bu bağlılıq ayrıca ölçülməyib). Qeyd: bu əməliyyat nöqtəsi (ümumi FPR≤5%) xərc threshold-undan (32% vs 14%) fərqlidir, ona görə rəqəmləri birbaşa müqayisə etmək olmaz. Final modeldə yaş saxlanıb, çünki PR-AUC fərqi kiçikdir və yaş real siqnaldır; yaşsız model qəbul edilə bilən alternativdir. Bu tam fairness auditi deyil, ilkin yoxlamadır.

## 10. Deploy və production
**Streamlit-də niyə tam data yox?** Pipeline (preprocessing + model + calibration) tək `model.joblib` faylındadır (~0.1 MB), app xam sətri verib eyni emaldan keçirir, train/serving uyğunsuzluğu olmur.

**Production-da nə edərdin?** Drift monitorinqi (aylıq PR-AUC, ehtimal paylanması), dövri retraining, threshold-un real xərclərlə yenilənməsi, label gecikməsinin nəzərə alınması, fairness monitorinqi.

## 11. Əlavə analiz (`03_extended_analysis.ipynb`)
**Model zamanla sabitdirmi?** ROC-AUC valid/test aylarında 0.897 (ay 5), 0.893 (ay 6), 0.897 (ay 7). Öyrənilən aylarda (in-sample) 0.893-0.926, yəni test aylarından yalnız bir qədər yüksəkdir (tuning olunmuş model güclü overfit etmir). Seçilən threshold-da recall 78-82%, alarm nisbəti 16-19%, xərc 1000 müraciətə 2840 (valid) və 3045-3070 (test). Fraud nisbəti 1.18%-dən 1.47%-ə qalxsa da threshold qaydası işləyir.

**Data drift varmı?** Bəli. Velocity feature-lərində çox güclü drift var (PSI: `velocity_4w` 3.74, `velocity_24h` 1.89, `velocity_6h` 1.09), həmçinin `zip_count_4w` 0.49, `date_of_birth_distinct_emails_4w` 0.40, `credit_risk_score` 0.26 (praktik qayda: >0.25 güclü dəyişmə). Amma model skorunun PSI-si yalnız 0.016, yəni çıxış paylanması sabitdir. Velocity feature-ləri permutation importance top-20-də yoxdur, bu, modelin driftə nisbətən dayanıqlı olmasının ehtimal olunan səbəbidir (səbəb əlaqəsi yoxlanılmayıb). Production-da feature drift monitorinqi mütləq lazımdır.

**Lift və KS?** Test-də ilk decile (müraciətlərin 10%-i) fraudun 67%-ni, ilk iki decile 82%-ni tutur. Lift 6.7x, KS = 0.637.

**Nəticələr nə qədər dəqiqdir?** Test bootstrap 95% inam intervalı: PR-AUC 0.193 [0.180, 0.210], ROC-AUC 0.894 [0.888, 0.901], TPR@5%FPR 0.542 [0.523, 0.559]. Bu interval yalnız test nümunəsinin təsadüfiliyini əks etdirir; yenidən öyrətmə və zaman driftindən gələn qeyri-müəyyənliyi daxil etmir.

**Hansı fraudlar qaçır?** Test-də 2305 TP, 573 FN, 34,026 FP. Qaçan fraudlar tutulanlardan fərqli profildədir: orta limit 499 vs 881, `credit_risk_score` 146 vs 199, yaş 34 vs 42, `name_email_similarity` 0.45 vs 0.37, əvvəlki ünvanı çatışmayanların payı 77% vs 94%. Yəni model "klassik" riskli profilləri tutur, adi müştəriyə bənzəyən fraudları isə qaçırır. Yanlış alarmlar isə düzgün legit-lərdən daha yaşlı (39 vs 33), yüksək limitli (700 vs 428) və əvvəlki ünvanı çatışmayan (92% vs 66%) müştərilərdir. Bu təsviri müqayisədir, səbəb sübutu deyil.

**Zamana görə cross-validation nə göstərdi?** Genişlənən pəncərə ilə 5 fold (ay 3-7): ROC-AUC 0.902 ± 0.009 (0.913, 0.910, 0.897, 0.893, 0.897), PR-AUC 0.186 ± 0.019, TPR@5%FPR 0.553 ± 0.025. Tək split nəticəsi təsadüf deyil. ROC-AUC son aylarda bir qədər aşağıdır və yeni aylar əlavə olunsa da artmayıb, bu driftlə uyğundur (fold-lar fərqli aylar üzərində ölçüldüyü üçün bu müşahidəyə ehtiyatla yanaşmaq lazımdır).

## 12. Ən böyük məhdudiyyətlər
Xərc fərziyyələri illüstrativdir; tuning axtarışı məhduddur (genişləndirilmiş 10 sınaq yaxşılaşma göstərmədi, amma azdır); feature engineering ölçülə bilən qazanc vermədi; fairness fərqi aşkardır və tam həll olunmayıb; dataset sintetik/anonimləşdirilmişdir; bir train/valid/test bölgüsü var (təkrar split və ya rolling-window yoxlaması aparılmayıb).
