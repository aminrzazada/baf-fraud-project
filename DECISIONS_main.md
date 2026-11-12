# Qərarlar jurnalı

Hər qərar: **nə → niyə → alternativ**. Rəqəmlər notebook çıxışlarındadır.

## Data və split
| Qərar | Səbəb | Alternativ və niyə seçilmədi |
|---|---|---|
| BAF dataseti | 2022 NeurIPS, real bank datasına əsaslanır, ~1% fraud, `month` var, threshold/calibration üçün təbii uyğundur | Titanic/Telco/ULB Credit Card: hamı işləyir, zaman ölçüsü yoxdur |
| Zamana görə split (0-4 / 5 / 6-7) | Production-da model keçmişdən öyrənib gələcəyi proqnozlaşdırır. Təsadüfi split gələcək məlumatı train-ə qatıb nəticəni şişirər | Random split: sızıntı. K-fold CV: eyni səbəb |
| Ayrıca valid ayı | Tuning, calibration, threshold test-i "yandırmasın" deyə | Yalnız train/test: test üzərində seçim edilərdi |
| `month` feature deyil | Split açarıdır. Gələcək ay train-də görünməyən dəyər olar, model zamana yapışar | Feature kimi saxlamaq |

## Preprocessing
| Qərar | Səbəb | Alternativ |
|---|---|---|
| `-1` → NaN + `*_missing` indikator | BAF-da `-1` real dəyər deyil, "yoxdur" kodudur. Rəqəm kimi saxlasaq model saxta əlaqə öyrənər. Missing-in özü fraud siqnalı ola bilər, ona görə indikator saxlanır | Sətirləri silmək: çox data və siqnal itərdi |
| `intended_balcon_amount` mənfi → NaN | Mənfi dəyərlər "yoxdur" mənasındadır (EDA-da yoxlanır) | - |
| Sabit sütun drop | Variansı 0 olan sütun məlumat vermir. Qərar kodla (`nunique<=1`) verilir | Əl ilə seçmək |
| Median imputation | Çəp paylanmalarda ortalamadan dayanıqlıdır | Mean: outlier-dən təsirlənir. KNN/iterative: 1M sətirdə bahalıdır |
| Outlier silinmir | Fraud özü kənar dəyərlərdə cəmləşə bilər, silmək hədəf siqnalını atardı. Xətti modeldə scaling, ağac modellərdə dayanıqlılıq kifayətdir | Winsorize/clip |
| StandardScaler | Logistic Regression üçün vacibdir. Ağac modellərə zərəri yoxdur, tək preprocessing yolu saxlanır | MinMax: outlier-ə həssasdır |
| OneHot (kateqoriklər) | Az səviyyəli nominal dəyişənlər, sıralama yoxdur. `handle_unknown="ignore"` app-da yeni dəyərə qarşı qoruyur | Label encoding: saxta sıra yaradır |

## Feature engineering (22 yeni feature)
| Qrup | Feature-lər | Məntiq |
|---|---|---|
| Missing indikatorları (6) | `*_missing` | Məlumat verməməyin özü siqnal |
| Velocity | `velocity_ratio_6h_24h`, `velocity_ratio_24h_4w`, `velocity_accel` | Qısa müddətli aktivlik sıçrayışı fraud üçün tipikdir |
| Maliyyə | `credit_limit_to_income`, `credit_risk_per_limit`, `balcon_to_limit` | Gəlirə/risk skoruna görə həddindən böyük limit tələbi |
| Yaşayış stabilliyi | `addr_months_total`, `current_addr_share_of_life` | Qısa yaşayış tarixçəsi riskli profil |
| Kimlik anomaliyası | `distinct_emails_total`, `phone_valid_count`, `email_risk` | Çox email/etibarsız telefon/pulsuz email + ad-email uyğunsuzluğu |
| Sessiya | `session_log`, `short_session` | Çox qısa sessiya bot davranışı ola bilər |
| Log transform | `zip_count_log`, `branch_count_log` | Ağır quyruq, xətti model üçün |
| İnteraksiya | `teleapp_foreign` | TELEAPP kanalı + xaricdən sorğu birgə risk |

**Ablation nəticəsi (real data):** FE olmadan (56 feature) test PR-AUC 0.1946, FE ilə (72 feature) 0.1930; valid-də 0.1876 vs 0.1906. Fərq kiçikdir və istiqaməti splitə görə dəyişir, ROC-AUC eynidir. Logistic Regression ilə eyni ablation-da FE kiçik, amma müsbət qazanc verdi (valid PR-AUC 0.1552 → 0.1615, test 0.1597 → 0.1650, `scripts/fe_ablation_linear.py`): xətti model interaksiyanı özü öyrənə bilmir, boosting isə öyrənir. Yəni əlavə feature-lər bu datasetdə ölçülə bilən qazanc vermədi. Top-20 permutation importance-da yalnız `email_risk` və iki missing indikator var. Ehtimal olunan səbəb: boosting qeyri-xətti kombinasiyaları özü öyrənir və BAF-ın feature-ləri artıq mühəndislik olunmuş formada gəlir (bu səbəb yoxlanılmayıb, fərziyyədir).

## Modellər
| Model | Niyə var |
|---|---|
| Logistic Regression | Sadə, interpretasiya olunan baseline. Daha mürəkkəb model onu keçməlidir |
| Random Forest | Qeyri-xətti, interaksiyaları özü tapır, bagging variansı azaldır |
| HistGradientBoosting | Tabular datada adətən ən güclü, NaN-ı birbaşa emal edir, böyük datada sürətlidir, əlavə kitabxana tələb etmir |

- **Imbalance:** `class_weight` (SMOTE yox). Sintetik fraud sətirləri real paylanmanı pozur, sonradan calibration-ı çətinləşdirir.
- **Seçim meyarı:** valid PR-AUC. Test seçimdə iştirak etmir.
- **XGBoost/LightGBM/CatBoost:** `scripts/compare_boosters.py` ilə yoxlanıb: valid PR-AUC 0.1881-0.1942, HGB tuned 0.1913. Paired bootstrap-a görə LightGBM və CatBoost HGB-dən əhəmiyyətli fərqlənmir, XGBoost bir qədər pisdir. HGB sadəlik (əlavə asılılıq yoxdur) səbəbilə saxlanıldı.

## Metrik
| Metrik | Rol | Səbəb |
|---|---|---|
| PR-AUC | Əsas | Fraud ~1%. Precision-recall mübadiləsini birbaşa ölçür, true negative-lərdən təsirlənmir |
| TPR@5%FPR | İkinci | BAF məqaləsinin əməliyyat metrikası: "5% yanlış alarmla fraudun neçə faizini tuturuq" |
| ROC-AUC | Məlumat üçün | Çoxlu true negative səbəbilə imbalance-də şişir |
| Accuracy | İstifadə edilmir | Hamıya "legit" desək ~99% alınır |

## Tuning, feature selection
- Tuning yalnız qalib model üçün, valid ayında (təsadüfi CV zaman sızıntısı verər). Az sayda random sınaq (6) məhdud hesablama büdcəsinə görədir.
- Permutation importance: model-agnostikdir, korrelyasiyalı feature-lərdə impurity importance-dən etibarlıdır.
- Seçim qaydası: müsbət importance-ın 95% kumulyativ payı (min 10) → 26 feature. Seçilmiş feature-lərlə model **yenidən qurulur** və valid-də paired bootstrap ilə müqayisə olunur: az feature-li model əhəmiyyətli dərəcədə pis deyilsə, o seçilir.
- **Nəticə:** ΔPR-AUC = −0.0057, 95% CI [−0.0105, −0.0015] → əhəmiyyətli dərəcədə pis, 72 feature saxlanıldı.
- **Dürüst qeyd:** ilk versiyada qayda 'valid PR-AUC-in ≥99%-i' idi. Nəticələri gördükdən sonra onu bootstrap testi ilə əvəz etdim, çünki 1% həddi özbaşına idi. Hər iki qayda eyni qərara (72 feature) gətirdi, yəni nəticə qayda dəyişikliyindən asılı olmadı.
- Tuning-də ən yaxşı parametrlər (max_depth=3, min_samples_leaf=20, l2=0) şəbəkənin kənarında idi. `scripts/wider_tuning.py` ilə genişləndirilmiş şəbəkədə 10 sınaq işlətdim: ən yaxşı valid PR-AUC 0.1919, mövcud 0.1913 (+0.0006, səs-küy daxilində), parametrlər dəyişmədi. Test seçimdə istifadə olunmadı. 10 sınaq azdır, nəticə 'səth düzdür' deməyə kifayətdir, 'optimum tapıldı' deməyə yox.

## Calibration və threshold
- `class_weight` ehtimalları şişirir, bank isə ehtimala görə qərar verir, ona görə calibration var. Sigmoid vs isotonic valid-in ayrıca hissəsində Brier ilə müqayisə olunur, amma yalnız sıralamanı (PR-AUC) pozmayan üsul seçilə bilər.
- **Dürüst qeyd:** ilk versiyada seçim yalnız Brier ilə idi. Sigmoid və isotonic arasında Brier fərqi 0.00002 idi (0.01042 vs 0.01044), yəni praktik olaraq bərabər. Notebook yenidən işlədiləndə (başqa scikit-learn versiyası) isotonic seçildi və test PR-AUC 0.194-dən 0.177-yə, TPR@5%FPR 0.543-dən 0.472-yə düşdü, çünki isotonic pilləli funksiyadır və bərabər skorlar yaradır. Ona görə qaydaya şərt əlavə etdim: calibration sıralamanı pozmamalıdır.
- Threshold 0.5 deyil: imbalance-də 0.5 demək olar heç nəyi fraud saymır. Threshold `C_FP/(C_FP+C_FN)` nəzəri optimumu ilə empirik xərc minimumu müqayisə edilir.
- Xərc fərziyyələri (FN=500, FP=10) illüstrativdir; həssaslıq cədvəli nisbət dəyişəndə nəticənin necə dəyişdiyini göstərir.

## Məlum məhdudiyyətlər (səmimi qeydlər)
- Importance valid üzərində hesablanır və seçim qərarı da valid-dədir: kiçik optimizm riski var. Test toxunulmaz olduğu üçün final rəqəm təmizdir.
- Threshold və calibrator eyni valid ayında seçilir; aylar arası drift olarsa test-də fərq görünə bilər.
- Train-test fərqi həm overfitting, həm də zaman driftini ehtiva edir. Valid-in ortada olması bunları ayırmağa kömək edir.

- Fairness: 50+ yaşlılarda FPR 32%, <50-də 14%. Real fraud nisbəti də fərqlidir (2.34% vs 0.83%), bu alarm nisbətini qismən izah edir, FPR fərqini birbaşa yox. Qrup-üzrə threshold və yaşsız model sınanıb (`scripts/age_ablation.py`): yaşsız modeldə FPR fərqi (ümumi FPR≤5% nöqtəsində) 9.1 → 6.6 faiz bəndi azalır, PR-AUC 0.1938 → 0.1906. Fərq tam aradan qalxmır (50+ yaşlılarda FPR 10.9%, <50-də 4.4%), final modeldə yaş saxlanıb.
- Xərc fərziyyəsi ilə optimal threshold 17.7% alarm verir, bu praktikada yoxlama komandasının tutumundan çox ola bilər. Tutum məhdudiyyəti olduqda FPR≤5% və ya həssaslıq cədvəlindən istifadə olunmalıdır.
