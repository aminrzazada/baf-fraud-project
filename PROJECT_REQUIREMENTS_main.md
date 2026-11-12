# Müəllimin tələbləri: status

| Tələb (müəllimin siyahısı) | Status | Harada |
|---|---|---|
| Python Jupyter Notebook | Hazırdır, real data ilə işlədilib, çıxışlarla | `01_eda.ipynb`, `02_modeling.ipynb`, `03_extended_analysis.ipynb` |
| GitHub | Repo hazırdır, push sizdən | bu qovluq |
| Dokumentasiya (md) | Hazırdır | `README.md`, `DECISIONS.md`, `DEFENSE_QA.md` |
| Streamlit deploy | Kod hazırdır, lokal test və deploy sizdən | `app.py` |
| 2-3 fərqli model | 3 model (xətti, bagging, boosting) | modeling §6 |
| Əsas metrik və seçim əsası | PR-AUC, valid üzrə | modeling §6, DECISIONS.md |
| Train/test score fərqi | Cədvəl və qrafik | modeling §6 |
| Pipeline | sklearn Pipeline | `src/features.py`, modeling §5 |
| Koda komment | Notebook-larda var, `src/` yoxlanılmalıdır | notebook-lar |
| Funksiyalara qısa izah (docstring) | Notebook funksiyalarında tam, `src/` yoxlanılmalıdır | notebook-lar |
| Notebook-da uzun mətn yoxdur | Markdown hər bölmədə 1-3 cümlə | notebook-lar |
| Düzgün başlıqlar (Preprocessing və s.) | Var | notebook-lar |
| Vizuallar | 26 qrafik | notebook-lar |
| Best features seçib modeli yenidən qurmaq | 72 → 26, model yenidən quruldu, paired bootstrap ilə müqayisə | modeling §8 |
| Threshold optimization | Xərcə görə + həssaslıq | modeling §10 |
| Calibration | Sigmoid/isotonic, Brier/ECE | modeling §9 |
| Geniş feature engineering | 22 yeni feature, ablation ilə | `src/features.py`, modeling §4 |
| Hər model/parametr/qərarın əsaslandırılması | Parametr cədvəli və qərar cədvəlləri | `DEFENSE_QA.md` §5, `DECISIONS.md` |
| Əlavə (tələb olunmayıb) | Sabitlik, drift, lift/KS, CI, xəta analizi, PDP, zamana görə CV | `03_extended_analysis.ipynb` |

## Sizdən qalan addımlar
1. `pip install -r requirements.txt` və `streamlit run app.py` ilə app-ı lokal yoxlayın (bu mühitdə Streamlit sınanmayıb).
2. Deploy-dan sonra model yüklənməsə: notebook 02-ni lokal işlədib yeni `models/model.joblib` commit edin (`scikit-learn==1.9.1` pinlənib, `model.joblib` bu versiya ilə hazırlanıb).
3. GitHub-a push, Streamlit Cloud-da deploy.
4. `02_modeling.ipynb`-ni yenidən işlədin (Run All): calibration seçim qaydası düzəldilib, `models/` və README cədvəli yenilənəcək. Sonra `pip install -r requirements-extra.txt` edib bu skriptləri repo kökündən işlədin və nəticələri `DEFENSE_QA.md`-yə yazın: `compare_boosters.py` (§5, hazırdır), `fe_ablation_linear.py` (§6), `age_ablation.py` (§9), `wider_tuning.py` (§5).
5. `.gitignore` faylı bu qovluqda artıq düzgün adladır.
6. `DEFENSE_QA.md`-ni oxuyub sualları səsli cavablandırmağa məşq edin.
