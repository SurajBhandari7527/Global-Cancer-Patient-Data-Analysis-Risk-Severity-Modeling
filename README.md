## 📌 Project Overview
A global dataset containing **50,000 cancer patient records (2015–2024)** was analyzed in this project. Patient demographics were explored, the effects of lifestyle and genetic risk factors on cancer severity were tested, treatment costs across different countries were evaluated, and Machine Learning models were trained to predict patient outcomes.

---

## 📁 Dataset Details
* **Total Patients:** 50,000
* **Total Columns:** 15 columns
* **Columns Included:** `Patient_ID`, `Age`, `Gender`, `Country_Region`, `Year`, `Genetic_Risk`, `Air_Pollution`, `Alcohol_Use`, `Smoking`, `Obesity_Level`, `Cancer_Type`, `Cancer_Stage`, `Treatment_Cost_USD`, `Survival_Years`, `Target_Severity_Score`
Link: https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024
Notebook: https://www.kaggle.com/code/surajbhandari527/cancer-data-analysis
---

## 🔬 Step-by-Step Data Analysis

### Step 1: Data Quality Check
* **Question/Problem:** Was the dataset clean, or were missing values and duplicate rows present?
* **Solution:** Dataset information was summarized using `data.info()`, and duplicate rows were calculated using `data.duplicated().sum()`.
* **Insight:** A total of **50,000 rows and 15 columns** with **0 missing values** and **0 duplicate rows** were identified. The dataset was found to be completely clean for further analysis.


---

### Step 2: Patient Age Distribution
* **Question/Problem:** What age range was represented by the patients in this study?
* **Solution:** A KDE (density) plot and a Histogram plot were generated using Seaborn, and descriptive statistics were calculated using `data["Age"].describe()`.
<img width="1227" height="478" alt="Screenshot 2026-08-12 175215" src="https://github.com/user-attachments/assets/4220bfb8-1fa5-4303-be8b-156541ad3c6d" />

* **Insight:** An average patient age of **54.4 years** was observed, ranging from a minimum of **20 years** to a maximum of **89 years**. Patient ages were found to be evenly distributed across the dataset.

---

### Step 3: Gender Distribution
* **Question/Problem:** How were patients distributed across male, female, and other gender categories?
* **Solution:** Value counts were calculated using `value_counts()`, and a Seaborn bar plot was created with exact numeric labels placed on top of each bar.
    <img width="728" height="557" alt="image" src="https://github.com/user-attachments/assets/316befb3-f9b7-4e9b-94c2-62fdfd9cbe15" />

* **Insight:** Patients were divided almost equally among three gender categories:
  * **Male:** 16,796 patients
  * **Female:** 16,709 patients
  * **Other:** 16,495 patients


---

### Step 4: Country and Region Distribution
* **Question/Problem:** Which countries were included in the dataset, and how were patients distributed among them?
<img width="576" height="513" alt="image" src="https://github.com/user-attachments/assets/a95cfc74-50e4-49e3-8f3b-7cdc8c35087e" />

* **Solution:** Patient counts for `Country_Region` were evaluated and displayed using a Pie Chart with percentage labels (`autopct`).
* **Insight:** Patients were found to be evenly distributed across multiple countries (including the UK, China, Pakistan, Brazil, and the USA), with no single country dominating the dataset.

---

### Step 5: Cancer Types Distribution
* **Question/Problem:** What were the most common cancer types recorded in the dataset?
<img width="718" height="557" alt="image" src="https://github.com/user-attachments/assets/965ff9e9-4360-424f-9120-f6c77c3e97c4" />

* **Solution:** Unique cancer types were identified using `.unique()`, and patient counts for each category were plotted using a labeled bar chart.
* **Insight:** Eight distinct cancer types (Lung, Leukemia, Breast, Colon, Skin, Cervical, Prostate, and Liver) were identified, with balanced patient counts observed across all categories.

---

### Step 6: Cancer Stages Distribution
* **Question/Problem:** How were patient diagnoses distributed across different cancer stages?
<img width="731" height="552" alt="image" src="https://github.com/user-attachments/assets/4d95b540-6b4a-4db2-b347-9d1442c57ed3" />

* **Solution:** Values for `Cancer_Stage` were counted and visualized using a bar chart across all stage levels.
* **Insight:** Cancer stages ranging from **Stage 0 to Stage IV** were recorded, and patient counts were found to be evenly balanced across all 5 stages.

---

### Step 7: Treatment Cost Distribution
* **Question/Problem:** What was the distribution of treatment costs among cancer patients?
* **Solution:** A KDE plot, a Histogram, and summary statistics (`.describe()`) were generated for `Treatment_Cost_USD`.
  <img width="1217" height="443" alt="Screenshot 2026-08-12 175805" src="https://github.com/user-attachments/assets/33f5e983-bce1-448f-aa46-5f76625d4f1c" />

* **Insight:** 
  * An **average cost** of $52,467 USD was calculated.
  * A **minimum cost** of $5,000 USD and a **maximum cost** of $99,999 USD were recorded.
  * Treatment costs were observed to be uniformly spread between $5,000 and $100,000 USD.

---

### Step 8: Risk Factors Overview
* **Question/Problem:** What average levels were recorded for various health and environmental risk factors?
* **Solution:** Risk factors (`Genetic_Risk`, `Air_Pollution`, `Alcohol_Use`, `Smoking`, and `Obesity_Level`) were summarized using `.agg(['mean', 'std', 'min', 'max'])`.
<img width="742" height="212" alt="image" src="https://github.com/user-attachments/assets/c8be7a15-204f-49c9-81e7-e6f3812512fd" />

* **Insight:** Risk scores ranging from **0 to 10** were recorded for all factors, with an average score of approximately **5.0** observed across each risk category.

---

### Step 9: Risk Factors vs. Cancer Severity Score
* **Question/Problem:** Which risk factor exerted the strongest influence on cancer severity scores?
* **Solution:** Linear regression analyses were conducted using `scipy.stats.linregress` for each risk factor against `Target_Severity_Score`, and trendlines were plotted.
<img width="1990" height="1190" alt="image" src="https://github.com/user-attachments/assets/c1e54bf5-8046-4a38-8cd7-adefedd37ca5" />

* **Insight:** 
  * The strongest positive relationship with cancer severity was shown by **Smoking ($r \approx 0.484$)** and **Genetic Risk ($r \approx 0.479$)**.
  * Moderate positive effects were demonstrated by **Air Pollution ($r \approx 0.367$)** and **Alcohol Use ($r \approx 0.363$)**.
  * The smallest effect was displayed by **Obesity Level ($r \approx 0.251$)**.

---

### Step 10: Early-Stage Diagnoses by Cancer Type
* **Question/Problem:** What proportion of patients was diagnosed at early stages (Stage 0 and Stage I) for each cancer type?
* **Solution:** Patient records for Stage 0 and Stage I were filtered, combined, and calculated as a percentage of total cases for each cancer type.
* **Insight:** Early-stage diagnosis proportions ranging between **38% and 40%** were observed across all 8 cancer types:
  * **Liver:** 40.6%
  * **Colon:** 40.4%
  * **Prostate:** 40.2%
  * **Cervical:** 39.8%
  * **Leukemia:** 39.5%
  * **Breast:** 39.4%
  * **Lung:** 38.4%

---

### Step 11: Feature Correlation Analysis
* **Question/Problem:** How were numerical features correlated with `Survival_Years` and `Target_Severity_Score`?
* **Solution:** Both **Pearson** (linear) and **Spearman** (rank) correlation matrices were calculated across numerical features using `data.corr()`.
<img width="981" height="407" alt="image" src="https://github.com/user-attachments/assets/4b900088-f8f0-4ffa-a40b-086370e79e1f" />

* **Insight:** 
  * Strong to moderate positive correlations were exhibited between all risk factors and `Target_Severity_Score`.
  * Near-zero correlations (all values close to 0.00) were observed between any numerical feature and `Survival_Years`.

---

### Step 12: Machine Learning - Predicting Cancer Severity Score
* **Question/Problem:** Could cancer severity scores be accurately predicted using Machine Learning?
* **Solution:** Categorical text columns were encoded using `LabelEncoder`, data was split into 80% training and 20% testing sets, and a `RandomForestRegressor` model (200 trees) was trained and evaluated using $R^2$ score and feature importances.
* **Insight:** 
  * A **Training $R^2$ Score** of `0.969` was achieved.
  * A **Testing $R^2$ Score** of `0.775` was achieved.
  * High predictive performance was demonstrated, and **Smoking** and **Genetic Risk** were confirmed as the most critical predictive inputs.

---

### Step 13: Machine Learning - Predicting Survival Years
* **Question/Problem:** Could patient survival years be predicted using Machine Learning?
* **Solution:** A `RandomForestRegressor` model combined with `GridSearchCV` hyperparameter tuning was applied to predict `Survival_Years`.
* **Insight:** 
  * A **Testing $R^2$ Score** of `-0.0001` was obtained.
  * It was proven that `Survival_Years` contains uniform random noise in this dataset and cannot be predicted using demographic or clinical variables.

---

### Step 14: Treatment Cost by Demographics and Countries
* **Question/Problem:** Were treatment costs influenced by country, age group, or gender?
* **Solution:** Patient ages were categorized into 5 bins (`0-30`, `31-45`, `46-60`, `61-75`, `76+`), average costs were calculated using `.groupby()`, and grouped bar charts along with a pivot heatmap were plotted.
  <img width="1728" height="668" alt="image" src="https://github.com/user-attachments/assets/38e5c0c7-e805-481f-9dca-cc7a4809b31e" />

* **Insight:** Uniform average treatment costs (~$52,000 USD) were recorded across all age groups, genders, and countries. No demographic group was found to incur significantly higher or lower treatment costs.

---

### Step 15: Treatment Cost vs. Survival Years
* **Question/Problem:** Was longer patient survival associated with higher treatment expenditure?
* **Solution:** Pearson and Spearman correlation tests were performed using `scipy.stats`, and a linear regression line was plotted using `sns.regplot`.
<img width="690" height="518" alt="image" src="https://github.com/user-attachments/assets/c9fd5cb6-9b02-4ce7-9486-a8b1b56e3f14" />

* **Insight:** 
  * **Pearson Correlation:** `-0.0004` ($p = 0.923$)
  * **Spearman Correlation:** `-0.0004` ($p = 0.920$)
  * Because $p > 0.05$, the null hypothesis was not rejected. Higher treatment expenditure was **not** found to be associated with longer survival.

---

### Step 16: Cancer Stage vs. Treatment Cost and Survival Years
* **Question/Problem:** Were higher treatment costs or reduced survival years caused by advanced cancer stages (Stage III/IV)?
* **Solution:** Data was grouped by `Cancer_Stage`, normality was tested using the **Shapiro-Wilk test** (`shapiro`), and non-parametric **Kruskal-Wallis tests** (`kruskal`) were performed across stage groups.
* **Insight:** 
  * A p-value of **0.425** was obtained for treatment cost across stages.
  * A p-value of **0.603** was obtained for survival years across stages.
  * It was proven that cancer stage exerts **no statistically significant effect** on treatment cost or survival duration in this dataset.

---

## 🎯 Summary of Key Findings
1. **Smoking and Genetic Risk** were identified as the primary drivers of cancer severity.
2. **Cancer severity scores** were accurately predicted using a Random Forest model ($R^2 = 0.775$).
3. **Treatment costs** were found to be uniformly distributed (~$52.4k average) and were not affected by country, age, gender, or cancer stage.
4. **Survival years** were shown to operate independently of treatment costs, cancer stage, and risk factors.
