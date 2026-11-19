# 🌾 Machine Learning-Based Evapotranspiration Estimation for Irrigation Support

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Ready](https://img.shields.io/badge/Kaggle-Notebook%20Ready-20BEFF.svg)](https://www.kaggle.com/)
[![Data Source](https://img.shields.io/badge/Data-NASA%20POWER%20API-red.svg)](https://power.larc.nasa.gov/)
[![Framework](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

An applied machine learning and agro-meteorological research project aimed at accurately estimating daily **Reference Evapotranspiration ($ET_0$)** under limited meteorological data conditions for precision agricultural irrigation scheduling.

---

## 📌 1. Project Overview & Motivation

Reference evapotranspiration ($ET_0$) represents the rate of water loss from soil and reference crop plants. It is the international benchmark for determining crop water requirements and setting irrigation pump schedules to prevent over- or under-watering.

* **The Problem:** The standard benchmark method—the **FAO-56 Penman–Monteith equation**—requires an exhaustive suite of meteorological inputs: maximum & minimum temperatures, relative humidity, wind speed, solar radiation, and atmospheric pressure.
* **The Rural Challenge:** In rural agricultural weather stations across Northern India and developing regions, **precision solar radiation sensors (pyranometers, costing \$1,000–\$2,500 USD)** and cup anemometers are frequently absent, uncalibrated, or broken due to dust and weathering.
* **Core Research Question:**  
  > *Can machine learning models (Linear Regression, Random Forest, SVR) accurately approximate daily $ET_0$ using only basic, low-cost sensors (temperature and humidity, $<\$10$ USD), enabling farmers to eliminate expensive solar radiation sensors without compromising irrigation accuracy?*

---

## 📍 2. Study Area & Dataset Specifications

* **Location:** Hisar, Haryana, Northern India ($29.15^\circ\text{ N}, 75.72^\circ\text{ E}$, Elevation: $215\,\text{m}$)
* **Agro-Climatic Relevance:** Aligns directly with the wheat-rice cropping belt examined in the foundational study by *Shaloo et al.* (*Journal of Atmospheric and Solar-Terrestrial Physics*).
* **Observation Span:** **2010-01-01 to 2023-12-31** (14 full calendar years, **5,113 continuous daily records**).
* **Data Source:** NASA POWER Agroclimatology Daily Point API (combining GEOS/MERRA-2 atmospheric assimilation and CERES satellite solar irradiance).
* **Data Quality Audit:** **0 missing values (100% complete)**; verified row-by-row with 1:1 bit-exact precision.

### Meteorological Variable Summary (5,113 Days)

| Variable | Dataset Key | Units | Minimum | Mean $\pm$ Std | Maximum | Equivalent Hardware Sensor |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Max Air Temp** | `max_temp` | $^\circ\text{C}$ | $13.44$ | $33.15 \pm 7.20$ | $47.89$ | Thermistor / Digital Probe ($<\$5$) |
| **Min Air Temp** | `min_temp` | $^\circ\text{C}$ | $0.25$ | $19.49 \pm 8.18$ | $34.69$ | Thermistor / Digital Probe ($<\$5$) |
| **Mean Air Temp** | `mean_temp` | $^\circ\text{C}$ | $6.67$ | $25.87 \pm 7.92$ | $40.90$ | Calculated average |
| **Relative Humidity** | `relative_humidity` | $\%$ | $7.72$ | $42.29 \pm 17.76$ | $90.17$ | Capacitive Hygrometer ($<\$10$) |
| **Wind Speed (2m)** | `wind_speed` | $\text{m/s}$ | $0.45$ | $1.89 \pm 0.79$ | $6.08$ | 3-Cup Anemometer ($\approx \$50$) |
| **Solar Radiation** | `solar_radiation` | $\text{MJ/m}^2/\text{d}$ | $1.40$ | $17.52 \pm 5.69$ | $29.92$ | Precision Pyranometer ($>\$1,500$) |
| **Surface Pressure** | `surface_pressure` | $\text{kPa}$ | $96.67$ | $98.30 \pm 0.70$ | $99.83$ | Digital Barometric Sensor ($<\$10$) |
| **Target $ET_0$** | **`ET0`** | **$\text{mm/day}$** | **$0.66$** | **$5.14 \pm 2.37$** | **$14.53$** | **FAO-56 Ground-Truth Target ($y$)** |

---

## 📐 3. Mathematical Ground-Truth Target: FAO-56 Penman–Monteith

The standardized FAO-56 formulation (*Allen et al., 1998*) calculates daily reference evapotranspiration in $\text{mm/day}$:

$$ET_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T + 273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}$$

### Physical Sub-Components
1. **Psychrometric Constant ($\gamma$):** $\gamma = 0.000665 \times P$ ($\text{kPa}/^\circ\text{C}$).
2. **Slope of Saturation Vapor Pressure Curve ($\Delta$):** Rate of air moisture capacity expansion per degree rise.
3. **Vapor Pressure Deficit ($e_s - e_a$):** The drying power of air computed via Tetens equation and relative humidity.
4. **Solar Geometry & Net Radiation ($R_n$):** Net shortwave sunlight absorbed by reference grass (albedo $\alpha = 0.23$) minus outgoing longwave thermal radiation at night.
5. **Soil Heat Flux ($G$):** Daily $G \approx 0$.

> **Verification:** Benchmarked against Python's certified hydrology package (`pyet`). The mean absolute discrepancy across all 5,113 days is **$< 0.00012\,\text{mm/day}$**, verifying 100% mathematical implementation accuracy.

---

## 🤖 4. Machine Learning Experimental Design & Phase 3 Results

### Reduced-Input Feature Configurations Evaluated

| Configuration | Feature Set ($X$) | Target Hardware Simulation |
| :--- | :--- | :--- |
| **Config 1: Temp Only** | `max_temp`, `min_temp` | Ultra-low cost station ($<\$5$) |
| **Config 2: Temp + Humidity** | `max_temp`, `min_temp`, `relative_humidity` | Standard low-cost station ($<\$10$) |
| **Config 3: Temp + RH + Wind** | `max_temp`, `min_temp`, `relative_humidity`, `wind_speed` | Moderate-cost station ($\approx \$60$) |
| **Upper Bound Benchmark** | All weather variables + Solar Radiation | Complete station with pyranometer ($>\$1,500$) |

### Algorithms & Training Strategy
* **Linear Regression:** Baseline parametric reference.
* **Support Vector Regression (SVR):** Non-linear kernel regression (RBF kernel, standardized inputs).
* **Random Forest Regressor:** Primary ensemble learning model (150 estimators, max depth 12).
* **Data Split:** Chronological 80/20 train/test split (Train: 4,090 days [2010–2021], Test: 1,023 unseen days [2021–2023]) to prevent lookahead data leakage.

### 🏆 Phase 3 Experimental Benchmark Results (1,023 Unseen Test Days)

| Configuration | Model Architecture | MAE (mm/day) | RMSE (mm/day) | $R^2$ Score | Accuracy Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Config 1 (Temp Only)** | Linear Regression | 0.763 | 0.997 | 0.797 | Reasonable seasonal baseline |
| **Config 1 (Temp Only)** | Random Forest | 0.705 | 0.946 | 0.817 | Captures non-linear summer extremes |
| **Config 1 (Temp Only)** | Support Vector Regression (SVR) | 0.647 | 0.873 | 0.844 | Best temperature-only model |
| **Config 2 (Temp + RH)** | Linear Regression | 0.708 | 0.929 | 0.823 | Accounts for vapor deficit damping |
| **Config 2 (Temp + RH)** | Random Forest | 0.673 | 0.907 | 0.832 | Robust across seasonal shifts |
| **Config 2 (Temp + RH)** | Support Vector Regression (SVR) | 0.630 | 0.852 | 0.852 | Smooth non-linear humidity fit |
| **Config 3 (Temp + RH + Wind)** | Linear Regression | 0.498 | 0.662 | 0.910 | High precision linear approximation |
| **Config 3 (Temp + RH + Wind)** | **Random Forest Regressor** | **0.384** | **0.501** | **0.949** | **Near-ideal crop water prediction** |
| **Config 3 (Temp + RH + Wind)** | **Support Vector Regression (SVR)** | **0.354** | **0.478** | **0.953** | **Overall Best Model ($\mathbf{R^2 = 0.953}$)** |
| *Config 4 (Full Upper Bound)* | *Linear Regression* | *0.342* | *0.501* | *0.949* | *Upper bound reference* |
| *Config 4 (Full Upper Bound)* | *Random Forest Regressor* | *0.172* | *0.233* | *0.989* | *Ideal station with pyranometer* |
| *Config 4 (Full Upper Bound)* | *Support Vector Regression (SVR)* | *0.079* | *0.108* | *0.998* | *Theoretical ceiling* |

### 🌲 Random Forest Feature Importance Analysis
Gini impurity reduction from the 150-tree ensemble trained on Configuration 3:
* **Maximum Temperature (`max_temp`):** **`82.43%`** — Overwhelming thermal driver expanding atmospheric vapor capacity.
* **Wind Speed (`wind_speed`):** **`12.89%`** — Essential aerodynamic vapor transport stripping humid crop boundary layers.
* **Relative Humidity (`relative_humidity`):** **`3.82%`** — Modulates vapor pressure deficit.
* **Minimum Temperature (`min_temp`):** **`0.86%`** — Nighttime temperature baseline.

### 💡 Core Research Conclusion
> **Can ML estimate $ET_0$ accurately using fewer variables than FAO-56?**  
> **Yes.** Configuration 3 (Temperature + Humidity + Wind Speed) achieves **$R^2 = 0.953$ and $\text{RMSE} = 0.478\,\text{mm/day}$**, capturing over 95% of evapotranspiration variance without solar radiation sensors. This allows farm irrigation systems to eliminate expensive pyranometers ($>\$1,500$ USD) while maintaining high precision.

---

## 📂 5. Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── nasa_power_hisar_2010_2023.json       # Raw NASA POWER API JSON response
│   │   └── raw_weather_data.csv                  # Direct unaltered raw CSV
│   └── evapotranspiration_dataset_with_ET0.csv   # Clean ML-ready dataset with ground-truth ET0
├── reports/
│   ├── figures/                                  # High-resolution evaluation charts
│   │   ├── feature_importance.png                # RF Gini feature importance bar chart
│   │   ├── model_comparison_metrics.png          # Side-by-side R2 & RMSE benchmark chart
│   │   ├── predicted_vs_actual_scatter.png       # 1:1 scatter plots across configs
│   │   └── timeseries_prediction_overlay.png     # Actual vs. Predicted time series (2023)
│   ├── model_benchmark_results.csv               # Complete CSV of all 12 model evaluations
│   ├── Phase_3_Model_Evaluation_Report.pdf       # Phase 3 technical benchmark report
│   ├── Phase_1_and_2_Detailed_Report.pdf         # Comprehensive Phase 1 & 2 technical report
│   └── Project_Progress_Report_2_Pages.pdf       # 2-Page student progress report
├── src/                                          # Reusable Python modules
│   ├── __init__.py
│   ├── data_fetcher.py                           # Automated NASA POWER API query module
│   ├── et0_calculator.py                         # FAO-56 Penman-Monteith physical calculator
│   └── preprocessor.py                           # Cleaning, missing-value audit & formatting
├── evapotranspiration_fao56_calculation.ipynb    # Kaggle-ready notebook (Phase 1 & Phase 2)
├── evapotranspiration_estimation_complete.ipynb  # End-to-end master notebook with ML models
├── fetch_and_prepare_data.py                     # Data extraction and target generation script
├── train_models.py                               # Phase 3 model training & evaluation engine
├── requirements.txt                              # Python environment dependencies
├── LICENSE                                       # MIT License
└── README.md                                     # Project documentation
```

---

## 🚀 6. Getting Started

### Local Setup
```bash
# 1. Clone repository
git clone https://github.com/vipulagarwal110-create/ML-Evapotranspiration-Estimation.git
cd ML-Evapotranspiration-Estimation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run data preparation pipeline
python fetch_and_prepare_data.py
```

### Running on Kaggle
1. Upload `evapotranspiration_fao56_calculation.ipynb` to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Upload `data/raw/raw_weather_data.csv` as input dataset.
3. Click **Run All**. The notebook automatically detects input paths, calculates $ET_0$, renders visualizations, and provides a 1-click download link for the final processed dataset.

---

## 📚 7. Key Literature & References

1. **Shaloo et al.** *"Modeling daily reference evapotranspiration and evaluating uncertainty analysis in machine learning under limited meteorological data conditions for Northern India."* *Journal of Atmospheric and Solar-Terrestrial Physics*.
2. **Faloye, O. T. et al. (2026).** *"Performance evaluation of machine learning algorithms for estimating reference evapotranspiration based on NASA POWER weather data."* *Frontiers in Artificial Intelligence*.
3. **Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).** *"Crop evapotranspiration: Guidelines for computing crop water requirements."* *FAO Irrigation and Drainage Paper No. 56*, Food and Agriculture Organization of the United Nations, Rome.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
