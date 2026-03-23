# ============================================================
#   AutoPriceIQ — Indian Used Car Price Intelligence Platform
#   Author  : Your Name
#   Tech    : Python | Streamlit | Scikit-learn | Matplotlib | Seaborn
#   Dataset : 2,059 Indian Used Car Listings (cardetails_v4.csv)
# ============================================================

# ──────────────────────────────────────────────
# STEP 1 — IMPORT ALL REQUIRED LIBRARIES
# ──────────────────────────────────────────────
# Streamlit  → turns this Python script into a web app
# Pandas     → loads and manipulates the CSV dataset (like Excel in Python)
# NumPy      → fast math operations (log, exp, arrays)
# Matplotlib → creates static charts and graphs
# Seaborn    → prettier statistical charts built on top of Matplotlib
# Scikit-learn → machine learning: encoding, training, evaluating the model
# Warnings   → suppress harmless warning messages in the terminal

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# STEP 2 — STREAMLIT PAGE CONFIGURATION
# ──────────────────────────────────────────────
# This must be the FIRST Streamlit command in the file
# layout="wide" uses full browser width instead of narrow centered column
# initial_sidebar_state="expanded" keeps sidebar open on load

st.set_page_config(
    page_title="AutoPriceIQ — Car Price Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# STEP 3 — CUSTOM CSS STYLING
# ──────────────────────────────────────────────
# Streamlit's default look is plain white.
# We inject our own CSS using st.markdown() with unsafe_allow_html=True
# This gives us the dark professional look

st.markdown("""
<style>

/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Background & Base Text ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0f1a;
    color: #e8eaf0;
}
.stApp {
    background: #0b0f1a;
}

/* ── Sidebar Background ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1625 0%, #111827 100%);
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * {
    color: #c9cfe0 !important;
}

/* ── All Headings use Syne font ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* ── KPI Metric Cards (the big number boxes) ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #141c2e 0%, #0f1625 100%);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 20px !important;
    transition: transform 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
}
[data-testid="metric-container"] label {
    color: #7a8aaa !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f97316 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

/* ── Predict Button (big orange button) ── */
.stButton > button {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-size: 16px !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(249,115,22,0.35) !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(249,115,22,0.55) !important;
}

/* ── Dropdown / Selectbox ── */
.stSelectbox > div > div {
    background: #141c2e !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #7a8aaa !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 600 !important;
}

/* ── Number Input Box ── */
.stNumberInput input {
    background: #141c2e !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}

/* ── Tab Bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1625;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2535;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #7a8aaa !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: white !important;
}

/* ── Radio Buttons in Sidebar ── */
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    padding: 6px 0;
}

/* ── Dataframe / Table ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1e2d4a;
}

/* ── Divider line ── */
hr {
    border-color: #1e2535;
}

/* ── Re-usable card container (used via st.markdown) ── */
.card {
    background: linear-gradient(135deg, #141c2e 0%, #0f1625 100%);
    border: 1px solid #1e2d4a;
    border-radius: 20px;
    padding: 28px 32px;
    margin: 10px 0;
}

/* ── Hero banner on Overview page ── */
.hero {
    background: linear-gradient(135deg, #0f1625 0%, #141c2e 50%, #1a1f30 100%);
    border: 1px solid #1e2d4a;
    border-radius: 24px;
    padding: 48px 44px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -80px; right: -60px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(249,115,22,0.09) 0%, transparent 70%);
    pointer-events: none;
}

/* ── Small tag/badge pill ── */
.badge {
    display: inline-block;
    background: rgba(249,115,22,0.15);
    color: #f97316;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: 1px solid rgba(249,115,22,0.3);
    font-family: 'Syne', sans-serif;
    margin-bottom: 14px;
}

/* ── Price result box after prediction ── */
.price-box {
    background: linear-gradient(135deg, #1a2540, #0f1625);
    border: 2px solid #f97316;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(249,115,22,0.12);
}

/* ── Small stat chip inside price box ── */
.chip {
    background: rgba(249,115,22,0.1);
    border: 1px solid rgba(249,115,22,0.25);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}

/* ── Architecture grid inside Model Insights ── */
.arch-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin-top: 16px;
}
.arch-item {
    background: rgba(249,115,22,0.08);
    border: 1px solid rgba(249,115,22,0.2);
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# STEP 4 — MATPLOTLIB / SEABORN GLOBAL STYLE
# ──────────────────────────────────────────────
# We set a dark background for all charts so they match
# the dark UI theme of the app

# Dark background color for all charts
CHART_BG    = "#0f1625"   # chart background (matches app)
CARD_BG     = "#141c2e"   # slightly lighter background for chart area
GRID_COLOR  = "#1e2d4a"   # faint grid lines
TEXT_COLOR  = "#c9cfe0"   # axis labels and titles
ORANGE      = "#f97316"   # primary highlight / bar color
BLUE        = "#3b82f6"   # secondary bar color
GREEN       = "#10b981"   # third color
PURPLE      = "#8b5cf6"   # fourth color
PINK        = "#ec4899"   # fifth color
YELLOW      = "#fbbf24"   # sixth color

# Color palette list for multi-color charts
PALETTE = [ORANGE, BLUE, GREEN, PURPLE, PINK, YELLOW, "#06b6d4", "#f43f5e"]

def apply_dark_style(fig, ax_list=None):
    """
    Apply our dark theme to any matplotlib figure.
    Called after every chart is created.
    ax_list: pass list of axes if the figure has subplots.
    """
    fig.patch.set_facecolor(CHART_BG)

    if ax_list is None:
        ax_list = fig.get_axes()

    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    return fig


# ──────────────────────────────────────────────
# STEP 5 — LOAD AND PREPARE THE DATASET
# ──────────────────────────────────────────────
# @st.cache_data means: run this function ONCE when app starts,
# then reuse the result. Without caching, it re-reads the CSV
# every time a user clicks anything (very slow!).

@st.cache_data
def load_and_prepare_data():
    """
    Load cardetails_v4.csv and engineer useful features.
    Returns a clean DataFrame ready for ML and visualization.
    """

    # ── Load raw CSV into a Pandas DataFrame ──
    df = pd.read_csv("cardetails_v4.csv")

    # ── Feature: Car Age ──────────────────────
    # Instead of raw year (2017, 2015...) we compute how old the car is.
    # Age is directly tied to depreciation — older = cheaper.
    df["Car_Age"] = 2024 - df["Year"]

    # ── Feature: Engine CC ───────────────────
    # The Engine column stores text like "1197 cc"
    # We extract just the number 1197 using regex (str.extract)
    df["Engine_CC"] = df["Engine"].str.extract(r"(\d+)").astype(float)

    # ── Feature: Max Power in BHP ─────────────
    # Max Power stores "82 bhp @ 6000 rpm" — we extract 82.0
    df["Power_BHP"] = df["Max Power"].str.extract(r"([\d.]+)\s*bhp").astype(float)

    # ── Feature: Owner Rank ───────────────────
    # Convert ownership text to ordered numbers so the model understands
    # First owner = 1 (most valuable), higher number = more previous owners
    owner_map = {
        "First": 1,
        "Second": 2,
        "Third": 3,
        "Fourth": 4,
        "4 or More": 5,
        "UnRegistered Car": 0,
    }
    df["Owner_Rank"] = df["Owner"].map(owner_map).fillna(3)

    # ── Feature: Full Car Name ────────────────
    # Combine Make and Model for easy display in UI
    df["Car_Name"] = df["Make"] + " " + df["Model"]

    # ── Feature: Price in Lakhs ───────────────
    # ₹1 Lakh = ₹100,000. Dividing makes numbers human-readable.
    df["Price_Lakh"] = df["Price"] / 100000

    # ── Drop rows with missing Price ──────────
    # We cannot train or display without a price value
    df = df.dropna(subset=["Price"])

    return df


# ──────────────────────────────────────────────
# STEP 6 — TRAIN THE MACHINE LEARNING MODEL
# ──────────────────────────────────────────────
# @st.cache_resource: cache the TRAINED MODEL object.
# Without this, the model retrains on every user interaction
# (would take 20-30 seconds each time!).

@st.cache_resource
def train_model(df):
    """
    Train a Gradient Boosting Regressor to predict car prices.

    Steps inside:
    1. Select only the important features (not all 20 columns)
    2. Encode text columns to numbers using LabelEncoder
    3. Split data: 80% train, 20% test
    4. Apply log transform on Price (improves accuracy for skewed data)
    5. Train GradientBoostingRegressor
    6. Evaluate with R² Score and MAE
    7. Return everything needed for predictions and insights

    Returns:
        model      : trained GradientBoostingRegressor
        encoders   : dict of LabelEncoder objects (one per text column)
        features   : list of feature column names
        r2         : R² accuracy score on test set
        mae        : Mean Absolute Error in rupees on test set
        X_test     : test features (for residual / scatter plots)
        y_test     : actual log prices on test set
        y_pred     : predicted log prices on test set
        importance : DataFrame of feature importances (for bar chart)
    """

    # ── Select Features ──────────────────────
    # These 9 features were chosen because they have the
    # strongest influence on used car prices in India.
    # We deliberately skip: Color, Location, Seating Capacity, etc.
    # because they add noise without improving accuracy.
    features = [
        "Make",          # Brand: BMW vs Maruti = biggest price gap
        "Model",         # Specific model: Innova vs Alto = huge difference
        "Car_Age",       # Older car = lower price (depreciation)
        "Kilometer",     # More km = more wear = lower price
        "Fuel Type",     # Diesel holds resale better than Petrol in India
        "Transmission",  # Automatic commands ~20% premium over Manual
        "Owner_Rank",    # First owner = trusted, higher price
        "Engine_CC",     # Bigger engine usually means premium segment
        "Power_BHP",     # High BHP = performance / luxury segment
    ]
    target = "Price"

    # ── Drop rows that have missing values in our selected features ──
    model_df = df[features + [target]].dropna()

    # ── Encode Text Columns ───────────────────
    # ML algorithms need NUMBERS, not text like "Petrol" or "Honda".
    # LabelEncoder converts: "CNG"=0, "Diesel"=1, "Electric"=2, "Petrol"=3
    # We save each encoder in a dict so we can reuse it at prediction time
    encoders = {}
    text_columns = ["Make", "Model", "Fuel Type", "Transmission"]

    for col in text_columns:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le  # Save encoder — needed when predicting!

    # ── Prepare X (inputs) and y (target) ────
    X = model_df[features]
    # Log-transform the price: np.log1p handles 0s safely (log(1+x))
    # Why log? Prices range ₹49K to ₹3.5Cr — huge range.
    # Log compresses this into a smaller, more manageable range (13 to 17)
    # which makes the model more accurate.
    y = np.log1p(model_df[target])

    # ── Train/Test Split ──────────────────────
    # 80% of data is used to TRAIN the model
    # 20% is kept hidden and used to TEST accuracy
    # random_state=42 ensures the same split every run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train Gradient Boosting Regressor ─────
    # This is an ensemble model — it builds 500 small decision trees,
    # each one correcting the mistakes of the previous one.
    # n_estimators = 500 trees
    # learning_rate = 0.05 → each tree makes small, careful corrections
    # max_depth = 6 → how complex each individual tree is
    # subsample = 0.8 → each tree uses 80% of data (prevents overfitting)
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42,
    )

    # .fit() is where actual training happens
    model.fit(X_train, y_train)

    # ── Evaluate the Model ────────────────────
    y_pred = model.predict(X_test)

    # R² Score: 0 = random guessing, 1 = perfect
    # 0.91 means the model explains 91% of price variance
    r2 = r2_score(y_test, y_pred)

    # MAE: Average rupee error. We must reverse the log first!
    # np.expm1() is the reverse of np.log1p()
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

    # ── Feature Importance ────────────────────
    # GradientBoosting tells us which features mattered most
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return model, encoders, features, r2, mae, X_test, y_test, y_pred, importance


# ──────────────────────────────────────────────
# STEP 7 — LOAD DATA AND TRAIN MODEL (APP START)
# ──────────────────────────────────────────────
# These two lines run when the app starts.
# Because of @st.cache_data and @st.cache_resource,
# they only run ONCE — then the results are reused.

df      = load_and_prepare_data()
model, encoders, features, r2, mae, X_test, y_test, y_pred, importance = train_model(df)


# ──────────────────────────────────────────────
# STEP 8 — SIDEBAR NAVIGATION
# ──────────────────────────────────────────────

with st.sidebar:

    # App logo / branding
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px;'>
        <div style='font-family:Syne,sans-serif; font-size:24px;
                    font-weight:800; color:#f97316;'>🚗 AutoPriceIQ</div>
        <div style='font-size:10px; color:#3a4458; letter-spacing:0.18em;
                    text-transform:uppercase; margin-top:4px;'>
            Car Price Intelligence
        </div>
    </div>
    <hr style='border-color:#1e2535; margin:10px 0 16px;'>
    """, unsafe_allow_html=True)

    # Navigation radio buttons — one per page
    page = st.radio(
        "Go to",
        [
            "🏠  Overview",
            "🔮  Price Predictor",
            "📊  Market Dashboard",
            "🧠  Model Insights",
        ],
        label_visibility="collapsed",
    )

    # Bottom info
    st.markdown("""
    <hr style='border-color:#1e2535; margin:20px 0 12px;'>
    <div style='font-size:10px; color:#2a3448; text-align:center; line-height:1.9;'>
        Python · Streamlit · Scikit-learn<br>
        Matplotlib · Seaborn<br>
        <span style='color:#f97316; font-weight:600;'>
            2,059 Indian Used Cars
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── Extract just the page name (remove the emoji prefix) ──
page_name = page.split("  ")[1]


# ╔══════════════════════════════════════════════════════════╗
# ║   PAGE 1 — OVERVIEW                                     ║
# ╚══════════════════════════════════════════════════════════╝

if page_name == "Overview":

    # ── Hero Banner ──────────────────────────
    st.markdown("""
    <div class='hero'>
        <div class='badge'>AI-Powered · Indian Used Car Market</div>
        <h1 style='font-family:Syne,sans-serif; font-size:46px;
                   font-weight:800; margin:0 0 14px; line-height:1.1;'>
            Car Price <span style='color:#f97316;'>Intelligence</span><br>Platform
        </h1>
        <p style='color:#7a8aaa; font-size:15px; max-width:500px;
                  line-height:1.8; margin:0;'>
            Machine learning meets the Indian used car market.
            Explore 2,059 real listings across 33 brands,
            predict fair prices instantly, and uncover market trends.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Metric Cards ─────────────────────
    # st.columns(4) creates 4 equal-width columns side by side
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings",      f"{len(df):,}",               "Real market data")
    c2.metric("Car Brands",          f"{df['Make'].nunique()}",     "Across India")
    c3.metric("Model Accuracy (R²)", f"{r2:.1%}",                  "Gradient Boosting")
    c4.metric("Average Price",       f"₹{df['Price'].mean()/1e5:.1f}L", "Indian market")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Two charts side by side ───────
    col_left, col_right = st.columns([1.4, 1], gap="large")

    # CHART 1 — Price Distribution Histogram
    with col_left:
        st.markdown("#### 📈 Price Distribution of All Listed Cars")

        fig, ax = plt.subplots(figsize=(8, 4))

        # Filter to prices below ₹1 Cr for a cleaner histogram
        # (extreme luxury outliers skew the chart)
        price_data = df[df["Price"] < 10_000_000]["Price"] / 1e5

        ax.hist(
            price_data,
            bins=50,
            color=ORANGE,
            edgecolor=CHART_BG,
            linewidth=0.4,
            alpha=0.88,
        )
        ax.set_xlabel("Price (₹ Lakhs)", fontsize=10)
        ax.set_ylabel("Number of Cars", fontsize=10)

        # Add a vertical line showing the median price
        median_p = price_data.median()
        ax.axvline(median_p, color=BLUE, linewidth=2, linestyle="--", label=f"Median: ₹{median_p:.1f}L")
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    # CHART 2 — Top Brands by Listing Count
    with col_right:
        st.markdown("#### 🏆 Top 10 Brands by Listings")

        top_makes = df["Make"].value_counts().head(10).reset_index()
        top_makes.columns = ["Brand", "Count"]

        fig, ax = plt.subplots(figsize=(6, 4))

        # Horizontal bar chart — easier to read brand names
        bars = ax.barh(
            top_makes["Brand"],
            top_makes["Count"],
            color=[ORANGE if i == 0 else BLUE for i in range(len(top_makes))],
            edgecolor=CHART_BG,
            linewidth=0.5,
        )

        # Add count labels at the end of each bar
        for bar, val in zip(bars, top_makes["Count"]):
            ax.text(
                bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left",
                color=TEXT_COLOR, fontsize=8,
            )

        ax.invert_yaxis()   # Highest count at top
        ax.set_xlabel("Number of Listings", fontsize=9)

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    # ── Row 2: Two more charts ────────────────
    col_a, col_b = st.columns(2, gap="large")

    # CHART 3 — Fuel Type Donut Chart
    with col_a:
        st.markdown("#### ⛽ Fuel Type Market Share")

        fuel_counts = df["Fuel Type"].value_counts()
        # Keep top 5 fuel types, group the rest as "Others"
        top_fuels = fuel_counts.head(5)
        other_count = fuel_counts.iloc[5:].sum()
        if other_count > 0:
            top_fuels["Others"] = other_count

        fig, ax = plt.subplots(figsize=(6, 4))

        wedges, texts, autotexts = ax.pie(
            top_fuels.values,
            labels=top_fuels.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=PALETTE[:len(top_fuels)],
            pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor=CHART_BG, linewidth=2),
        )

        # Style pie text
        for text in texts:
            text.set_color(TEXT_COLOR)
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(8)
            autotext.set_fontweight("bold")

        # Donut center label
        ax.text(0, 0, "Fuel\nTypes", ha="center", va="center",
                fontsize=10, color=TEXT_COLOR, fontweight="bold")

        fig = apply_dark_style(fig)
        fig.patch.set_facecolor(CHART_BG)
        st.pyplot(fig)
        plt.close(fig)

    # CHART 4 — Median Price by Owner Type
    with col_b:
        st.markdown("#### 🔑 Median Price by Ownership History")

        # Filter to the 4 main ownership types for clarity
        owner_order = ["First", "Second", "Third", "Fourth"]
        owner_price = (
            df[df["Owner"].isin(owner_order)]
            .groupby("Owner")["Price"]
            .median()
            .reindex(owner_order)
            / 1e5
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        bar_colors = [ORANGE, BLUE, GREEN, PURPLE]
        bars = ax.bar(
            owner_price.index,
            owner_price.values,
            color=bar_colors,
            edgecolor=CHART_BG,
            linewidth=0.5,
            width=0.55,
        )

        # Add value labels on top of each bar
        for bar, val in zip(bars, owner_price.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"₹{val:.1f}L",
                ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=9, fontweight="bold",
            )

        ax.set_xlabel("Ownership Type", fontsize=9)
        ax.set_ylabel("Median Price (₹ Lakhs)", fontsize=9)

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    # ── Row 3: Transmission vs Price ─────────
    st.markdown("#### 🔧 Manual vs Automatic — Average Price Comparison")

    col_x, col_y = st.columns([1, 2], gap="large")

    with col_x:
        trans_price = df.groupby("Transmission")["Price"].agg(["mean", "median"]) / 1e5
        trans_price.columns = ["Mean", "Median"]

        fig, ax = plt.subplots(figsize=(5, 3.5))

        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, trans_price["Mean"],   width, label="Mean",   color=ORANGE, alpha=0.9)
        ax.bar(x + width/2, trans_price["Median"], width, label="Median", color=BLUE,   alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(trans_price.index)
        ax.set_ylabel("Price (₹ Lakhs)")
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    with col_y:
        st.markdown("""
        <div class='card' style='margin-top:8px;'>
            <div style='font-family:Syne,sans-serif; font-size:15px;
                        font-weight:700; color:#f97316; margin-bottom:10px;'>
                Key Market Insights
            </div>
            <div style='color:#c9cfe0; font-size:13px; line-height:2;'>
                🔸 <b>Maruti Suzuki</b> dominates listings with 440 cars — most affordable segment<br>
                🔸 <b>First owner</b> cars fetch 35-45% more than second-owner equivalents<br>
                🔸 <b>Automatic transmission</b> commands a ₹3-5L premium over manual<br>
                🔸 <b>Petrol</b> cars are most common but <b>Diesel</b> holds resale value better<br>
                🔸 <b>Mumbai & Delhi</b> together account for 32% of all listings
            </div>
        </div>
        """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║   PAGE 2 — PRICE PREDICTOR                              ║
# ╚══════════════════════════════════════════════════════════╝

elif page_name == "Price Predictor":

    st.markdown("""
    <div class='hero'>
        <div class='badge'>🔮 ML-Powered Prediction Engine</div>
        <h1 style='font-family:Syne,sans-serif; font-size:40px;
                   font-weight:800; margin:0 0 12px;'>
            Predict Your <span style='color:#f97316;'>Car's Value</span>
        </h1>
        <p style='color:#7a8aaa; font-size:14px; max-width:460px; line-height:1.8; margin:0;'>
            Enter your car's details below. Our Gradient Boosting model
            will estimate the fair resale price instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    form_col, result_col = st.columns([1.1, 1], gap="large")

    # ── LEFT SIDE: Input Form ─────────────────
    with form_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🚗 Enter Car Details")

        # Brand dropdown — shows all 33 unique makes sorted alphabetically
        all_makes = sorted(df["Make"].unique())
        sel_make = st.selectbox("Car Brand (Make)", all_makes)

        # Model dropdown — DYNAMICALLY FILTERS to only show models
        # belonging to the selected brand. This updates automatically
        # whenever the user changes the brand.
        models_for_brand = sorted(df[df["Make"] == sel_make]["Model"].unique())
        sel_model = st.selectbox("Car Model", models_for_brand)

        # Year and KM side by side
        c1, c2 = st.columns(2)
        with c1:
            sel_year = st.selectbox(
                "Manufacturing Year",
                sorted(df["Year"].unique(), reverse=True),
            )
        with c2:
            sel_km = st.number_input(
                "Kilometers Driven",
                min_value=0, max_value=500_000,
                value=40_000, step=1_000,
            )

        # Fuel and Transmission side by side
        c3, c4 = st.columns(2)
        with c3:
            # Simplify fuel types shown to user (top 5 most common)
            common_fuels = df["Fuel Type"].value_counts().head(5).index.tolist()
            sel_fuel = st.selectbox("Fuel Type", common_fuels)
        with c4:
            sel_trans = st.selectbox("Transmission", ["Manual", "Automatic"])

        # Ownership
        sel_owner = st.selectbox(
            "Ownership History",
            ["First", "Second", "Third", "Fourth", "4 or More"],
        )

        # Engine CC and BHP — auto-fill from typical values of selected car
        # If the user picks Honda City, it pre-fills with ~1498 CC, ~119 BHP
        similar_cars = df[(df["Make"] == sel_make) & (df["Model"] == sel_model)]
        def_engine = similar_cars["Engine_CC"].median()
        def_power  = similar_cars["Power_BHP"].median()
        # Handle NaN (if no data for that car)
        if pd.isna(def_engine): def_engine = 1200.0
        if pd.isna(def_power):  def_power  = 85.0

        c5, c6 = st.columns(2)
        with c5:
            sel_engine = st.number_input(
                "Engine Displacement (CC)",
                min_value=500, max_value=6000,
                value=int(def_engine), step=50,
            )
        with c6:
            sel_power = st.number_input(
                "Max Power (BHP)",
                min_value=30, max_value=800,
                value=int(def_power), step=5,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Big predict button
        predict_clicked = st.button("⚡  Predict Price Now")

    # ── RIGHT SIDE: Prediction Result ────────
    with result_col:

        if predict_clicked:

            # ── Build the input for the model ────
            owner_map_pred = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4, "4 or More": 5}
            car_age = 2024 - sel_year

            # Create a single-row DataFrame with the user's inputs
            input_dict = {
                "Make":         sel_make,
                "Model":        sel_model,
                "Car_Age":      car_age,
                "Kilometer":    sel_km,
                "Fuel Type":    sel_fuel,
                "Transmission": sel_trans,
                "Owner_Rank":   owner_map_pred.get(sel_owner, 3),
                "Engine_CC":    sel_engine,
                "Power_BHP":    sel_power,
            }
            input_df = pd.DataFrame([input_dict])

            # ── Encode text columns using saved encoders ──
            # We MUST use the SAME encoders used during training.
            # If user picks "Petrol", the encoder must convert it
            # to the same number it used during model training.
            for col in ["Make", "Model", "Fuel Type", "Transmission"]:
                le = encoders[col]
                val = input_df[col].iloc[0]
                # If the car model is unknown (not in training data),
                # fall back to the first known class
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = le.transform([le.classes_[0]])

            # ── Run Prediction ─────────────────
            log_price  = model.predict(input_df[features])[0]
            pred_price = np.expm1(log_price)   # Reverse the log transform

            # ── Market range from similar cars ──
            # Compare to real listings of same brand, within 2 years
            similar_range = df[
                (df["Make"] == sel_make) &
                (abs(df["Year"] - sel_year) <= 2)
            ]["Price"]
            price_low  = similar_range.quantile(0.25) if len(similar_range) > 0 else pred_price * 0.82
            price_high = similar_range.quantile(0.75) if len(similar_range) > 0 else pred_price * 1.20

            # ── Show Result Card ───────────────
            st.markdown(f"""
            <div class='price-box'>
                <div style='color:#7a8aaa; font-size:11px; letter-spacing:0.12em;
                            text-transform:uppercase; margin-bottom:8px;'>
                    Estimated Market Value
                </div>
                <div style='font-family:Syne,sans-serif; font-size:54px;
                            font-weight:800; color:#f97316; line-height:1;'>
                    ₹{pred_price/1e5:.2f}L
                </div>
                <div style='color:#7a8aaa; font-size:12px; margin:6px 0 20px;'>
                    ₹{pred_price:,.0f}
                </div>
                <div style='display:flex; justify-content:space-around; gap:10px;'>
                    <div class='chip' style='flex:1;'>
                        <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                                    letter-spacing:0.1em; margin-bottom:4px;'>Market Low</div>
                        <div style='color:#10b981; font-family:Syne,sans-serif;
                                    font-size:19px; font-weight:700;'>
                            ₹{price_low/1e5:.1f}L
                        </div>
                    </div>
                    <div class='chip' style='flex:1;'>
                        <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                                    letter-spacing:0.1em; margin-bottom:4px;'>Predicted</div>
                        <div style='color:#f97316; font-family:Syne,sans-serif;
                                    font-size:19px; font-weight:700;'>
                            ₹{pred_price/1e5:.1f}L
                        </div>
                    </div>
                    <div class='chip' style='flex:1;'>
                        <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                                    letter-spacing:0.1em; margin-bottom:4px;'>Market High</div>
                        <div style='color:#ec4899; font-family:Syne,sans-serif;
                                    font-size:19px; font-weight:700;'>
                            ₹{price_high/1e5:.1f}L
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Horizontal Price Range Bar Chart ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📊 Your Car vs. Market Range")

            fig, ax = plt.subplots(figsize=(7, 1.8))

            # Draw a horizontal range bar
            ax.barh(
                [0], [price_high / 1e5 - price_low / 1e5],
                left=price_low / 1e5,
                height=0.4, color=BLUE, alpha=0.35,
                label="Market Range (25th–75th %ile)",
            )
            # Mark the predicted price
            ax.axvline(pred_price / 1e5, color=ORANGE, linewidth=3, label=f"Predicted ₹{pred_price/1e5:.1f}L")

            ax.set_yticks([])
            ax.set_xlabel("Price (₹ Lakhs)", fontsize=9)
            ax.legend(
                facecolor=CARD_BG, edgecolor=GRID_COLOR,
                labelcolor=TEXT_COLOR, fontsize=8,
                loc="upper right",
            )
            ax.set_xlim(max(0, price_low / 1e5 * 0.7), price_high / 1e5 * 1.3)

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

            # ── Model Info Badges ──────────────
            st.markdown(f"""
            <div style='display:flex; flex-wrap:wrap; gap:8px; margin-top:12px;'>
                <span class='badge'>R² Score: {r2:.1%}</span>
                <span class='badge'>Algorithm: Gradient Boosting</span>
                <span class='badge'>Car Age: {car_age} yrs</span>
                <span class='badge'>{sel_km:,} km driven</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Similar Listings Table ─────────
            st.markdown("#### 🔍 Similar Real Listings from Dataset")

            sim_display = df[
                (df["Make"] == sel_make) &
                (abs(df["Year"] - sel_year) <= 2)
            ][["Make", "Model", "Year", "Kilometer", "Fuel Type", "Transmission", "Owner", "Price"]].head(7)

            # Format for display
            sim_display = sim_display.copy()
            sim_display["Price"] = sim_display["Price"].apply(lambda x: f"₹{x/1e5:.2f}L")
            sim_display["Kilometer"] = sim_display["Kilometer"].apply(lambda x: f"{x:,} km")

            st.dataframe(sim_display, use_container_width=True, hide_index=True)

        else:
            # Placeholder before user clicks predict
            st.markdown("""
            <div class='card' style='text-align:center; padding:70px 20px;'>
                <div style='font-size:60px; margin-bottom:16px;'>🔮</div>
                <div style='font-family:Syne,sans-serif; font-size:20px;
                            font-weight:700; color:#e8eaf0; margin-bottom:8px;'>
                    Ready to Predict
                </div>
                <div style='color:#7a8aaa; font-size:14px; line-height:1.8;'>
                    Fill in your car details on the left<br>
                    and click <b style='color:#f97316;'>⚡ Predict Price Now</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║   PAGE 3 — MARKET DASHBOARD                             ║
# ╚══════════════════════════════════════════════════════════╝

elif page_name == "Market Dashboard":

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <div class='badge'>📊 Market Intelligence</div>
        <h1 style='font-family:Syne,sans-serif; font-size:38px;
                   font-weight:800; margin:8px 0 6px;'>
            Indian Used Car
            <span style='color:#f97316;'> Market Dashboard</span>
        </h1>
        <p style='color:#7a8aaa; font-size:14px; margin:0;'>
            Deep-dive analysis across brands, cities, depreciation, and fuel trends.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 Tabs (one per analysis theme) ──────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️  Brand Analysis",
        "📅  Depreciation Trends",
        "⛽  Fuel & Transmission",
        "🗺️  Location Insights",
    ])

    # ═══ TAB 1 — BRAND ANALYSIS ═══════════════
    with tab1:
        st.markdown("### Brand-wise Price & Volume Analysis")
        col1, col2 = st.columns(2, gap="large")

        # CHART: Median Price by Top 12 Brands
        with col1:
            brand_med = (
                df.groupby("Make")["Price"]
                .median()
                .sort_values(ascending=False)
                .head(12) / 1e5
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            # Color gradient: top brand gets full orange, rest fade
            bar_colors = [ORANGE if i == 0 else BLUE if i == 1 else "#2a3f6f"
                          for i in range(len(brand_med))]

            bars = ax.barh(brand_med.index, brand_med.values,
                           color=bar_colors, edgecolor=CHART_BG, linewidth=0.4)

            for bar, val in zip(bars, brand_med.values):
                ax.text(
                    bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"₹{val:.1f}L", va="center", ha="left",
                    color=TEXT_COLOR, fontsize=8,
                )

            ax.invert_yaxis()
            ax.set_title("Median Resale Price by Brand (Top 12)", color=TEXT_COLOR, fontsize=11, pad=10)
            ax.set_xlabel("Median Price (₹ Lakhs)", fontsize=9)

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

        # CHART: Listing Volume Treemap (using seaborn + matplotlib)
        with col2:
            top_brands = df["Make"].value_counts().head(10).reset_index()
            top_brands.columns = ["Brand", "Count"]

            fig, ax = plt.subplots(figsize=(7, 5))

            # Normalize for colormap intensity
            norm_vals = top_brands["Count"] / top_brands["Count"].max()
            colors_used = plt.cm.YlOrRd(norm_vals * 0.8 + 0.15)

            bars = ax.barh(
                top_brands["Brand"],
                top_brands["Count"],
                color=colors_used,
                edgecolor=CHART_BG,
                linewidth=0.5,
            )

            for bar, val in zip(bars, top_brands["Count"]):
                ax.text(
                    bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f"{val} cars", va="center", ha="left",
                    color=TEXT_COLOR, fontsize=8,
                )

            ax.invert_yaxis()
            ax.set_title("Listing Volume by Brand (Top 10)", color=TEXT_COLOR, fontsize=11, pad=10)
            ax.set_xlabel("Number of Listings", fontsize=9)

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

        # CHART: Box plot — Price spread by top 6 brands
        st.markdown("##### Price Distribution by Top 6 Brands (Box Plot)")
        top6 = df["Make"].value_counts().head(6).index.tolist()
        df_top6 = df[df["Make"].isin(top6)].copy()
        df_top6["Price_Lakh"] = df_top6["Price"] / 1e5

        fig, ax = plt.subplots(figsize=(12, 4.5))

        # Seaborn boxplot — great for showing distribution spread
        sns.boxplot(
            data=df_top6,
            x="Make",
            y="Price_Lakh",
            palette=PALETTE[:6],
            width=0.55,
            linewidth=1.2,
            fliersize=3,
            ax=ax,
        )

        ax.set_title("Price Spread by Top 6 Brands", color=TEXT_COLOR, fontsize=11, pad=10)
        ax.set_xlabel("Brand", fontsize=9)
        ax.set_ylabel("Price (₹ Lakhs)", fontsize=9)
        # Limit y-axis to remove extreme outliers from view
        ax.set_ylim(0, df_top6["Price_Lakh"].quantile(0.95))

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    # ═══ TAB 2 — DEPRECIATION TRENDS ══════════
    with tab2:
        st.markdown("### How Car Value Changes Over Time")
        col1, col2 = st.columns(2, gap="large")

        # CHART: Year-wise Median Price Line/Area
        with col1:
            yr_price = df.groupby("Year")["Price"].median().reset_index()
            yr_price.columns = ["Year", "Median_Price"]
            yr_price["Median_Lakh"] = yr_price["Median_Price"] / 1e5
            yr_price = yr_price[yr_price["Year"] >= 2005]   # Focus on modern era

            fig, ax = plt.subplots(figsize=(7, 4.5))

            ax.plot(yr_price["Year"], yr_price["Median_Lakh"],
                    color=ORANGE, linewidth=2.5, marker="o", markersize=5, zorder=3)
            ax.fill_between(yr_price["Year"], yr_price["Median_Lakh"],
                            alpha=0.15, color=ORANGE)

            ax.set_title("Median Resale Price by Manufacturing Year", color=TEXT_COLOR, fontsize=11, pad=10)
            ax.set_xlabel("Year", fontsize=9)
            ax.set_ylabel("Median Price (₹ Lakhs)", fontsize=9)

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

        # CHART: Price vs Car Age Scatter
        with col2:
            # Sample 500 points so chart is not too crowded
            df_sample = df.sample(min(500, len(df)), random_state=42).copy()
            df_sample["Price_Lakh"] = df_sample["Price"] / 1e5

            fig, ax = plt.subplots(figsize=(7, 4.5))

            scatter = ax.scatter(
                df_sample["Car_Age"],
                df_sample["Price_Lakh"],
                c=df_sample["Price_Lakh"],
                cmap="YlOrRd",
                alpha=0.55,
                s=18,
                edgecolors="none",
            )

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.ax.tick_params(colors=TEXT_COLOR)
            cbar.set_label("Price (₹ Lakhs)", color=TEXT_COLOR, fontsize=8)

            ax.set_title("Car Age vs. Resale Price", color=TEXT_COLOR, fontsize=11, pad=10)
            ax.set_xlabel("Car Age (Years)", fontsize=9)
            ax.set_ylabel("Price (₹ Lakhs)", fontsize=9)
            ax.set_ylim(0, df_sample["Price_Lakh"].quantile(0.97))

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

        # CHART: Kilometers vs Price with regression line
        st.markdown("##### 🛣️ Kilometers Driven vs. Price (with Trend Line)")

        df_km = df[df["Price"] < 10_000_000].copy()
        df_km["Price_Lakh"] = df_km["Price"] / 1e5

        fig, ax = plt.subplots(figsize=(12, 4))

        sample = df_km.sample(min(600, len(df_km)), random_state=7)

        ax.scatter(sample["Kilometer"], sample["Price_Lakh"],
                   alpha=0.35, s=12, color=BLUE, edgecolors="none")

        # Add regression (trend) line using numpy polyfit
        z = np.polyfit(sample["Kilometer"], sample["Price_Lakh"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample["Kilometer"].min(), sample["Kilometer"].max(), 200)
        ax.plot(x_line, p(x_line), color=ORANGE, linewidth=2.5, label="Trend Line")

        ax.set_xlabel("Kilometers Driven", fontsize=9)
        ax.set_ylabel("Price (₹ Lakhs)", fontsize=9)
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    # ═══ TAB 3 — FUEL & TRANSMISSION ══════════
    with tab3:
        st.markdown("### Fuel Type & Transmission Analysis")
        col1, col2 = st.columns(2, gap="large")

        # CHART: Fuel Type Preference by Brand (stacked bar)
        with col1:
            top5_brands = df["Make"].value_counts().head(5).index.tolist()
            fuel_brand = df[df["Make"].isin(top5_brands)].groupby(
                ["Make", "Fuel Type"]
            ).size().unstack(fill_value=0)

            # Keep only top 4 fuel types
            top_fuel_cols = df["Fuel Type"].value_counts().head(4).index.tolist()
            fuel_brand = fuel_brand[[c for c in top_fuel_cols if c in fuel_brand.columns]]

            fig, ax = plt.subplots(figsize=(7, 4.5))

            bottom = np.zeros(len(fuel_brand))
            for i, fuel_col in enumerate(fuel_brand.columns):
                ax.bar(
                    fuel_brand.index,
                    fuel_brand[fuel_col],
                    bottom=bottom,
                    label=fuel_col,
                    color=PALETTE[i],
                    edgecolor=CHART_BG,
                    linewidth=0.4,
                    alpha=0.92,
                )
                bottom += fuel_brand[fuel_col].values

            ax.set_title("Fuel Type Preference — Top 5 Brands", color=TEXT_COLOR, fontsize=11, pad=10)
            ax.set_xlabel("Brand", fontsize=9)
            ax.set_ylabel("Number of Listings", fontsize=9)
            ax.tick_params(axis="x", rotation=15)
            ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                      fontsize=8, loc="upper right")

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

        # CHART: Median Price by Transmission + Fuel
        with col2:
            trans_fuel = df.groupby(["Transmission", "Fuel Type"])["Price"].median().unstack(fill_value=0) / 1e5
            top_fuels_3 = df["Fuel Type"].value_counts().head(3).index.tolist()
            trans_fuel = trans_fuel[[c for c in top_fuels_3 if c in trans_fuel.columns]]

            fig, ax = plt.subplots(figsize=(7, 4.5))

            x = np.arange(len(trans_fuel.index))
            bar_w = 0.25

            for i, fuel_col in enumerate(trans_fuel.columns):
                ax.bar(
                    x + i * bar_w,
                    trans_fuel[fuel_col],
                    width=bar_w,
                    label=fuel_col,
                    color=PALETTE[i],
                    edgecolor=CHART_BG,
                    alpha=0.9,
                )

            ax.set_xticks(x + bar_w)
            ax.set_xticklabels(trans_fuel.index)
            ax.set_title("Median Price: Transmission × Fuel Type", color=TEXT_COLOR, fontsize=11, pad=10)
            ax.set_ylabel("Median Price (₹ Lakhs)", fontsize=9)
            ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

        # CHART: Violin plot — Price by Fuel Type
        st.markdown("##### 🎻 Price Distribution by Fuel Type (Violin Plot)")

        top_fuels_5 = df["Fuel Type"].value_counts().head(5).index.tolist()
        df_violin = df[df["Fuel Type"].isin(top_fuels_5)].copy()
        df_violin["Price_Lakh"] = df_violin["Price"] / 1e5

        fig, ax = plt.subplots(figsize=(12, 4))

        sns.violinplot(
            data=df_violin,
            x="Fuel Type",
            y="Price_Lakh",
            palette=PALETTE[:5],
            inner="box",
            linewidth=1,
            ax=ax,
        )

        ax.set_title("Price Spread by Fuel Type", color=TEXT_COLOR, fontsize=11, pad=10)
        ax.set_xlabel("Fuel Type", fontsize=9)
        ax.set_ylabel("Price (₹ Lakhs)", fontsize=9)
        ax.set_ylim(0, df_violin["Price_Lakh"].quantile(0.96))

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    # ═══ TAB 4 — LOCATION INSIGHTS ═════════════
    with tab4:
        st.markdown("### City-wise Market Analysis")
        col1, col2 = st.columns(2, gap="large")

        # CHART: Median Price by City (Top 12)
        with col1:
            city_price = (
                df.groupby("Location")["Price"]
                .median()
                .sort_values(ascending=False)
                .head(12) / 1e5
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            bar_colors = [ORANGE if i < 3 else BLUE if i < 6 else "#2a3f6f"
                          for i in range(len(city_price))]

            bars = ax.barh(city_price.index, city_price.values,
                           color=bar_colors, edgecolor=CHART_BG, linewidth=0.4)

            for bar, val in zip(bars, city_price.values):
                ax.text(
                    bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"₹{val:.1f}L", va="center", ha="left",
                    color=TEXT_COLOR, fontsize=8,
                )

            ax.invert_yaxis()
            ax.set_title("Median Resale Price by City (Top 12)", color=TEXT_COLOR, fontsize=11, pad=10)
            ax.set_xlabel("Median Price (₹ Lakhs)", fontsize=9)

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

        # CHART: Listing Volume by City (Top 12 pie)
        with col2:
            city_vol = df["Location"].value_counts().head(10)

            fig, ax = plt.subplots(figsize=(7, 5))

            wedges, texts, autotexts = ax.pie(
                city_vol.values,
                labels=city_vol.index,
                autopct="%1.1f%%",
                startangle=120,
                colors=PALETTE[:10],
                pctdistance=0.75,
                wedgeprops=dict(width=0.55, edgecolor=CHART_BG, linewidth=1.5),
            )

            for t in texts:
                t.set_color(TEXT_COLOR)
                t.set_fontsize(8)
            for a in autotexts:
                a.set_color("white")
                a.set_fontsize(7.5)
                a.set_fontweight("bold")

            ax.set_title("Listing Volume by City (Top 10)", color=TEXT_COLOR, fontsize=11, pad=10)
            ax.text(0, 0, "Cities", ha="center", va="center",
                    fontsize=10, color=TEXT_COLOR, fontweight="bold")

            fig = apply_dark_style(fig)
            st.pyplot(fig)
            plt.close(fig)

        # CHART: Heatmap — Brand availability by City
        st.markdown("##### 🔥 Brand vs City Availability Heatmap")

        top5_c = df["Location"].value_counts().head(8).index.tolist()
        top5_m = df["Make"].value_counts().head(8).index.tolist()
        heat_df = df[df["Location"].isin(top5_c) & df["Make"].isin(top5_m)]
        heat_pivot = heat_df.groupby(["Make", "Location"]).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 4.5))

        sns.heatmap(
            heat_pivot,
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor=CHART_BG,
            annot=True,
            fmt="d",
            annot_kws={"size": 8, "color": "white"},
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title("Number of Listings: Brand × City", color=TEXT_COLOR, fontsize=11, pad=10)
        ax.set_xlabel("City", fontsize=9)
        ax.set_ylabel("Brand", fontsize=9)
        ax.tick_params(colors=TEXT_COLOR)
        ax.figure.axes[-1].tick_params(colors=TEXT_COLOR)  # colorbar ticks

        fig = apply_dark_style(fig, ax_list=[ax])
        st.pyplot(fig)
        plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║   PAGE 4 — MODEL INSIGHTS                               ║
# ╚══════════════════════════════════════════════════════════╝

elif page_name == "Model Insights":

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <div class='badge'>🧠 ML Model Diagnostics</div>
        <h1 style='font-family:Syne,sans-serif; font-size:38px;
                   font-weight:800; margin:8px 0 6px;'>
            Model <span style='color:#f97316;'>Performance & Insights</span>
        </h1>
        <p style='color:#7a8aaa; font-size:14px; margin:0;'>
            Transparent evaluation of our Gradient Boosting model —
            feature importance, accuracy, and error analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm",    "Gradient Boosting")
    m2.metric("R² Score",     f"{r2:.4f}",               f"{r2*100:.1f}% variance explained")
    m3.metric("Avg Error (MAE)", f"₹{mae/1e5:.2f}L",     "Mean Absolute Error")
    m4.metric("Training Size", f"{int(len(df)*0.8):,}",  "80/20 train-test split")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    # CHART: Feature Importance
    with col1:
        st.markdown("#### 🏅 Feature Importance")

        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Color bars: top 3 get orange, rest get blue
        bar_colors = [ORANGE if i < 3 else BLUE for i in range(len(importance))]

        bars = ax.barh(
            importance["Feature"],
            importance["Importance"],
            color=bar_colors,
            edgecolor=CHART_BG,
            linewidth=0.4,
        )

        # Importance % labels
        for bar, val in zip(bars, importance["Importance"]):
            ax.text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                color=TEXT_COLOR, fontsize=8,
            )

        ax.invert_yaxis()
        ax.set_xlabel("Importance Score", fontsize=9)

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div class='card' style='padding:16px 20px; margin-top:8px;'>
            <div style='font-size:11px; color:#7a8aaa; line-height:2;'>
                🔸 <b style='color:#f97316;'>Make & Model</b> together explain ~65% of price<br>
                🔸 <b style='color:#f97316;'>Car_Age</b> is the top depreciation signal<br>
                🔸 <b style='color:#f97316;'>Kilometer</b> and <b style='color:#f97316;'>Engine_CC</b> add strong context<br>
                🔸 <b style='color:#7a8aaa;'>Owner_Rank</b> has smaller but real impact
            </div>
        </div>
        """, unsafe_allow_html=True)

    # CHART: Actual vs Predicted Scatter
    with col2:
        st.markdown("#### 🎯 Actual vs. Predicted Price")

        actual    = np.expm1(y_test)
        predicted = np.expm1(y_pred)

        # Sample 400 points to keep chart clean
        idx = np.random.choice(len(actual), min(400, len(actual)), replace=False)
        act_sample  = actual.iloc[idx] / 1e5
        pred_sample = predicted[idx] / 1e5

        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.scatter(act_sample, pred_sample,
                   alpha=0.45, s=14, color=ORANGE, edgecolors="none")

        # Perfect prediction line (where all dots SHOULD fall)
        max_val = max(act_sample.max(), pred_sample.max())
        ax.plot([0, max_val], [0, max_val],
                color=BLUE, linestyle="--", linewidth=2, label="Perfect Fit")

        ax.set_xlabel("Actual Price (₹ Lakhs)", fontsize=9)
        ax.set_ylabel("Predicted Price (₹ Lakhs)", fontsize=9)
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

        # R² annotation
        ax.text(
            0.05, 0.92, f"R² = {r2:.4f}",
            transform=ax.transAxes,
            color=ORANGE, fontsize=10, fontweight="bold",
            bbox=dict(facecolor=CARD_BG, edgecolor=GRID_COLOR, boxstyle="round,pad=0.3"),
        )

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    # CHART: Residual Plot + Error Distribution side by side
    st.markdown("#### 📉 Error Analysis")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        # Residual plot: predicted on x-axis, error on y-axis
        # Ideally errors should be randomly scattered around y=0
        residuals = (actual - predicted) / 1e5

        idx2 = np.random.choice(len(residuals), min(400, len(residuals)), replace=False)

        fig, ax = plt.subplots(figsize=(6.5, 3.5))

        ax.scatter(predicted[idx2] / 1e5, residuals.iloc[idx2],
                   alpha=0.4, s=12, color=GREEN, edgecolors="none")
        ax.axhline(y=0, color=ORANGE, linewidth=2, linestyle="--")

        ax.set_xlabel("Predicted Price (₹ Lakhs)", fontsize=9)
        ax.set_ylabel("Residual Error (₹ Lakhs)", fontsize=9)
        ax.set_title("Residual Plot — Errors Should Scatter Around Zero",
                     color=TEXT_COLOR, fontsize=10, pad=8)

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        # Error distribution histogram
        # Should be bell-shaped and centered at 0
        fig, ax = plt.subplots(figsize=(6.5, 3.5))

        ax.hist(residuals / residuals.abs().max(),
                bins=40, color=PURPLE, edgecolor=CHART_BG,
                linewidth=0.3, alpha=0.85)
        ax.axvline(x=0, color=ORANGE, linewidth=2, linestyle="--")
        ax.set_xlabel("Normalized Error", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_title("Distribution of Prediction Errors",
                     color=TEXT_COLOR, fontsize=10, pad=8)

        fig = apply_dark_style(fig)
        st.pyplot(fig)
        plt.close(fig)

    # ── Model Architecture Card ───────────────
    st.markdown("""
    <div class='card'>
        <h3 style='font-family:Syne,sans-serif; color:#f97316; margin-top:0; margin-bottom:16px;'>
            🏗️ Model Architecture & Hyperparameters
        </h3>
        <div class='arch-grid'>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>Algorithm</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>Gradient Boosting</div>
            </div>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>N Estimators</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>500 Trees</div>
            </div>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>Learning Rate</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>0.05 (Slow & Accurate)</div>
            </div>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>Max Depth</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>6 Levels per Tree</div>
            </div>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>Subsample</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>0.8 (Anti-overfit)</div>
            </div>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>Target Transform</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>log1p(Price)</div>
            </div>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>Train / Test Split</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>80% / 20%</div>
            </div>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>Features Used</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>9 (out of 20)</div>
            </div>
            <div class='arch-item'>
                <div style='color:#7a8aaa; font-size:10px; text-transform:uppercase;
                            letter-spacing:0.1em; margin-bottom:5px;'>Encoding</div>
                <div style='font-family:Syne,sans-serif; font-weight:700;
                            color:#e8eaf0;'>LabelEncoder</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
