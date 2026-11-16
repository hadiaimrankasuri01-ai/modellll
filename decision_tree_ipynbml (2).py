# -*- coding: utf-8 -*-
"""Starbucks Nutrition & Decision Tree Classifier - Repository Ready App"""

import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Starbucks Nutrition Classifier", layout="wide")
st.title("Starbucks Nutrition & Decision Tree Classifier")

# Built-in fallback dataset so the app always runs even if CSV is missing from the repo
SAMPLE_DATA = """Drink_Name,Calories,Fat (g),Carb. (g),Fiber (g),Protein,Sodium
Cool Lime Starbucks Refreshers Beverage,45,0,11,0,0,10
Strawberry Acai Starbucks Refreshers Beverage,80,0,18,1,0,10
Very Berry Hibiscus Starbucks Refreshers Beverage,60,0,14,1,0,10
Evolution Fresh Organic Ginger Limeade,110,0,28,0,0,5
Iced Coffee,0,0,0,0,0,0
Iced Coffee with Milk,80,2.5,11,0,2,35
Iced Espresso Classics - Vanilla Latte,130,2.5,21,0,5,65
Iced Espresso Classics - Caffe Mocha,140,2.5,23,0,5,90
Iced Espresso Classics - Caramel Macchiato,130,2.5,21,0,5,65
Starbucks Doubleshot on Ice Beverage,45,1,5,0,3,40
Coffee Frappuccino Blended Coffee,230,3,45,0,3,190
Mocha Frappuccino Blended Coffee,280,2.5,60,2,4,220
Caramel Frappuccino Blended Coffee,280,3.5,59,0,3,220
Java Chip Frappuccino Blended Coffee,340,7,64,2,5,260
Green Tea Creme Frappuccino Blended Creme,350,8,64,1,6,240
White Chocolate Mocha,360,11,53,0,11,240
Caffe Latte,190,7,19,0,13,170
Caffe Mocha,290,8,42,2,10,140
Caramel Macchiato,250,7,35,0,10,150
Cappuccino,120,4,12,0,8,100
Hot Chocolate,320,9,47,1,11,160
Chai Tea Latte,240,4.5,45,0,8,115
Matcha Green Tea Latte,240,7,34,1,12,160
Pink Drink,140,2.5,27,1,1,65
Dragon Drink,130,3,26,1,1,60
Nitro Cold Brew,5,0,0,0,0,10
Vanilla Sweet Cream Cold Brew,110,5,14,0,1,25
Caramel Apple Spice,380,0,89,0,1,20
Steamed Apple Juice,220,0,53,0,1,15
Iced Passion Tango Tea,0,0,0,0,0,10
"""

# --- Sidebar Controls (Reset & Hyperparameters) ---
with st.sidebar:
    st.header("Controls")
    if st.button("Reset App & Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    max_depth = st.slider("Decision Tree Max Depth", min_value=1, max_value=15, value=5)
    test_size = st.slider("Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

# --- 1. Load Dataset & Exploratory Data Analysis (EDA) ---
st.header("1. Dataset Overview & EDA")

uploaded_file = st.file_uploader(
    "Upload 'starbucks-menu-nutrition-drinks.csv' (Optional — automatically uses repo CSV or built-in sample)",
    type=["csv"]
)
default_path = "starbucks-menu-nutrition-drinks.csv"

@st.cache_data
def load_csv_bytes(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    if df.columns[0].startswith("Unnamed"):
        df.rename(columns={df.columns[0]: "Drink_Name"}, inplace=True)
    return df

@st.cache_data
def load_csv_path(path):
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df.rename(columns={df.columns[0]: "Drink_Name"}, inplace=True)
    return df

if uploaded_file is not None:
    df = load_csv_bytes(uploaded_file.getvalue())
    st.info("Loaded uploaded CSV dataset.")
else:
    try:
        df = load_csv_path(default_path)
    except Exception:
        df = pd.read_csv(io.StringIO(SAMPLE_DATA))
        st.warning("Repo CSV not found — running with built-in sample dataset. Upload your CSV above anytime.")

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.subheader("First 5 Rows")
    st.dataframe(df.head(), use_container_width=True)
with col2:
    st.subheader("Missing Values per Column")
    st.dataframe(df.isnull().sum().rename("Missing Count"), use_container_width=True)

# --- 2. Data Cleaning & Drink Categories Distribution ---
st.header("2. Drink Categories Distribution")

numeric_cols = ["Calories", "Fat (g)", "Carb. (g)", "Fiber (g)", "Protein", "Sodium"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace("-", "", regex=False).str.strip(),
            errors="coerce"
        )

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

def categorize_calories(cal):
    if cal < 150:
        return "Low Calorie (<150)"
    elif cal <= 300:
        return "Medium Calorie (150-300)"
    else:
        return "High Calorie (>300)"

df["Calorie_Category"] = df["Calories"].apply(categorize_calories)

category_counts = df["Calorie_Category"].value_counts()
st.subheader("Distribution of Starbucks Drinks by Calorie Level")
st.bar_chart(category_counts)

summary_table = category_counts.reset_index()
summary_table.columns = ["Calorie Category", "Drink Count"]
st.dataframe(summary_table, use_container_width=True)

# --- 3. Split Data & Train Decision Tree Classifier ---
st.header("3. Train Decision Tree Classifier")

features = ["Fat (g)", "Carb. (g)", "Fiber (g)", "Protein", "Sodium"]
X = df[features]
y = df["Calorie_Category"]

min_class_count = y.value_counts().min()
stratify_param = y if min_class_count >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=stratify_param
)

clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
clf.fit(X_train, y_train)

st.success("Decision Tree model trained successfully!")

# --- 4. Evaluate Model Performance ---
st.header("4. Model Performance Evaluation")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.metric(label="Test Accuracy Score", value=f"{accuracy:.2%}")

st.subheader("Classification Report")
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)

# --- 5. Compare Actual vs. Predicted Labels ---
st.header("5. Actual vs. Predicted Labels (Test Samples)")

sample_count = min(10, len(X_test))
test_indices = X_test.head(sample_count).index
drink_col = "Drink_Name" if "Drink_Name" in df.columns else df.columns[0]
test_drinks = df.loc[test_indices, drink_col].values
actual_samples = y_test.head(sample_count).values
predicted_samples = y_pred[:sample_count]

comparison_df = pd.DataFrame({
    "Drink Name": test_drinks,
    "Actual Category": actual_samples,
    "Predicted Category": predicted_samples,
    "Match": actual_samples == predicted_samples
})

st.dataframe(comparison_df, use_container_width=True)
