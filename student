import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Page Setup
st.set_page_config(page_title="Student Performance Analyzer", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Analysis & Prediction App")
st.write("Analyze student habits and predict final grades (`G3`) using Machine Learning!")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("Student.csv")

df = load_data()

# Navigation Sidebar
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Select View:", ["📊 Exploratory Data Analysis", "🔮 Machine Learning Predictor"])

if menu == "📊 Exploratory Data Analysis":
    st.header("1. Dataset Overview & Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df[['studytime', 'absences', 'failures', 'G1', 'G2', 'G3']].describe(), use_container_width=True)
        
    st.markdown("---")
    st.subheader("Visual Analysis")
    
    col3, col4 = st.columns(2)
    with col3:
        st.write("### Final Grade Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['G3'], kde=True, color='purple', bins=15, ax=ax)
        ax.set_title("Distribution of Final Grades (G3)")
        st.pyplot(fig)
        
    with col4:
        st.write("### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[['studytime', 'absences', 'failures', 'G1', 'G2', 'G3']].corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

elif menu == "🔮 Machine Learning Predictor":
    st.header("2. Predict Student Final Grade (G3)")
    st.write("Adjust the sliders below to predict a student's final score out of 20!")
    
    # Train Random Forest Regressor Model
    X = df[['studytime', 'absences', 'failures', 'G1', 'G2']]
    y = df['G3']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    r2 = r2_score(y_test, model.predict(X_test))
    st.info(f"💡 **Model Accuracy (R² Score):** {r2*100:.1f}%")
    
    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        studytime = st.slider("Weekly Study Time (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)", 1, 4, 2)
        absences = st.slider("Number of Absences", 0, 32, 2)
        failures = st.slider("Past Class Failures", 0, 4, 0)
    with col2:
        g1 = st.slider("First Period Grade (G1) [0 - 20]", 0, 20, 12)
        g2 = st.slider("Second Period Grade (G2) [0 - 20]", 0, 20, 12)
        
    if st.button("🚀 Predict Final Grade"):
        prediction = model.predict([[studytime, absences, failures, g1, g2]])[0]
        st.success(f"🎯 **Predicted Final Score (G3):** {prediction:.2f} / 20")

