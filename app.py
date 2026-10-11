import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Medical AI Demo", page_icon="🩺", layout="wide")

@st.cache_resource
def train_demo_model():
    # Synthetic data only: generated for demonstrating an ML workflow.
    X, y = make_classification(
        n_samples=1200,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=0.9,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=120, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, test_accuracy = train_demo_model()

st.title("🩺 Medical AI Prediction Assistant")
st.caption("One-day educational ML project • Runs locally on your computer")

st.warning(
    "Educational demonstration only. The dataset is synthetic, the labels are invented, "
    "and outputs are NOT medical advice or clinically meaningful risk estimates."
)

with st.sidebar:
    st.header("Sample input")
    st.write("Change the fictional measurements below, then click Predict.")
    feature_1 = st.slider("Measurement A", -3.0, 3.0, 0.2, 0.1)
    feature_2 = st.slider("Measurement B", -3.0, 3.0, -0.4, 0.1)
    feature_3 = st.slider("Measurement C", -3.0, 3.0, 0.8, 0.1)
    feature_4 = st.slider("Measurement D", -3.0, 3.0, 0.1, 0.1)
    feature_5 = st.slider("Measurement E", -3.0, 3.0, -0.2, 0.1)
    predict = st.button("Predict sample category", type="primary", use_container_width=True)

col1, col2 = st.columns([1.25, 1])
with col1:
    st.subheader("How the demo works")
    st.markdown("""
    1. Generate a **synthetic dataset**.
    2. Split it into training and test sets.
    3. Train a Random Forest classifier.
    4. Enter five fictional measurements.
    5. Display the model's predicted class and score.
    """)
    st.info("The inputs are generic demo measurements, not real patient measurements or clinical features.")

with col2:
    st.subheader("Model information")
    st.metric("Held-out demo accuracy", f"{test_accuracy:.1%}")
    st.write("Algorithm: Random Forest")
    st.write("Training examples: 960")
    st.write("Test examples: 240")
    st.write("Data source: synthetically generated at startup")

if predict:
    sample = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    score = float(probabilities[prediction])

    st.divider()
    st.subheader("Prediction result")
    result_col, score_col = st.columns(2)
    with result_col:
        st.metric("Predicted demo class", f"Category {prediction}")
    with score_col:
        st.metric("Model score for predicted class", f"{score:.1%}")
    st.progress(score)
    st.caption(
        "This score reflects the model's output on invented data. It is not a probability "
        "of disease, a diagnosis, or a calibrated clinical confidence estimate."
    )

    st.subheader("Your input values")
    st.dataframe(pd.DataFrame({
        "Feature": ["Measurement A", "Measurement B", "Measurement C", "Measurement D", "Measurement E"],
        "Value": [feature_1, feature_2, feature_3, feature_4, feature_5]
    }), use_container_width=True, hide_index=True)

st.divider()
st.caption("Next upgrade: replace the synthetic demo with a properly evaluated public dataset, then add medical-image classification and explainability.")
