import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1a2748 0%, #0d1322 35%, #070c16 100%);
        color: #edf2ff;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .dashboard-shell {
        margin-bottom: 1.25rem;
    }

    .hero-panel {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        background: linear-gradient(120deg, rgba(17, 24, 39, 0.98), rgba(30, 41, 59, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 24px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.35);
    }

    .hero-copy {
        flex: 1;
    }

    .eyebrow {
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
    }

    .hero-panel h1 {
        margin: 0;
        font-size: clamp(2.1rem, 3vw, 3rem);
        color: #f8fbff;
        line-height: 1.1;
    }

    .hero-panel p {
        margin: 0.65rem 0 0 0;
        color: rgba(226, 232, 240, 0.82);
        font-size: 1rem;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.7rem 1rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #bbf7d0;
        font-weight: 700;
        font-size: 0.78rem;
        white-space: nowrap;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .stat-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.3);
    }

    .stat-card span {
        display: block;
        color: #93c5fd;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }

    .stat-card strong {
        color: #f8fbff;
        font-size: 1.1rem;
    }

    .panel {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 22px;
        padding: 1.25rem 1.3rem;
        box-shadow: 0 14px 30px rgba(2, 6, 23, 0.3);
    }

    .panel h3 {
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
        color: #f8fbff;
    }

    .panel .stNumberInput {
        margin-bottom: 0.35rem;
    }

    .info-card {
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }

    .info-card h4 {
        margin: 0 0 0.8rem 0;
        color: #e2e8f0;
        font-size: 1rem;
    }

    .info-card p {
        margin: 0.38rem 0;
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .species-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .species-list li {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        color: #e2e8f0;
        font-weight: 500;
    }

    .species-list li:last-child {
        border-bottom: none;
    }

    .result-card {
        margin-top: 1.2rem;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        border-radius: 22px;
        padding: 1.3rem 1.2rem;
        color: white;
        box-shadow: 0 18px 35px rgba(99, 102, 241, 0.3);
    }

    .result-card .label {
        display: inline-block;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.86;
        margin-bottom: 0.5rem;
    }

    .result-card h2 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.1;
    }

    .result-card p {
        margin: 0.65rem 0 0 0;
        font-size: 0.96rem;
        opacity: 0.97;
    }

    .probability-box {
        margin-top: 1rem;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 1rem;
    }

    .probability-row {
        margin-bottom: 0.7rem;
    }

    .probability-row:last-child {
        margin-bottom: 0;
    }

    .probability-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.35rem;
        color: #e2e8f0;
        font-size: 0.85rem;
    }

    .probability-track {
        width: 100%;
        height: 10px;
        background: rgba(148, 163, 184, 0.18);
        border-radius: 999px;
        overflow: hidden;
    }

    .probability-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%);
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-weight: 700;
        padding: 0.8rem 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 12px 28px rgba(79, 70, 229, 0.26);
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 28px rgba(79, 70, 229, 0.34);
    }

    div[data-testid="stButton"] > button:focus {
        box-shadow: 0 0 0 0.25rem rgba(79, 70, 229, 0.24);
    }

    .stNumberInput > div {
        border-radius: 12px;
        border: 1px solid rgba(96, 165, 250, 0.28);
        background: rgba(15, 23, 42, 0.65);
    }

    .stNumberInput label {
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load trained model
model = joblib.load("models/iris_model.pkl")

st.markdown(
    """
    <div class="dashboard-shell">
        <div class="hero-panel">
            <div class="hero-copy">
                <p class="eyebrow">Iris Analytics Suite</p>
                <h1>🌸 Flower Recognition Dashboard</h1>
                <p>Predict iris species from petal and sepal dimensions using a trained logistic regression model.</p>
            </div>
            <div class="status-badge">Live model ready</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model_col, classes_col, inputs_col = st.columns(3)

with model_col:
    st.markdown(
        """
        <div class="stat-card">
            <span>Model</span>
            <strong>Logistic Regression</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

with classes_col:
    st.markdown(
        """
        <div class="stat-card">
            <span>Classes</span>
            <strong>3 Species</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

with inputs_col:
    st.markdown(
        """
        <div class="stat-card">
            <span>Inputs</span>
            <strong>4 Features</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

left_col, right_col = st.columns([1.75, 1.1])

with left_col:
    st.markdown('<div class="panel"><h3>Measurement Input</h3>', unsafe_allow_html=True)

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1,
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1,
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
    )

    if st.button("🔍 Predict Species"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        flower_names = ["Setosa", "Versicolor", "Virginica"]
        predicted_flower = flower_names[prediction[0]]
        confidence = probabilities[0][prediction[0]] * 100

        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Prediction</div>
                <h2>{predicted_flower}</h2>
                <p>Confidence: <strong>{confidence:.2f}%</strong></p>
            </div>
            <div class="probability-box">
                {''.join(
                    f'''<div class="probability-row">
                        <div class="probability-head">
                            <span>{name}</span>
                            <span>{probabilities[0][idx] * 100:.1f}%</span>
                        </div>
                        <div class="probability-track">
                            <div class="probability-fill" style="width: {probabilities[0][idx] * 100:.1f}%"></div>
                        </div>
                    </div>'''
                    for idx, name in enumerate(flower_names)
                )}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown(
        """
        <div class="panel">
            <h3>Model Overview</h3>
            <div class="info-card">
                <h4>About the model</h4>
                <p><strong>Algorithm:</strong> Logistic Regression</p>
                <p><strong>Dataset:</strong> Iris Dataset</p>
                <p><strong>Input Features:</strong> 4</p>
                <p><strong>Output Classes:</strong> 3</p>
            </div>
            <div class="info-card">
                <h4>Species</h4>
                <ul class="species-list">
                    <li>🌼 Setosa</li>
                    <li>🌿 Versicolor</li>
                    <li>🌺 Virginica</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )