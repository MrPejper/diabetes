import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
from langfuse import Langfuse
import uuid
import os
from dotenv import load_dotenv


load_dotenv()


LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")  

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST   
)
# ==== WCZYTANIE MODELU ====
model = load_model("diabetes_model")

# ==== INTERFEJS ====
st.title("🧪 Przewidywanie cukrzycy")
st.write("Wprowadź dane pacjenta, aby przewidzieć ryzyko cukrzycy i sprawdzić zgodność wskaźników z normami.")

with st.form("form"):
    gender = st.selectbox("Płeć", ["Male", "Female"])
    age = st.number_input("Wiek (1–120)", min_value=0, max_value=150, value=30)
    hypertension = st.selectbox("Nadciśnienie", [0, 1], format_func=lambda x: "Tak" if x == 1 else "Nie")
    heart_disease = st.selectbox("Choroba serca", [0, 1], format_func=lambda x: "Tak" if x == 1 else "Nie")
    smoking_history = st.selectbox("Historia palenia", ["No Info", "never", "former", "current", "not current"])
    bmi = st.number_input("BMI (norma: 18.5 – 24.9)", value=22.0)
    hba1c = st.number_input("HbA1c (%) (norma: <5.7)", value=5.5)
    glucose = st.number_input("Poziom glukozy we krwi (mg/dL) (norma: 70 – 99)", value=90.0)

    submitted = st.form_submit_button("Przewiduj")

    if submitted:
        błędne_pola = []
        if not (1 <= age <= 120): błędne_pola.append("wiek (1–120)")
        if not (10 <= bmi <= 60): błędne_pola.append("BMI (10–60)")
        if not (3 <= hba1c <= 15): błędne_pola.append("HbA1c (3–15%)")
        if not (40 <= glucose <= 500): błędne_pola.append("glukoza (40–500 mg/dL)")

        if błędne_pola:
            st.error("🚫 Nieprawidłowe wartości wykryte. Proszę poprawić poniższe pola:")
            for pole in błędne_pola:
                st.warning(f"❗ {pole}")
        else:
            # ==== PRZYGOTOWANIE DANYCH ====
            input_df = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'smoking_history': [smoking_history],
                'bmi': [bmi],
                'HbA1c_level': [hba1c],
                'blood_glucose_level': [glucose]
            })

            # ==== LANGFUSE MONITORING ====
            user_id = str(uuid.uuid4())
            trace = langfuse.trace(name="diabetes-prediction", user_id=user_id)

            try:
                span = trace.span(name="pycaret-model-call", input=input_df.to_dict())

                prediction = predict_model(model, data=input_df)
                result = int(prediction.loc[0, 'prediction_label'])
                score = float(prediction.loc[0, 'prediction_score']) if 'prediction_score' in prediction.columns else None

                span.end(output={"result": result, "score": score})
                trace.score(name="confidence", value=score or 0.0)
                trace.metadata = {"gender": gender, "age": age}

                # ==== OCENA PARAMETRÓW ====
                st.subheader("📋 Ocena parametrów:")

                def check(label, val, low, high):
                    if low <= val <= high:
                        st.success(f"{label}: {val} ✅ (norma: {low} – {high})")
                    else:
                        st.warning(f"{label}: {val} ❌ (norma: {low} – {high})")

                check("BMI", bmi, 18.5, 24.9)
                check("HbA1c (%)", hba1c, 0, 5.6)
                check("Glukoza (mg/dL)", glucose, 70, 99)

                # ==== WYNIK ====
                st.subheader("🔍 Przewidywany wynik:")
                if result == 1:
                    st.error("🔴 Cukrzyca (1)")
                else:
                    st.success("🟢 Brak cukrzycy (0)")

            except Exception as e:
                span.end(output={"error": str(e)})
                st.error(f"Błąd predykcji: {str(e)}")
