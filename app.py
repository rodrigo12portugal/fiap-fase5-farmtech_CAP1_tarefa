import gradio as gr
import joblib
import pandas as pd

# carrega o pipeline completo (preprocess + modelo)
model = joblib.load("best_model.joblib")

def prever(cultura, precipitacao, umidade_especifica, umidade_relativa, temperatura):
    X = pd.DataFrame([{
        "cultura": cultura,
        "precipitacao_mm_dia": float(precipitacao),
        "umidade_especifica_gkg": float(umidade_especifica),
        "umidade_relativa_pct": float(umidade_relativa),
        "temperatura_c": float(temperatura)
    }])
    y = model.predict(X)[0]
    return round(float(y), 2)

demo = gr.Interface(
    fn=prever,
    inputs=[
        gr.Textbox(label="Cultura (ex.: milho, soja, trigo)"),
        gr.Number(label="Precipitação (mm/dia)"),
        gr.Number(label="Umidade específica (g/kg)"),
        gr.Number(label="Umidade relativa (%)"),
        gr.Number(label="Temperatura a 2 m (°C)")
    ],
    outputs=gr.Number(label="Rendimento previsto (t/ha)"),
    title="FarmTech Solutions - Previsão de Rendimento",
    description="Informe as condições e obtenha a previsão de t/ha."
)

if __name__ == "__main__":
    demo.launch()
