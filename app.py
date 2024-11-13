
import streamlit as st
from fastai.vision.all import*
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    img = PILImage.create(uploaded_file)
    
    
    learner = load_learner("transport_model.pkl")
    
    
    st.write(f"Learner turi: {type(learner)}")


    try:
        
        if isinstance(learner, Learner):
            pred, pred_idx, probs = learner.predict(img)
            st.image(img, caption='Yuklangan rasm', use_column_width=True)
            st.write(f"Bu rasm: {pred} (Ishonch: {probs[pred_idx]:.2f})")
        else:
            st.error("Learner obyektida xato. Model yuklash jarayonini tekshiring.")
    except Exception as e:
        st.error(f"Rasmni aniqlashda xato: {e}")