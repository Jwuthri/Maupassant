import os
import glob

import streamlit as st

from maupassant.settings import MODEL_PATH
from maupassant.stream_app.trained_model import TrainedModel
from maupassant.stream_app.summarizer import Summarizer
from maupassant.stream_app.normalization import Normalization


@st.cache(allow_output_mutation=True)
def load_tfidf():
    return Summarizer("TfIdf")


def main():
    """Run the streamlit application."""
    st.sidebar.title("Which application ?")
    applications = ["Index", "Summarization", "Normalization", "Trained Models"]
    app_mode = st.sidebar.selectbox("Choose the application", applications)
    if app_mode == "Index":
        st.subheader("Maupassant demo site!")

    elif app_mode == "Summarization":
        st.title("Text Summarization")
        model_name = st.selectbox("Choose the summarizer model", ["TfIdf", "GoogleT5"])
        if model_name == "GoogleT5":
            Summarizer("GoogleT5").main()
        else:
            tfidf_summ = load_tfidf()
            tfidf_summ.main()

    elif app_mode == "Normalization":
        st.title("Text normalization")
        Normalization().main()

    elif app_mode == "Trained Models":
        st.title("Use pretrained models")
        test_cached_models()


def test_cached_models():
    models = glob.glob(os.path.join(MODEL_PATH, "*"))
    model_mapping = {os.path.basename(model): model for model in models if "zip" not in os.path.basename(model)}
    model = st.selectbox("Model", [k for k in model_mapping.keys()])
    if model:
        TrainedModel(path=model_mapping[model]).predict()


if __name__ == '__main__':
    main()
