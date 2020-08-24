import os
import glob

import streamlit as st

from maupassant.settings import MODEL_PATH
from maupassant.stream_app.featuring import Featuring
from maupassant.stream_app.full_application import TextApplication
from maupassant.stream_app.trained_model import TrainedModel, TrainedModelV2
from maupassant.stream_app.summarizer import Summarizer
from maupassant.stream_app.normalization import Normalization


@st.cache(allow_output_mutation=True)
def load_tfidf():
    return Summarizer("TfIdf")


def main():
    """Run the streamlit application."""
    st.sidebar.title("Which application ?")
    applications = ["Index", "Summarization", "Normalization", "Models", "Models2", "Featuring", "Application"]
    app_mode = st.sidebar.selectbox("Choose the application", applications)
    if app_mode == "Index":
        st.title("Maupassant demo site!")

    elif app_mode == "Summarization":
        st.title("Text Summarization")
        model_name = st.selectbox("Choose the text_summarization model", ["TfIdf", "GoogleT5"])
        if model_name == "GoogleT5":
            Summarizer("GoogleT5").main()
        else:
            tfidf_summ = load_tfidf()
            tfidf_summ.main()

    elif app_mode == "Normalization":
        st.title("Text normalization")
        Normalization().main()

    elif app_mode == "Models":
        st.title("Use pretrained models")
        cached_models()

    elif app_mode == "Models2":
        st.title("Use pretrained models")
        cached_models2()

    elif app_mode == "Featuring":
        st.title('Text feature extraction')
        model_name = st.selectbox("Choose the feature extraction model", ["TfIdf", "MultilangEmbedding"])
        Featuring(model_name).main()

    elif app_mode == "Application":
        st.title("Text Application")
        TextApplication().main()


def cached_models():
    models = glob.glob(os.path.join(MODEL_PATH, "*"))
    model_mapping = {os.path.basename(model): model for model in models if "zip" not in os.path.basename(model)}
    model = st.selectbox("Model", [k for k in model_mapping.keys()])
    if model:
        TrainedModel(path=model_mapping[model]).predict()


def cached_models2():
    models = glob.glob(os.path.join(MODEL_PATH, "*"))
    model_mapping = {os.path.basename(model): model for model in models if "zip" not in os.path.basename(model)}
    model = st.selectbox("Model", [k for k in model_mapping.keys()])
    if model:
        TrainedModelV2(path=model_mapping[model]).predict3()


if __name__ == '__main__':
    main()
