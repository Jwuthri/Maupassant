import os
import glob

import streamlit as st

from maupassant.settings import MODEL_PATH
from maupassant.stream_app.try_model import TryModel


def main():
    """Run the streamlit application."""
    code = open(__file__).read()
    st.sidebar.title("Which application ?")
    app_mode = st.sidebar.selectbox(
        "Choose the application",
        [
            "INDEX",
            "SHOW_CODE",
            "TRY_MODEL"
        ],
    )
    if app_mode == "INDEX":
        st.subheader("Maupassant demo site!")
    elif app_mode == "SHOW_CODE":
        st.code(code)
    elif app_mode == "TRY_MODEL":
        models = glob.glob(os.path.join(MODEL_PATH, "*"))
        model_mapping = {}
        for model in models:
            name = os.path.basename(model)
            if ".zip" not in name:
                model_mapping[name] = model

        model = st.selectbox("Model", [k for k in model_mapping.keys()])
        if model != "":
            TryModel(path=model_mapping[model]).predict()


if __name__ == '__main__':
    main()
