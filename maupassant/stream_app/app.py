import os
import glob

import streamlit as st

from maupassant.stream_app.inference import Inference
from maupassant.settings import MODEL_PATH


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
        st.write(
            "In this website, you can try a certain number of our apps in development.  "
            "Select the application you want to try on the drop-down on your left (click on index)  "
            "Have fun!"
        )
    elif app_mode == "SHOW_CODE":
        st.code(code)
    elif app_mode == "TRY_MODEL":
        models = glob.glob(os.path.join(MODEL_PATH, "*"))
        model_mapping = {}
        for model in models:
            name = os.path.basename(model)
            model_mapping[name] = model

        model = st.selectbox("Model", [k for k in model_mapping.keys()])
        if model != "":
            Inference(path=model_mapping[model]).main()


if __name__ == '__main__':
    main()
