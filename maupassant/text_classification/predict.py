from maupassant.settings import MODEL_PATH
from maupassant.utils import predict_format, timer, not_none
from maupassant.preprocessing.normalization import TextNormalization
from maupassant.tensorflow_models_compile import BaseTensorflowModel, ModelSaverLoader


class Predictor(BaseTensorflowModel):

    def __init__(self, base_path=MODEL_PATH, name='model_name', cleaning_func=None, gpu=False):
        self.normalizer = TextNormalization()
        self.cleaning_func = cleaning_func
        msl = ModelSaverLoader(base_path, name, True)
        info = msl.load_info()
        label_type = info.get('label_type')
        architecture = info.get('architecture')
        number_labels = info.get('number_labels')
        pretrained_embedding = info.get('pretrained_embedding')
        embedding_size = info.get('embedding_size')
        super().__init__(label_type, architecture, number_labels, pretrained_embedding, base_path, name, True)
        self.use_gpu(gpu)
        self.build_model(0, embedding_size, 0)
        self.model = self.load_weights(self.model)
        self.encoder = self.load_encoder()
        self.classes_mapping = dict(enumerate(self.encoder.classes_))
        _ = self.predict("")

    def clean_text(self, text):
        text = self.normalizer.replace_char_rep(text=text)
        text = self.normalizer.replace_words_rep(text=text)
        if self.cleaning_func:
            text = self.cleaning_func(text)
        text = self.normalizer.remove_multiple_spaces(text=text)
        text = text.strip()

        return text

    @predict_format
    def predict_probabilities(self, x):
        return self.model.predict(x)[0]

    def predict_classes(self, prediction, threshold):
        results = [(self.classes_mapping[label], float(th)) for label, th in enumerate(prediction) if float(th) >= threshold]

        return dict(results)

    # @timer
    @not_none
    def predict(self, x, threshold=0.5, clean=False):
        if clean:
            x = self.clean_text(x)
        probabilities = self.predict_probabilities(x=x)
        prediction = self.predict_classes(probabilities, threshold)

        return {"input_text": x, "predictions": prediction}


if __name__ == '__main__':
    # predictor = Predictor(MODEL_PATH, "2020_09_30_12_25_35_text_classification")
    # predictor = Predictor(MODEL_PATH, "2020_09_30_14_07_42_text_classification")
    # predictor = Predictor(MODEL_PATH, "2020_09_30_14_56_14_text_classification")
    predictor = Predictor(MODEL_PATH, "2020_09_30_20_35_10_intent")
    # predictor = Predictor(MODEL_PATH, "2020_09_30_16_49_55_sentiment")


    t = [
        "My order is damaged, I want a refund",
        "We Are Feel I will take them for life xxx❤️",
        "I was hoping you could help with pick which kind of tape I need",
        """I was hoping you could help with pick which kind of tape I need
I am a 34DD but I’m 33 years old and have faint and lost a lot of weight over the years so they’re not very full or firm.
The user sent some attachments.""",
        "cancel my order",
        "talk to someone",
        "Can you please cancel this order?",
        "I ordered tape i gmltOrderId and didnt mean to order it twice",
        "I ordered tape i gmltOrderId",
        "didnt mean to order it twice",
        "Can you please cancel this order? I ordered tape i gmltOrderId and didnt mean to order it twice",
        "asap",
        "WTF, nothing there",
        "Love it, thanks for your help",
        "Don't want anything from you anymore",
        "I hate your product",
        "Your serice is really bad",
        "I need this order very fast",
        "You are fucking retard",
        "C'est de la merde",
        "je deteste votre produit",
        "je suis pas content",
        "je n'aime pas votre support",
        "pas satisfait de votre support",
        "tres satisfait de votre support",

        """This item still hasnt arrived yet. Can someone follow up on the shipment that hasn't moved from California since Sept 6?

Usps: 9400111202555908829621""",
        "gmltName 😱",
        "Im gonna buy somewhere else",
        "I'm sorry you don't seem to understand what the sciences chemistry and physics are, but my edits were correct and you are poorly abusing your role as an administrator.",
        "\nIt\'s positive alright....positively insane. Considering that, according to the article anyway, it was supposedly going to replace the ""current"" type of Christianity, it sounds to me like they wern\'t actually compatible at all, in terms of both being Christian."
    ]
    for i in t:
        pred = predictor.predict(i, threshold=0.02)
        print(pred)