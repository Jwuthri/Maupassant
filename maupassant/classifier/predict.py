# def load_encoder(self, model_dir):
#     encoders_files = glob.glob(model_dir + "/*encoder.pkl")
#     encoders = {}
#     for file in encoders_files:
#         encoder = pickle.load(open(file, "rb"))
#         encoder_name = os.path.split(file)[1].split('.')[0]
#         encoders[encoder_name] = dict(enumerate(encoder.classes_))
#
#     return encoders


class PredictClassifier(object):

    def __init__(self):
        pass
