import pickle

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from ttk.utils import text_format, timer


class LabelEncoding(object):

    def __init__(self, multi_label):
        self.multi_label = multi_label
        self.classes_mapping = dict()
        self.nb_classes = 0
        self.lb = self.init_lb(multi_label)

    @staticmethod
    def init_lb(multi_label):
        if multi_label:
            return MultiLabelBinarizer()
        else:
            return LabelEncoder()

    @timer
    def fit_lb(self, y):
        self.lb.fit(y)
        self.classes_mapping = dict(enumerate(self.lb.classes_))
        self.nb_classes = len(self.lb.classes_)

    @timer
    def transform_lb(self, y):
        return self.lb.transform(y)

    def inverse_lb(self, y):
        return self.lb.inverse_transform(y)

    def save_lb(self, filename):
        pickle.dump(self.lb, open(filename, "wb"))

    def show_classes(self):
        f_blue = text_format(txt_color='blue')
        end = text_format(end=True)
        for (i, label) in enumerate(self.lb.classes_):
            print(f"{i}. {f_blue}{label}{end}")
