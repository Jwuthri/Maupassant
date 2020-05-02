import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from maupassant.utils import timer
from maupassant.summarizer.model import WeightedTfIdf, GoogleT5


class Predictor(object):

    def __init__(self, model_name='t5'):
        self.model_name = model_name
        self.model = self.set_model()

    def set_model(self):
        if self.model_name == "t5":
            return GoogleT5()
        else:
            return WeightedTfIdf()

    @timer
    def predict(self, text):
        return self.model.predict(text)


if __name__ == '__main__':
    p = Predictor()
    summarized = p.predict('Hello,I came across your Instagram and I absolutely love your clothing, they look absolutely amazing! I’m a lifestyle/ fashion/ beauty content creator based in Perth, Western Australia and I am very interested in collaborating with you. I think a collaboration would greatly benefit both myself and your company in many ways! The age range in my audience is 58% 18-24 and 26% 25-34 year olds. The top two countries my followers are located are 38% Australia and 10% United States. This would be great for your company to get more exposure other then Australia! This collaboration would benefit myself greatly to help myself grow and help myself get exposure. Plus increase the style in my closet! In case you’d like to check out my Instagram here is a link- instagram.com/__kaylaatkinsonI look forward to hearing from you!')
    print(summarized)
