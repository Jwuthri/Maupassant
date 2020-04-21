import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from maupassant.utils import timer


class Predictor(object):

    def __init__(self, model_name='t5'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.set_model()

    @timer
    def set_model(self):
        if self.model_name == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.device = torch.device('cpu')

    @timer
    def summarize(self, tokenized_text):
        return self.model.generate(
            tokenized_text, num_beams=4, no_repeat_ngram_size=2,
            min_length=30, max_length=100, early_stopping=True)

    @timer
    def predict(self, text):
        processed_text = text
        if self.model_name == "t5":
            processed_text = "summarize: " + text.strip()

        tokenized_text = self.tokenizer.encode(processed_text, return_tensors="pt").to(self.device)
        relevant_indexes = self.summarize(tokenized_text)

        return self.tokenizer.decode(relevant_indexes[0], skip_special_tokens=True)


if __name__ == '__main__':
    p = Predictor()
    summarized = p.predict('Hello,I came across your Instagram and I absolutely love your clothing, they look absolutely amazing! I’m a lifestyle/ fashion/ beauty content creator based in Perth, Western Australia and I am very interested in collaborating with you. I think a collaboration would greatly benefit both myself and your company in many ways! The age range in my audience is 58% 18-24 and 26% 25-34 year olds. The top two countries my followers are located are 38% Australia and 10% United States. This would be great for your company to get more exposure other then Australia! This collaboration would benefit myself greatly to help myself grow and help myself get exposure. Plus increase the style in my closet! In case you’d like to check out my Instagram here is a link- instagram.com/__kaylaatkinsonI look forward to hearing from you!')
    print(summarized)
