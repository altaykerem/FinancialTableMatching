from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from src.str_utils import preprocess_cased_ner


class CompanyExtractor:
    def __init__(self, model_referance, model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_referance, cache_dir=model_dir)
        model = AutoModelForTokenClassification.from_pretrained(
            model_referance, cache_dir=model_dir
        )
        self.ner = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
        )

    def organisations(self, text):
        text = preprocess_cased_ner(text)
        spans = []
        for entity in self.ner(text):
            if entity["entity_group"] != "ORG":
                continue
            start, end, score = entity["start"], entity["end"], float(entity["score"])
            if spans and start <= spans[-1][1] + 1:
                last = spans.pop()
                start, score = last[0], min(last[2], score)
            spans.append((start, end, score))
        return [(text[a:b].strip(), score) for a, b, score in spans]
