import torch
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EncoderRanker:
    def __init__(self, model_referance, model_dir):
        self.model_dir = model_dir
        self.cross_encoder = CrossEncoder(
            model_referance,
            model_kwargs={"cache_dir": model_dir},
        )
        self.temperature = 2

    def get_probabilities(self, query, text_list):
        scores = self.cross_encoder.predict([(query, x) for x in text_list])
        probs = torch.softmax(torch.tensor(scores / self.temperature), dim=-1)
        return probs


class CosineRanker:
    def __init__(self, model_referance, model_dir):
        self.model_dir = model_dir
        self.model = SentenceTransformer(
            model_referance, model_kwargs={"cache_dir": model_dir}
        )

    def get_probabilities(self, query, text_list):
        embeddings1 = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )
        embeddings2 = self.model.encode(
            [f"query: {t}" for t in text_list],
            normalize_embeddings=True,
        )

        scores = cosine_similarity(embeddings1, embeddings2)[0]
        probs = torch.softmax(torch.tensor(scores), dim=-1)
        return probs
