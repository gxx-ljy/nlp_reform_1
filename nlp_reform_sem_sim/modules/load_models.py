import spacy

class Models:
    def __init__(self, config):
        self.config = config

        self.spacy_nlp = spacy.load('en_core_web_sm')

        if 'fw' in self.config['signal_detection']:
            from fuzzywuzzy import fuzz
            self.fuzz = fuzz

        if 'bert' in self.config['signal_detection'] or 'PD' in self.config['param_detection']:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(r'D:\codes\req-model\paraphrase-distilroberta-base-v1')
            # self.embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')
            
        if 'TE' in self.config['param_detection']:
            from allennlp.predictors.predictor import Predictor
            self.predictor_TE = Predictor.from_path(
                r"D:\codes\req-model\decomposable-attention-elmo-2020.04.09.tar.gz")
            # self.predictor_TE = Predictor.from_path(
            #     "https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz")

        if 'SA' in self.config['param_detection']:
            from allennlp.predictors.predictor import Predictor
            self.predictor_SA = Predictor.from_path(r"D:\codes\req-\model\basic_stanford_sentiment_treebank-2020.06.09.tar.gz")
            # self.predictor_SA = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models"
            #                                    "/basic_stanford_sentiment_treebank-2020.06.09.tar.gz")