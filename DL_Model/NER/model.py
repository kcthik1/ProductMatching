from transformers import BertForTokenClassification

class NutritionNERModel(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
