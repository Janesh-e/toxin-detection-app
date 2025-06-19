from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from app.config import HAZARD_MODEL_PATH, PRODUCT_MODEL_PATH, ORIGINAL_MODEL_NAME, HAZARD_CLASSES, PRODUCT_CLASSES
from datasets import Dataset

import warnings
warnings.simplefilter("ignore", FutureWarning)

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")


class BERTPredictor:
    def __init__(self):
        # Load tokenizer once (shared)
        self.tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)

        # Load hazard type model and trainer
        hazard_model = AutoModelForSequenceClassification.from_pretrained(HAZARD_MODEL_PATH)
        self.hazard_trainer = Trainer(model=hazard_model, tokenizer=self.tokenizer)

        # Load product category model and trainer
        product_model = AutoModelForSequenceClassification.from_pretrained(PRODUCT_MODEL_PATH)
        self.product_trainer = Trainer(model=product_model, tokenizer=self.tokenizer)

    def predict(self, text: str):
        # Prepare a HuggingFace dataset from input text
        raw_data = [{"text": text}]
        dataset = Dataset.from_list(raw_data)

        # Tokenize with map (batched for efficiency)
        tokenized_dataset = dataset.map(
            lambda e: self.tokenizer(e["text"], truncation=True, padding=True, max_length=256),
            batched=True
        )

        # Run prediction
        hazard_pred = self.hazard_trainer.predict(tokenized_dataset)
        product_pred = self.product_trainer.predict(tokenized_dataset)
        print(product_pred.predictions.argmax(axis=1)[0])
        # Decode predictions
        hazard_label = HAZARD_CLASSES[hazard_pred.predictions.argmax(axis=1)[0]]
        product_label = PRODUCT_CLASSES[product_pred.predictions.argmax(axis=1)[0]]
        
        return hazard_label, product_label
