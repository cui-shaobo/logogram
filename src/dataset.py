import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class T5TitleDataset(Dataset):
    def __init__(self, jsonl_file, model_mode, shorthand_mode, tokenizer, max_length=512, max_decode_step=64, debug=False, temp_csv_file=None):
        self.max_length = max_length
        self.max_decode_step = max_decode_step
        self.model_mode = model_mode    # Specify different types of models we want to train for different methods
        self.shorthand_mode = shorthand_mode
        self.tokenizer = tokenizer
        
        # Load data
        with open(jsonl_file, 'r') as f:
            lines = f.read().splitlines()
        self.json_data = [json.loads(line) for line in lines]
        
        # Load temporary csv file
        if temp_csv_file is not None:
            self.temp_csv_data = pd.read_csv(temp_csv_file)
            assert len(self.temp_csv_data) == len(self.json_data), "Length of temporary csv file does not match length of json data"
        else:
            self.temp_csv_data = None
        
        # Set total size
        self.total_size = len(self.json_data)
        if debug:
            self.total_size = 500
        
    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        json_dict = self.json_data[idx]
        abbreviation = json_dict["Acronym"]
        if self.shorthand_mode == "character":
            abbreviation = " ".join(list(abbreviation))

        if self.model_mode == "abstract2description":
            input_text = "Abstract: " + json_dict["Abstract"] + " Title: "
            output_text = json_dict["Description"]
        elif self.model_mode == "abstract-description2shorthand":
            if self.temp_csv_data is not None:
                temp_csv_dict = self.temp_csv_data.iloc[idx].to_dict()
                input_text = "Abstract: " + json_dict["Abstract"] + " Title: " + temp_csv_dict["prediction_description"] + " Abbreviation: "
            else:
                input_text = "Abstract: " + json_dict["Abstract"] + " Title: " + json_dict["Description"] + " Abbreviation: "
            output_text = abbreviation
        elif self.model_mode == "abstract2shorthand:description":    
            input_text = "Abstract: " + json_dict["Abstract"] + " Abbreviation and Title: "
            output_text = abbreviation + ": " + json_dict["Description"]
        elif self.model_mode == "abstract2description:shorthand":
            input_text = "Abstract: " + json_dict["Abstract"] + " Title and Abbreviation: "
            output_text = json_dict["Description"] + ": " + abbreviation
            
        inputs = self.tokenizer.encode_plus(input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        inputs["input_ids"] = inputs["input_ids"].squeeze()
        inputs["attention_mask"] = inputs["attention_mask"].squeeze()
        labels = self.tokenizer.encode(output_text, max_length=self.max_decode_step, padding='max_length', truncation=True, return_tensors='pt').squeeze()
        inputs['labels'] = labels
        return inputs