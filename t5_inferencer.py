from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import NamedTuple
import torch
import os
import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm


class BruteT5Inferencer(object):
    def __init__(self, model, args, test_dataset, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        
        self.set_seed(args.seed)
        
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.per_device_test_batch_size, shuffle=False, drop_last=False)
        
        self.accelerator = Accelerator()
        
        # Distribute to multiple GPUs
        self.model, self.test_dataloader = self.accelerator.prepare(
            self.model, self.test_dataloader
        )
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def predict(self):
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels.
        
        Returns:
            predictions: prediction texts
            labels: ground truth texts
        """
        self.model.eval()
        
        total_steps = len(self.test_dataloader)
        
        # Initialize the progress bar
        progress_bar = tqdm(total=total_steps, desc='Testing', position=0, leave=True, disable=not self.accelerator.is_local_main_process)
        
        generated_texts = []
        labels_texts = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                inputs = self._prepare_inputs(batch)
                generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=64,
                    num_beams=4,
                    early_stopping=True
                )
                batch_outputs = self.accelerator.pad_across_processes(generated_tokens, dim=1)
                batch_labels = self.accelerator.pad_across_processes(inputs['labels'], dim=1)
                batch_outputs = self.accelerator.gather(batch_outputs).cpu().tolist()
                batch_labels = self.accelerator.gather(batch_labels).cpu().tolist()
                generated_texts.extend([self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in batch_outputs])
                labels_texts.extend([self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in batch_labels])
                
                progress_bar.set_description(f'Step: {batch_idx}')
                progress_bar.update()
        
        # Close the progress bar at the end of training
        progress_bar.close()
           
        return generated_texts, labels_texts
    
    def save_predictions_to_csv(self, predictions, labels):
        """
        Convert tokens of predictions and labels to real text and export to a CSV file.
        Args:
            predictions: prediction texts
            labels: ground truth texts
        """
        if self.accelerator.is_local_main_process:
            
            if self.args.model_mode == "abstract2description" or self.args.model_mode == "abstract-description2shorthand":
                self.save_pipeline_predictions_to_csv(predictions, labels)
                return

            # Split shorthand and description
            if self.args.model_mode == "abstract2shorthand:description":
                predictions_shorthand = [pred.split(":")[0].strip() for pred in predictions]
                predictions_description = ["".join(pred.split(":")[1:]).strip() for pred in predictions]
                labels_shorthand = [label.split(":")[0].strip() for label in labels]
                labels_description = ["".join(label.split(":")[1:]).strip() for label in labels]
                
                if self.args.shorthand_mode == "character":
                    predictions_shorthand = ["".join(pred.split()).strip() for pred in predictions_shorthand]
                    labels_shorthand = ["".join(label.split()).strip() for label in labels_shorthand]
                    
            elif self.args.model_mode == "abstract2description:shorthand":
                predictions_shorthand = ["".join(pred.split(":")[1:]).strip() for pred in predictions]
                predictions_description = [pred.split(":")[0].strip() for pred in predictions]
                labels_shorthand = ["".join(label.split(":")[1:]).strip() for label in labels]
                labels_description = [label.split(":")[0].strip() for label in labels]
                
                if self.args.shorthand_mode == "character":
                    predictions_shorthand = ["".join(pred.split()).strip() for pred in predictions_shorthand]
                    labels_shorthand = ["".join(label.split()).strip() for label in labels_shorthand]
            
            # Save predictions and ground truth to a CSV
            df = pd.DataFrame({
                'prediction_shorthand': predictions_shorthand,
                'ground_truth_shorthand': labels_shorthand,
                'prediction_description': predictions_description,
                'ground_truth_description': labels_description
            })
            df = df.drop_duplicates(subset=["prediction_description", "ground_truth_description"]).reset_index(drop=True)
            
            # Fill nan values
            if self.args.model_mode == "abstract2shorthand:description":
                description = df.loc[df['prediction_description']=="", 'prediction_shorthand']
                df.loc[df['prediction_description']=="", 'prediction_shorthand'] = ""
                df.loc[df['prediction_description']=="", 'prediction_description'] = description
            elif self.args.model_mode == "abstract2description:shorthand":
                df.loc[df['prediction_shorthand']=="", 'prediction_shorthand'] = ""

            df.to_csv(self.args.prediction_save_path, index=False)
            
    def save_pipeline_predictions_to_csv(self, predictions, labels):
        """
        Convert tokens of predictions and labels to real text and export to a CSV file for the pipeline mode.
        Args:
            predictions: prediction texts
            labels: ground truth texts
        """
        if self.accelerator.is_local_main_process:

            if self.args.model_mode == "abstract2description":
                # Save predictions and ground truth to a CSV
                df = pd.DataFrame({
                    'prediction_description': predictions,
                    'ground_truth_description': labels
                })
                df = df.drop_duplicates(subset=["prediction_description", "ground_truth_description"]).reset_index(drop=True)
                df.to_csv(self.args.prediction_save_path, index=False)
                    
            elif self.args.model_mode == "abstract-description2shorthand":
                assert os.path.isfile(self.args.prediction_save_path), "Please save the intermediate predictions in abstract2description mode first."
                df_description = pd.read_csv(self.args.prediction_save_path)
                
                predictions_shorthand = predictions
                labels_shorthand = labels
                
                if self.args.shorthand_mode == "character":
                    predictions_shorthand = ["".join(pred.split()).strip() for pred in predictions_shorthand]
                    labels_shorthand = ["".join(label.split()).strip() for label in labels_shorthand]
            
                # Save predictions and ground truth to a CSV
                df_shorthand = pd.DataFrame({
                    'prediction_shorthand': predictions_shorthand,
                    'ground_truth_shorthand': labels_shorthand
                })
                df_shorthand = df_shorthand.loc[:1000]
                df = pd.concat([df_description, df_shorthand], axis=1)
                df = df.drop_duplicates(subset=["prediction_description", "ground_truth_description"]).reset_index(drop=True)
                df = df[['prediction_shorthand', 'ground_truth_shorthand', 'prediction_description', 'ground_truth_description']]
                df.to_csv(self.args.prediction_save_path, index=False)
        
    def _prepare_inputs(self, inputs):
        return inputs