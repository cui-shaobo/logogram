from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import NamedTuple
import torch
import os
import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm
import shutil
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model, set_seed
from trl.core import LengthSampler
from transformers import AutoTokenizer
from reward.reward import Reward


class BruteT5Trainer(object):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, optimizer=Adafactor):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        
        self.set_seed(args.seed)
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True, drop_last=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.args.per_device_eval_batch_size, shuffle=False, drop_last=False)
        
        self.accelerator = Accelerator()
        self.optimizer = optimizer(self.model.parameters(), weight_decay=args.weight_decay)
        
        # Distribute to multiple GPUs
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )
        
        self.global_step = 0
        self.logging_loss = 0.0
        self.best_eval_loss = 1e9
        self.epoch = 0
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
    def train(self):
        """
        Train the model.
        """
        self.model.train()
        total_steps = len(self.train_dataloader) * self.args.num_train_epochs
        
        # Initialize the progress bar
        progress_bar = tqdm(total=total_steps, desc='Training', position=0, leave=True, disable=not self.accelerator.is_local_main_process)
    
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch+1
            self.model.train()
            for batch in self.train_dataloader:
                inputs = self._prepare_inputs(batch)
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                # Logging the average loss each step and update the progress bar
                self.logging_loss = self.avg_loss(loss.item(), self.logging_loss)
                progress_bar.set_description(f'Epoch: {epoch+1}/{self.args.num_train_epochs}, Step: {self.global_step}, Avg Loss: {self.logging_loss:.4f}')
                progress_bar.update()
                    
                self.global_step += 1
                
            # Evaluate at the end of each epoch
            eval_loss = self.evaluate()
            self.save_model()
            # Save model checkpoint if needed
            # if eval_loss < self.best_eval_loss:
            #     self.best_eval_loss = eval_loss
            #     self.save_model()
                
        # Close the progress bar at the end of training
        progress_bar.close()

    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            eval_loss: float
        """
        self.model.eval()
        total_steps = len(self.eval_dataloader)
        
        # Initialize the progress bar
        progress_bar = tqdm(total=total_steps, desc='Validation', position=0, leave=True, disable=not self.accelerator.is_local_main_process)
        
        total_eval_loss = 0
        for batch_idx, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                inputs = self._prepare_inputs(batch)
                outputs = self.model(**inputs)
                
                # Logging the average loss each step and update the progress bar
                progress_bar.set_description(f'Step: {batch_idx}, Loss: {outputs.loss.item():.4f}')
                progress_bar.update()
                
                total_eval_loss += outputs.loss.item()
                
        avg_eval_loss = total_eval_loss / len(self.eval_dataloader)
        self.accelerator.print(f'Epoch {self.epoch} - Evaluation loss: {avg_eval_loss}')
        
        # Close the progress bar at the end of training
        progress_bar.close()
        
        return avg_eval_loss
    
    def save_model(self):
        if self.accelerator.is_local_main_process:
            # Save model
            model_save_path = os.path.join(self.args.model_save_path, f"checkpoint-{self.epoch}")
            self.accelerator.unwrap_model(self.model).save_pretrained(model_save_path)
            self.tokenizer.save_pretrained(model_save_path)
            
            # Handle save total limit
            checkpoints = sorted([int(x.split('-')[-1]) for x in os.listdir(self.args.model_save_path) if x.startswith('checkpoint-')], reverse=True)
            if len(checkpoints) > self.args.save_total_limit:
                for chkpt in checkpoints[self.args.save_total_limit:]:
                    shutil.rmtree(os.path.join(self.args.model_save_path, f"checkpoint-{chkpt}"))
                
    def avg_loss(self, loss, running_avg_loss, decay=0.99):
        if running_avg_loss == 0:
            return loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
            return running_avg_loss
        
    def _prepare_inputs(self, inputs):
        return inputs
    
    
class BruteT5PPOTrainer(object):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, optimizer=Adam):
        self.args = args
        self.model = model

        # We create a reference model by sharing layers
        # self.ref_model = create_reference_model(model, num_shared_layers=args.num_shared_layers)
        self.ref_model = create_reference_model(model)
        self.tokenizer = tokenizer
        
        set_seed(args.seed)
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.accelerator = Accelerator()
        self.args.learning_rate *= self.accelerator.num_processes
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        
        self.ppo_config = PPOConfig(
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            log_with=args.log_with,
            ppo_epochs=args.ppo_epochs,
            init_kl_coef=args.init_kl_coef,
            target_kl=args.target_kl,
            mini_batch_size=args.mini_batch_size,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ratio_threshold=args.ratio_threshold,
            kl_penalty=args.kl_penalty,
        )
        
        self.ppo_trainer = PPOTrainer(
            self.ppo_config,
            self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.train_dataset,
            optimizer=self.optimizer,
        )
        
        # We then define the arguments to pass to the `generate` function. These arguments
        # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
        # the `generate` function of the trained model.
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            # "pad_token_id": self.tokenizer.eos_token_id,
            # "eos_token_id": -1,
        }
        self.output_min_length = 48
        self.output_max_length = 64
        self.output_length_sampler = LengthSampler(self.output_min_length, self.output_max_length)

        self.model_save_path = self.args.model_save_path
        
        self.global_step = 0
        self.reward_scorer = Reward(
            tokenizer_type='t5-base',
            wordlikeliness=True,
            lcsratio=True,
            wordcoverage=False
        )
        
    def train(self):
        """
        Train the model.
        """
        
        total_steps = len(self.ppo_trainer.dataloader) * self.args.num_train_epochs
        
        # Initialize the progress bar
        progress_bar = tqdm(total=total_steps, desc='Training', position=0, leave=True, disable=not self.ppo_trainer.accelerator.is_local_main_process)
        
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(self.ppo_trainer.dataloader):
                query_tensors = list(batch["input_ids"].unbind(0))

                # Get response from the policy model
                response_tensors = []
                for query in query_tensors:
                    query = query[query != self.tokenizer.pad_token_id] # Remove padding
                    gen_len = self.output_length_sampler()
                    self.generation_kwargs["max_new_tokens"] = gen_len
                    response = self.ppo_trainer.generate(query, **self.generation_kwargs)
                    response = response.squeeze()[-gen_len:]
                    if response[response != self.tokenizer.pad_token_id].nelement() == 0:
                        response[0] = self.tokenizer.eos_token_id  # If the response is full of padding, replace the first token with eos
                    response_tensors.append(response)
                batch["query"] = [self.tokenizer.decode(q.squeeze(), skip_special_tokens=True) for q in query_tensors]
                batch["response"] = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

                # Split shorthand and description
                texts = batch["response"]
                if self.args.model_mode == "abstract2shorthand:description":
                    shorthands = [text.split(":")[0].strip() for text in texts]
                    descriptions = ["".join(text.split(":")[1:]).strip() for text in texts]
                    if self.args.shorthand_mode == "character":
                        shorthands = ["".join(text.split()) for text in shorthands]        
                elif self.args.model_mode == "abstract2description:shorthand":
                    shorthands = ["".join(text.split(":")[1:]).strip() for text in texts]
                    descriptions = [text.split(":")[0].strip() for text in texts]
                    if self.args.shorthand_mode == "character":
                        shorthands = ["".join(text.split()).strip() for text in shorthands]
                        
                # Compute rewards
                rewards = [torch.tensor(self.reward_scorer.get_reward(description, shorthand)).to(self.accelerator.device) for description, shorthand in zip(descriptions, shorthands)]
                
                # Move query_tensors and response_tensors to device
                query_tensors = [q.to(self.accelerator.device) for q in query_tensors]
                response_tensors = [r.to(self.accelerator.device) for r in response_tensors]

                # Run PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                self.ppo_trainer.log_stats(stats, batch, rewards)
                
                progress_bar.set_description(f'Epoch: {epoch+1}/{self.args.num_train_epochs}, Step: {self.global_step}, Avg Reward: {torch.stack(rewards).mean().item():.4f}')
                progress_bar.update()
                
                self.global_step += 1
            
        # Save model
        if self.ppo_trainer.accelerator.is_main_process:
            self.ppo_trainer.save_pretrained(self.model_save_path)
