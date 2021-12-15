# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
import torch
import torch.nn.utils as nn_utils
from tqdm import tqdm
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


class Trainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self,
            model, 
            train_loader, 
            dev_loader, 
            log_dir, 
            log_steps, 
            validate_steps, 
            num_epochs, 
            lr, 
            warm_up_ratio=0.1, 
            weight_decay=0.01, 
            max_grad_norm=0.5,
            use_gpu=True
        ):

        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.validate_steps = validate_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_ratio = warm_up_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        total_steps = len(train_loader) * self.num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
            num_warmup_steps=self.warm_up_ratio * total_steps, 
            num_training_steps=total_steps)
        self.best_metric = 0.0

        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
            self.model.cuda()
        else:
            self.device = torch.device("cpu")
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        """
        Train the model.
        """
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        logging.info("Evaluate every {} batches.".format(self.validate_steps))

        best_model_store_path = os.path.join(self.log_dir, "best_model.bin")
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1))
            for batch_step, inputs in enumerate(tqdm(self.train_loader)):
                self.model.train()
                inputs['user_profile'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['user_profile']]
                inputs['knowledge'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['knowledge']]
                inputs['conversation'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['conversation']]
                inputs['plans'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['plans']]
                inputs['target'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['target']]

                loss = self.model(inputs)['loss']
                loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if batch_step > 0 and batch_step % self.log_steps == 0:
                    logging.info("Batch Step: {}\tloss: {:.3f}".format(batch_step, loss.item()))
                if batch_step > 0 and batch_step % self.validate_steps == 0:
                    logging.info("Evaluating...")
                    predicts_dict = self.evaluate(loader=self.dev_loader)
                    logging.info("Evaluation Acc: {:.3f}".format(predicts_dict["avg_acc"]))
                    if predicts_dict["avg_acc"] > self.best_metric:
                        self.best_metric = predicts_dict["avg_acc"]
                        logging.info("Epoch {} Batch Step {} -- Best Acc: {:.3f} -- PPL: {:.3f}".format(epoch + 1, batch_step, self.best_metric, predicts_dict['avg_ppl']))
                        torch.save(self.model, best_model_store_path)
                        logging.info("Saved to [%s]" % best_model_store_path)
            predicts_dict = self.evaluate(loader=self.dev_loader)
            if predicts_dict["avg_acc"] > self.best_metric:
                self.best_metric = predicts_dict["avg_acc"]
                logging.info("Epoch {} Best Avg Acc: {:.3f} -- PPL: {:.3f}".format(epoch, self.best_metric, predicts_dict['avg_ppl']))
                torch.save(self.model, best_model_store_path)
                logging.info("Saved to [%s]" % best_model_store_path)
            logging.info("Epoch {} training done.".format(epoch + 1))
            model_to_save = os.path.join(self.log_dir, "model_epoch_%d.bin"%(epoch + 1))
            torch.save(self.model, model_to_save)
            logging.info("Saved to [%s]" % model_to_save)

    def evaluate(self, loader):
        """
        Use trained model to perform evaluation.
        Args:
            loader: the DataLoader containing the data to run evaluation.         
        Returns:
            avg_acc: average accuracy.
            avg_ppl: average perplexity.
        """
        self.model.eval()
        
        total_acc = 0.0
        count_tok = 0.0
        ppls = []

        for inputs in tqdm(loader):
            inputs['user_profile'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['user_profile']]
            inputs['knowledge'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['knowledge']]
            inputs['conversation'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['conversation']]
            inputs['plans'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['plans']]
            inputs['target'] = [d.to(self.device).transpose(0,1).contiguous() for d in inputs['target']]
            with torch.no_grad():
                output = self.model(inputs)
                acc = output['accuracy']
                total_tokens = output['total_tokens']
                total_acc += acc
                count_tok += total_tokens
                ppl = float(torch.exp(output['loss']))
                ppls.append(ppl)
        
        avg_ppl = np.mean(ppls)
        avg_acc = total_acc / count_tok
        return_dict = {
            "avg_acc": avg_acc,
            "avg_ppl": avg_ppl,
        }
        return return_dict