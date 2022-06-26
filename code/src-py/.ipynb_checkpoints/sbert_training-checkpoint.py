"""
This script trains sentence transformers with a triplet loss function.
As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime
from zipfile import ZipFile

from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.datasets import NoDuplicatesDataLoader

from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers import util
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from typing import List


import csv
import logging
import os
import sys

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'distilbert-base-uncased'


def train_model(dataset_path, eval_data_path, subset_name, output_path, model_name, num_epochs=3, train_batch_size=16, model_suffix='', data_file_suffix='', max_seq_length=256, 
                add_special_token=False, loss='Triplet', sentence_transformer=False):
    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    output_path = output_path+model_name+ "-" + model_suffix + "-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if sentence_transformer:
        word_embedding_model = SentenceTransformer(model_name)
        word_embedding_model.max_seq_length = max_seq_length
        
        if add_special_token:
            word_embedding_model.tokenizer.add_tokens(['<SEP>'], special_tokens=True)
            word_embedding_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    else:
        word_embedding_model = models.Transformer(model_name)
        word_embedding_model.max_seq_length = max_seq_length
    
        if add_special_token:
            word_embedding_model.tokenizer.add_tokens(['<SEP>'], special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    logger.info("Read Triplet train dataset")
    train_examples = []
    with open(os.path.join(dataset_path, 'training_df_{}.csv'.format(data_file_suffix)), encoding="utf-8") as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if loss == 'ContrastiveLoss':
                train_examples.append(InputExample(texts=[row['argument'], row['keypoint']], label=int(row['label'])))
            else:
                train_examples.append(InputExample(texts=[row['anchor'], row['pos'], row['neg']], label=0))



    if loss == 'MultipleNegativesRankingLoss':
        # Special data loader that avoid duplicates within a batch
        train_dataloader = NoDuplicatesDataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        # Our training loss
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif loss == 'ContrastiveLoss':
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        train_loss = losses.ContrastiveLoss(model)
    else:
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model)
    

    evaluator = KeyPointEvaluator.from_eval_data_path(eval_data_path, subset_name, add_special_token, name='dev', show_progress_bar=False)


    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=500,
              warmup_steps=warmup_steps,
              output_path=output_path)
    

from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'recall' : recall,
        'precision': precision,
        'f1': f1,
    }


class QualityEvaluator(SentenceEvaluator):
    def __init__(self, eval_df, main_distance_function: SimilarityFunction = None, name: str = '', batch_size: int = 16, show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset
        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.eval_df = eval_df
        self.name = name
        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "precision" , "recall", "f1-score"]
        self.write_csv = write_csv


    @classmethod
    def from_eval_data_path(cls, eval_df, **kwargs):       
        return cls(eval_df, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("TripletEvaluator: Evaluating the model on "+self.name+" dataset"+out_txt)

        
        precision, recall, f1, _ = precision_recall_fscore_support(eval_df.Validity.tolist(), eval_df.pred_label.tolist(), average='binary')

        
        mAP_strict, mAP_relaxed = evaluate_predictions(merged_df)
        
        print(f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")
        
        logger.info("mAP strict:   \t{:.2f}".format(mAP_strict*100))
        logger.info("mAP relaxed:   \t{:.2f}".format(mAP_relaxed*100))
        
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, mAP_relaxed, mAP_strict])

            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, mAP_relaxed, mAP_strict])


        return (mAP_strict + mAP_relaxed)/2