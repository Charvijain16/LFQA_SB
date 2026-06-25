import pandas as pd
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from bert_score import score as bert_score
import torch
rouge = evaluate.load("rouge")


def metric_rouge(row: pd.Series) -> dict:

    if row["wtp_answer"] == None:
        results_wtp = {"rouge1":0,"rouge2":0, "rougeL":0 }
    else:
        # Compute the metrics
        results_wtp = rouge.compute(
            predictions=[row["wtp_answer"]], references=[row["answer"]]
        )

    if row["wtnp_answer"] == None:
        results_wtnp = {"rouge1":0,"rouge2":0, "rougeL":0 }
    else:
        results_wtnp = rouge.compute(
            predictions=[row["wtnp_answer"]], references=[row["answer"]]
        )
        
    if row["wot_answer"] == None:
        results_wot = {"rouge1":0,"rouge2":0, "rougeL":0 }
    else:
        results_wot = rouge.compute(
            predictions=[row["wot_answer"]], references=[row["answer"]]
        )

    return {
        "wtp_rouge1": results_wtp["rouge1"],
        "wtp_rouge2": results_wtp["rouge2"],
        "wtp_rougeL": results_wtp["rougeL"],
        "wtnp_rouge1": results_wtnp["rouge1"],
        "wtnp_rouge2": results_wtnp["rouge2"],
        "wtnp_rougeL": results_wtnp["rougeL"],
        "wot_rouge1": results_wot["rouge1"],
        "wot_rouge2": results_wot["rouge2"],
        "wot_rougeL": results_wot["rougeL"],
    }


def metric_bleu(row: pd.Series) -> dict:
    smoothie = SmoothingFunction().method4
    gt_sents = nltk.sent_tokenize(row["answer"])
    gt_tokens = [nltk.word_tokenize(sent) for sent in gt_sents]

    
    
    if row["wtp_answer"] == None:
        wtp_bleu_score = 0
    else:
        wtp_hyp_tokens = []
        for sent in nltk.sent_tokenize(row["wtp_answer"]):
            wtp_hyp_tokens.extend(nltk.word_tokenize(sent))

        wtp_bleu_score = sentence_bleu(
            gt_tokens, wtp_hyp_tokens, smoothing_function=smoothie
        )
        
    if row["wtnp_answer"] == None:
        wtnp_bleu_score = 0
    else:    
        wtnp_hyp_tokens = []
        for sent in nltk.sent_tokenize(row["wtnp_answer"]):
            wtnp_hyp_tokens.extend(nltk.word_tokenize(sent))
        wtnp_bleu_score = sentence_bleu(
            gt_tokens, wtnp_hyp_tokens, smoothing_function=smoothie
        )
        
    if row["wot_answer"] == None:
        wot_bleu_score = 0
    else:    
        wot_hyp_tokens = []
        for sent in nltk.sent_tokenize(row["wot_answer"]):
            wot_hyp_tokens.extend(nltk.word_tokenize(sent))
        wot_bleu_score = sentence_bleu(
            gt_tokens, wot_hyp_tokens, smoothing_function=smoothie
        )

    return {
        "wtp_bleu": wtp_bleu_score,
        "wtnp_bleu": wtnp_bleu_score,
        "wot_bleu": wot_bleu_score,
    }


def metric_bertscore(row: pd.Series) -> dict:
    # Compute the metrics
    
    if row["wtp_answer"] ==None:
        wtp_bertscore = torch.tensor([0.0])
    else:
        wtp_bertscore = bert_score(
            [row["wtp_answer"]],
            [row["answer"]],
            lang="en",
            verbose=True,
            rescale_with_baseline=True,
        )
    if row["wtnp_answer"] ==None:
        wtnp_bertscore = torch.tensor([0.0])
    else:
        wtnp_bertscore = bert_score(
            [row["wtnp_answer"]],
            [row["answer"]],
            lang="en",
            verbose=True,
            rescale_with_baseline=True,
        )
    if row["wot_answer"] ==None:
        wot_bertscore = torch.tensor([0.0])
    else:
        wot_bertscore = bert_score(
            [row["wot_answer"]],
            [row["answer"]],
            lang="en",
            verbose=True,
            rescale_with_baseline=True,
        )

    return {
        "wtp_bertscore_F1": wtp_bertscore[0].item(),
        "wtnp_bertscore_F1": wtnp_bertscore[0].item(),
        "wot_bertscore_F1": wot_bertscore[0].item(),
    }

