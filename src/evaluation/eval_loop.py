import argparse
from typing import List
import pandas as pd
import os
from functools import reduce
import pandas as pd
import pickle, os
from src.evaluation.metrics.llm_judges import metric_llm_judgement
from src.evaluation.metrics.text_similarity import metric_bertscore, metric_bleu, metric_rouge
import inspect
from src.utils import load_GT, load_model_output

def average_score():
    pass

def normalize_score():
    pass


def load_existing_results(output_file_path: str) -> pd.DataFrame:
    """
    Load existing evaluation results if available.
    
    Args:
        output_file_path: Path to the output CSV file
        
    Returns:
        DataFrame with existing results or empty DataFrame
    """
    if os.path.exists(output_file_path):
        try:
            existing_df = pd.read_csv(output_file_path)
            print(f"Loaded existing results from {output_file_path}")
            print(f"Found {len(existing_df)} existing entries")
            return existing_df
        except Exception as e:
            print(f"Could not load existing results from {output_file_path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def get_missing_evaluations(df: pd.DataFrame, existing_results: pd.DataFrame, 
                           metrics_list: List[str], llm_judges: List[str] = None, 
                           criteria: List[str] = None) -> pd.DataFrame:
    """
    Determine which evaluations are missing and need to be computed.
    
    Args:
        df: DataFrame with all data to be evaluated
        existing_results: DataFrame with existing evaluation results
        metrics_list: List of metrics to evaluate
        llm_judges: List of judge models (for LLM judgement metric)
        criteria: List of criteria (for LLM judgement metric)
        
    Returns:
        DataFrame containing only rows that need evaluation
    """
    if existing_results.empty:
        print("No existing results found, evaluating all rows")
        return df
    
    # Get IDs that already have complete results
    completed_ids = set()
    
    for idx, row in existing_results.iterrows():
        row_id = row.get('id')
        if row_id is None:
            continue
            
        # Check if all required metrics are present for this row
        has_all_metrics = True
        
        for metric in metrics_list:
            if metric == "llm_judgement" and llm_judges and criteria:
                # Check for all LLM judgement columns
                answer_types = ['wtnp', 'wtp']  # Add 'wot' if needed
                for answer_type in answer_types:
                    for judge in llm_judges:
                        for criterion in criteria:
                            col_name = f"{answer_type}_{judge}_{criterion}"
                            if col_name not in row or pd.isna(row[col_name]) or row[col_name] == -10:
                                has_all_metrics = False
                                break
                        if not has_all_metrics:
                            break
                    if not has_all_metrics:
                        break
            elif metric == "rouge":
                # Check for ROUGE metrics
                rouge_cols = ['rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1']
                if not all(col in row and not pd.isna(row[col]) for col in rouge_cols):
                    has_all_metrics = False
            elif metric == "bleu":
                # Check for BLEU metric
                if 'bleu_score' not in row or pd.isna(row['bleu_score']):
                    has_all_metrics = False
            elif metric == "bertscore":
                # Check for BERTScore metrics
                bert_cols = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
                if not all(col in row and not pd.isna(row[col]) for col in bert_cols):
                    has_all_metrics = False
            # Add checks for other metrics as needed
            
            if not has_all_metrics:
                break
        
        if has_all_metrics:
            completed_ids.add(row_id)
    
    # Filter out completed IDs
    missing_df = df[~df['id'].isin(completed_ids)]
    
    print(f"Found {len(completed_ids)} rows with complete results")
    print(f"Need to evaluate {len(missing_df)} remaining rows")
    
    return missing_df


def merge_results(existing_results: pd.DataFrame, new_results: pd.DataFrame) -> pd.DataFrame:
    """
    Merge existing results with new results, updating/adding as needed.
    
    Args:
        existing_results: DataFrame with existing evaluation results
        new_results: DataFrame with newly computed results
        
    Returns:
        Combined DataFrame
    """
    if existing_results.empty:
        return new_results
    
    if new_results.empty:
        return existing_results
    
    # Update existing results with new ones
    # Remove existing entries for IDs that were re-evaluated
    existing_filtered = existing_results[~existing_results['id'].isin(new_results['id'])]
    
    # Combine with new results
    combined_results = pd.concat([existing_filtered, new_results], ignore_index=True)
    
    print(f"Merged results: {len(existing_filtered)} existing + {len(new_results)} new = {len(combined_results)} total")
    
    return combined_results


def evaluate_batch(df: pd.DataFrame, eval_fn, *args, **kwargs) -> pd.DataFrame:
    """
    Calls eval_fn(row, *args, **kwargs) for each row with NO fallback.
    Unknown kwargs can be filtered or raise, depending on 'strict'.
    """
    strict = True  # set False to silently drop unknown kwargs

    # Work out which kwargs the function actually accepts
    sig = inspect.signature(eval_fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if kwargs:
        if accepts_var_kw:
            call_kwargs = kwargs
        else:
            allowed = {name for name, p in params.items()
                       if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                     inspect.Parameter.KEYWORD_ONLY)}
            unknown = set(kwargs) - allowed
            if strict and unknown:
                raise TypeError(
                    f"{eval_fn.__name__} got unexpected keyword(s): {sorted(unknown)}"
                )
            call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    else:
        call_kwargs = {}

    records = []

    for _, row in df.iterrows():
        # Filter by specific IDs if needed (uncomment and modify as required)
        if row["id"] in list(df["id"]): # Add specific IDs to process
            print(f"Currently running the idx {row['id']} from the dataset!!")
            # No fallback: if this fails, you'll see the real error
            metrics_output = eval_fn(row, *args, **call_kwargs)

            if not isinstance(metrics_output, dict):
                raise ValueError("eval_fn must return a dict of metric_name: value")

            records.append({"id": row["id"], **metrics_output})

    return pd.DataFrame.from_records(records)


def load_results_from_pickle(pkl_path):
    """Load results dataframe from pickle file if it exists"""
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as in_f:
                return pd.read_pickle(in_f)
        except (pickle.PickleError, EOFError):
            print(f"Could not load existing results from {pkl_path}, starting fresh")
            return None
    return None


def get_ground_truth(WORKSPACE, DATASET, TYPE, add_type):

    if DATASET == "bioasq":
        final_df = pd.read_csv(f"{WORKSPACE}/all_datasets/BioASQ-training12b/complete_summary_test_set.csv")
        #final_df= final_df.head(3)
    elif DATASET == "my_dataset":
        #questions = pd.read_csv(f"{WORKSPACE}/mini_dataset/human_written/SB_60Q.csv")
        final_df = load_GT(f"{WORKSPACE}/all_datasets/my_dataset/gt_entries_dec11.jsonl")
        
        final_df["total_intermediate_steps"] = None
        final_df["all_tools_list"] = None
        for index, row in final_df.iterrows():
            tp = row["parsed_thoughtprocess"]
            final_df["total_intermediate_steps"][index] = len(tp) - 1
            all_tools_list = []
            for trace in row["parsed_thoughtprocess"]:
                all_tools_list.append(trace.get("action", ""))
            cleaned_list = [x for x in all_tools_list if x is not None]
            final_df["all_tools_list"][index] = cleaned_list
            

 
    final_df["model_type"]="Human"  
    
    claude_results_filtered = load_results_from_pickle(f"{WORKSPACE}/all_outputs/{DATASET}/{TYPE}/claude-sonnet-4-5-20250929_{add_type}results.pkl")
    gemini_results_filtered = load_results_from_pickle(f"{WORKSPACE}/all_outputs/{DATASET}/{TYPE}/gemini-2.5-pro_{add_type}results.pkl")
    gpt_results_filtered = load_results_from_pickle(f"{WORKSPACE}/all_outputs/{DATASET}/{TYPE}/gpt-4.1-2025-04-14_{add_type}results.pkl")
    llama_results_filtered = load_results_from_pickle(f"{WORKSPACE}/all_outputs/{DATASET}/{TYPE}/meta-llama_Llama-3.3-70B-Instruct_{add_type}results.pkl")

    gpt_results_filtered["model_type"] = "GPT-4.1"
    llama_results_filtered["model_type"] = "LLAMA-3.3-70B"
    claude_results_filtered["model_type"] = "Claude-3.5-Sonnet"
    gemini_results_filtered["model_type"] = "Gemini-2.5-Pro"
    
    gpt_results_filtered["all_tools_list"] = None
    for index, row in gpt_results_filtered.iterrows():
        tp = row["wtnp_thought_process"]
        all_tools_list=[]
        for trace in tp:
            all_tools_list.append(trace.get("Action", ""))
        cleaned_list = [x for x in all_tools_list if x is not None]
        gpt_results_filtered["all_tools_list"][index] = cleaned_list
        

    llama_results_filtered["all_tools_list"] = None
    for index, row in llama_results_filtered.iterrows():
        tp = row["wtnp_thought_process"]
        all_tools_list=[]
        if tp !=None:
            for trace in tp:
                all_tools_list.append(trace.get("Action", ""))
        cleaned_list = [x for x in all_tools_list if x is not None]
        llama_results_filtered["all_tools_list"][index] = cleaned_list


    return final_df, gpt_results_filtered, llama_results_filtered, claude_results_filtered, gemini_results_filtered

# — Main loop —

def main():
    parser = argparse.ArgumentParser(
        description="Launch evaluation pipeline with smart caching"
    )

    parser.add_argument(
        "--workspace", "-w",
        default="/data/horse/ws/chja176b-sysbio_cj2025/chja176b-sysbio_cj-1737284518/git_repos/UI_SYSBIO_Sol_Lib",
        help="Path to the workspace directory"
    )

    parser.add_argument(
        "--type", "-t",
        required=True,
        help="few shot examples or not"
    )

    parser.add_argument(
        "--dataset", "-d",
        required=True,
        default="my_dataset",
        help="Choose between my_dataset or bioasq datasets"
    )

    parser.add_argument(
        "--force-recompute", "-f",
        action="store_true",
        help="Force recomputation of all metrics, ignoring cache"
    )

    args = parser.parse_args()
    WORKSPACE = args.workspace
    DATASET = args.dataset
    TYPE = args.type
    FORCE_RECOMPUTE = args.force_recompute

    if TYPE == "few_shot" or TYPE == "DPP_corrected_few_shot_3":
        add_type = "few_shot_"
    else:
        add_type = ""

    # — 0. Define exactly which metrics you want to compute/average —
    metrics_list = [
        # "rouge",
        # "bleu",
        # "bertscore",
        'llm_judgement',
        #'reference_overlap',
    ]

    gt, gpt_results_filtered, llama_results_filtered, claude_results_filtered, gemini_results_filtered = get_ground_truth(WORKSPACE, DATASET, TYPE, add_type)
    gt = gt.rename(columns={'final_answer': 'answer'})
    model_files_dfs = {
        'claude': claude_results_filtered,
        'gemini': gemini_results_filtered,
        'gpt': gpt_results_filtered,
        'llama': llama_results_filtered,
    }

    print("HERE ARE THE MODEL OUTPUT SHAPES: ", model_files_dfs["claude"].shape, model_files_dfs["gemini"].shape, model_files_dfs["gpt"].shape, model_files_dfs["llama"].shape)

    # Pre-define each component's metric names
    llm_judges = ["openai/gpt-oss-120b", "Qwen/Qwen3-Coder-30B-A3B-Instruct"] 
    llm_judgement_criterion = ["readability", "substance_density", "relevance", "expert_suitability", "coverage"]

    # Create output directory
    output_dir = f"{WORKSPACE}/RUN_DEC_16/llm_judge_report_evaluation/outputs/{DATASET}/{TYPE}/report_n_references/files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create cache directory
    cache_dir = f"{WORKSPACE}/RUN_DEC_16/llm_judge_report_evaluation/cache/{DATASET}/{TYPE}"
    os.makedirs(cache_dir, exist_ok=True)

    all_model_dfs = []

    for model_name, output_df in model_files_dfs.items():
        print(f"Processing outputs for {model_name}…")


        # Define output file paths
        individual_output_file = f"{output_dir}/llm_evaluation_individual_{model_name}.csv"
        
        # Load existing results if not forcing recompute
        existing_individual_results = pd.DataFrame()
        if not FORCE_RECOMPUTE:
            existing_individual_results = load_existing_results(individual_output_file)

        # Determine what needs to be evaluated
        if FORCE_RECOMPUTE:
            print(f"Force recompute enabled, evaluating all {len(output_df)} rows for {model_name}")
            rows_to_evaluate = output_df
        else:
            rows_to_evaluate = get_missing_evaluations(
                output_df, existing_individual_results, metrics_list, 
                llm_judges, llm_judgement_criterion
            )

        dfs = []
        
        if len(rows_to_evaluate) > 0:
            print(f"Evaluating {len(rows_to_evaluate)} rows for {model_name}")
            
            for metric in metrics_list:
                if metric == "rouge":
                    rouge_results = evaluate_batch(rows_to_evaluate, metric_rouge)
                    dfs.append(rouge_results)
                elif metric == "bleu":
                    bleu_results = evaluate_batch(rows_to_evaluate, metric_bleu)
                    dfs.append(bleu_results)
                elif metric == "bertscore":
                    bertscore_results = evaluate_batch(rows_to_evaluate, metric_bertscore)
                    dfs.append(bertscore_results)
                elif metric == "llm_judgement":
                    # Set up cache file for LLM judgement
                    cache_file = f"{cache_dir}/llm_judgement_cache_{model_name}.pkl"
                    kwargs = {
                        "llm_judges": llm_judges, 
                        "criteria": llm_judgement_criterion,
                        "cache_file_path": cache_file
                    }
                    print("LIST OF ACTIVE LLM JUDGES-----")
                    llm_judgement_results = evaluate_batch(rows_to_evaluate, metric_llm_judgement, **kwargs)
                    dfs.append(llm_judgement_results)

            # Combine new results
            if dfs:
                new_metrics_df = reduce(
                    lambda left, right: left.merge(right, on="id", how="inner"), dfs
                )
                new_metrics_df["model"] = model_name
            else:
                new_metrics_df = pd.DataFrame()

        else:
            print(f"No new evaluations needed for {model_name}")
            new_metrics_df = pd.DataFrame()

        # Merge with existing results
        if not existing_individual_results.empty and not new_metrics_df.empty:
            # Combine existing and new results
            final_metrics_df = merge_results(existing_individual_results, new_metrics_df)
        elif not existing_individual_results.empty:
            # Only existing results
            final_metrics_df = existing_individual_results
        elif not new_metrics_df.empty:
            # Only new results
            final_metrics_df = new_metrics_df
        else:
            # No results at all - this shouldn't happen but handle gracefully
            print(f"Warning: No results found for {model_name}")
            continue

        # Merge with full output data for additional calculations
        combined_df = pd.merge(output_df, final_metrics_df, on="id", how="inner")
        
        # Calculate additional metrics
        # combined_df['Tokens_per_step'] = combined_df.apply(
        #     lambda row: row['wtp_total_tokens'] / row['total_intermediate_steps'] if pd.notnull(row['wtp_total_tokens']) and pd.notnull(row['total_intermediate_steps']) and row['total_intermediate_steps'] != 0 else None, axis=1
        # )

        # combined_df['Completion_by_Prompt_ratio'] = combined_df.apply(
        #     lambda row: (
        #         row['wtp_completion_tokens'] / (row['wtp_total_tokens'] - row['wtp_completion_tokens'])
        #         if pd.notnull(row['wtp_completion_tokens']) and
        #         pd.notnull(row['wtp_total_tokens']) and
        #         (row['wtp_total_tokens'] - row['wtp_completion_tokens']) != 0
        #         else None
        #     ),
        #     axis=1
        # )
        
        # combined_df['Tool_density'] = combined_df.apply(
        #     lambda row: row['total_intermediate_steps'] / row['unique_tool_count'] if pd.notnull(row['total_intermediate_steps']) and pd.notnull(row['unique_tool_count']) and row['unique_tool_count'] != 0 else None,
        #     axis=1
        # )
        
        # Save individual model results
        combined_df.to_csv(individual_output_file, index=False)
        print(f"Saved individual results for {model_name} to {individual_output_file}")
        
        all_model_dfs.append(final_metrics_df.reset_index(drop=True))

    # 6. Concatenate per-id results for all models
    if all_model_dfs:
        final_sheet = pd.concat(all_model_dfs, ignore_index=True)
        
        without_avg_file = f"{output_dir}/llm_evaluation_without_averages.csv"
        final_sheet.to_csv(without_avg_file, index=False)
        print(f"✅ Done-per-model in {without_avg_file}")

        # Calculate averages
        numeric_cols = final_sheet.select_dtypes(include='number').columns
        avg_df = (
            final_sheet
            .groupby('model')[numeric_cols]
            .mean()
            .reset_index()
        )
        avg_df['id'] = 'average'

        # Put 'id' first, 'model' second, then the rest
        avg_df = avg_df[['id','model'] + numeric_cols.tolist()]

        # Save averages
        with_avg_file = f"{output_dir}/llm_evaluation_with_averages.csv"
        avg_df.to_csv(with_avg_file, index=False)
        print(f"✅ Done- per-model averages in {with_avg_file}")
    else:
        print("Warning: No results to process")

if __name__ == "__main__":
    main()