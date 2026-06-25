import os
import json
import pickle
import random
import re
import time
import pandas as pd
from langchain_openai import ChatOpenAI

import argparse
from datetime import datetime
import logging

from agents import react


def safe_json_parse(json_str, default=None):
    """Safely parse JSON string with fallback"""
    if json_str is None or json_str == "" or json_str == "None":
        return default
    
    if not isinstance(json_str, str):
        return json_str  # Already parsed
    
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARNING] JSON parse failed: {e}. Input: {str(json_str)[:100]}...")
        return default


def safe_get_item(obj, key, default=None):
    """Safely get item from object, handling type mismatches"""
    try:
        if obj is None:
            return default
        if isinstance(obj, str):
            print(f"[WARNING] Trying to access key '{key}' on string object: {str(obj)[:100]}...")
            return default
        if hasattr(obj, '__getitem__'):
            return obj[key] if key in obj else default
        return default
    except (TypeError, KeyError, IndexError) as e:
        print(f"[WARNING] Failed to get key '{key}': {e}")
        return default


def safe_unpack_result(result, expected_count, operation_name):
    """Safely unpack results with proper error handling"""
    if result is None:
        print(f"[WARNING] {operation_name} returned None")
        return [None] * expected_count
    
    # Handle string results (likely error messages)
    if isinstance(result, str):
        print(f"[WARNING] {operation_name} returned string instead of tuple: {result[:200]}...")
        return [None] * expected_count
    
    # Handle single value instead of tuple
    if not hasattr(result, '__len__') or not hasattr(result, '__getitem__'):
        print(f"[WARNING] {operation_name} returned non-iterable: {type(result)}")
        return [None] * expected_count
    
    try:
        result_list = list(result)
        result_length = len(result_list)
        
        if result_length != expected_count:
            print(f"[WARNING] {operation_name} returned {result_length} values, expected {expected_count}")
            # Pad with None or truncate as needed
            if result_length < expected_count:
                result_list.extend([None] * (expected_count - result_length))
            else:
                result_list = result_list[:expected_count]
        
        return result_list
    except Exception as e:
        print(f"[ERROR] Failed to unpack {operation_name} result: {e}")
        return [None] * expected_count


def safe_assign_to_dataframe(df, idx, col, value):
    """Safely assign a value to a DataFrame cell, handling lists properly"""
    try:
        # List of columns that should contain list/object data
        list_columns = [
            "unique_tools_used", "all_tools_list", "wtp_referenece_ids", 
            "wtnp_referenece_ids", "original_TP", "wtp_thought_process", 
            "wtnp_thought_process"
        ]
        
        if col in list_columns and value is not None and isinstance(value, list):
            # For list columns, ensure the column dtype is object and assign directly
            if df[col].dtype != 'object':
                df[col] = df[col].astype(object)
            df.at[idx, col] = value
        else:
            # For non-list columns, use normal assignment
            df.loc[idx, col] = value
    except Exception as e:
        print(f"[ERROR] Failed to assign value to DataFrame[{idx}, '{col}']: {e}")
        # Try alternative assignment method
        try:
            df.at[idx, col] = value
        except Exception as e2:
            print(f"[ERROR] Alternative assignment also failed: {e2}")


def validate_row_data(row, dataset):
    """Validate that row data is in expected format"""
    if row is None:
        return False, "Row is None"
    
    if isinstance(row, str):
        return False, f"Row is string instead of Series/dict: {row[:100]}..."
    
    try:
        # Check required fields based on dataset
        if dataset == "my_dataset":
            required_fields = ["id", "Name", "category", "question"]
        else:
            required_fields = ["id", "question"]
        
        for field in required_fields:
            if safe_get_item(row, field) is None:
                return False, f"Missing required field: {field}"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"


def process_row(row, react_pipeline, results_columns, dataset, existing_rec=None):
    """Process a single row and return the record dictionary with robust error handling"""
    
    # Validate input row
    is_valid, validation_msg = validate_row_data(row, dataset)
    if not is_valid:
        print(f"[ERROR] Invalid row data: {validation_msg}")
        return None
    
    # Special case handling
    if dataset == "my_dataset" and safe_get_item(row, "id") in [3,6,11,14, 17, 24,50, 52, 54, 56, 57,59]:
        return existing_rec

    # Initialize record safely
    if existing_rec is not None:
        try:
            if isinstance(existing_rec, str):
                print(f"[WARNING] existing_rec is string, creating new record")
                rec = dict.fromkeys(results_columns, None)
            else:
                rec = existing_rec.copy() if hasattr(existing_rec, 'copy') else dict(existing_rec)
                rec["error"] = None
        except Exception as e:
            print(f"[WARNING] Failed to copy existing_rec: {e}, creating new record")
            rec = dict.fromkeys(results_columns, None)
    else:
        rec = dict.fromkeys(results_columns, None)

    # Safely populate basic fields
    try:
        if dataset == "my_dataset":
            rec["id"] = safe_get_item(row, "id")
            rec["Name"] = safe_get_item(row, "Name")
            rec["category"] = safe_get_item(row, "category") 
            rec["question"] = safe_get_item(row, "question")
        else:
            rec["id"] = safe_get_item(row, "id")
            rec["question"] = safe_get_item(row, "question")
    except Exception as e:
        print(f"[ERROR] Failed to populate basic fields: {e}")
        rec["error"] = f"basic_fields - {type(e).__name__}: {e}"
        return rec

    # Safely get wtnp_thought_process
    wtnp_thought_process = safe_get_item(rec, "wtnp_thought_process")
    if not wtnp_thought_process or wtnp_thought_process in [None, "", "None"]:
        try:
            question_text = safe_get_item(row, "question", "")
            if isinstance(question_text, str):
                question_text = question_text.strip("\n")
            else:
                question_text = str(question_text).strip("\n") if question_text else ""
            
            print(f"[INFO] Executing agent for row {safe_get_item(row, 'id')}")
            result = react_pipeline.execute_agent(input=question_text, prompt_type="")
            
            # Safely unpack the result
            unpacked_result = safe_unpack_result(result, 8, "execute_agent")
            (
                original_TP,
                original_TP_completion_tokens,
                original_TP_total_tokens,
                wtnp_thought_process,
                total_intermediate_steps,
                unique_tool_count,
                unique_tools_used,
                all_tools_list,
                selected_few_shot_questions
            ) = unpacked_result
            
            # Safely assign values with JSON parsing for complex objects
            rec["original_TP"] = safe_json_parse(original_TP) if isinstance(original_TP, str) else (original_TP or None)
            rec["original_TP_completion_tokens"] = original_TP_completion_tokens or None
            rec["original_TP_total_tokens"] = original_TP_total_tokens or None
            rec["wtnp_thought_process"] = safe_json_parse(wtnp_thought_process) if isinstance(wtnp_thought_process, str) else (wtnp_thought_process or None)
            rec["total_intermediate_steps"] = total_intermediate_steps or None
            rec["unique_tool_count"] = unique_tool_count or None
            rec["unique_tools_used"] = safe_json_parse(unique_tools_used) if isinstance(unique_tools_used, str) else (unique_tools_used or None)
            rec["all_tools_list"] = safe_json_parse(all_tools_list) if isinstance(all_tools_list, str) else (all_tools_list or None)
            rec["selected_few_shot_questions"] = selected_few_shot_questions or None
           
        except Exception as e:
            print(f"[ERROR] Error in execute_agent for row {safe_get_item(row, 'id')}: {e}")
            rec["error"] = f"execute_agent - {type(e).__name__}: {e}"
            wtnp_thought_process = safe_get_item(rec, "wtnp_thought_process")
    else:
        print(f"[INFO] Reusing existing wtnp_thought_process for row {safe_get_item(row, 'id')}")

    # Process get_answers if we have valid thought process
    if wtnp_thought_process and wtnp_thought_process not in [None, "", "None"]:
        try:
            question_text = safe_get_item(row, "question", "")
            if isinstance(question_text, str):
                question_text = question_text.strip("\n")
            else:
                question_text = str(question_text).strip("\n") if question_text else ""
            
            print(f"[INFO] Getting answers for row {safe_get_item(row, 'id')}")
            result = react_pipeline.get_answers(wtnp_thought_process, question_text)
            
            # Safely unpack the result (expecting 17 values)
            unpacked_result = safe_unpack_result(result, 17, "get_answers")
            (
                wtp_thought_process,
                wtp_answer,
                wtp_references,
                wtp_referenece_ids,
                wtp_completion_tokens, 
                wtp_total_tokens,
                formatted_TP_tokens,
                clean_TP_tokens,
                drop_tokens_after_preprocessing, 
                wtnp_answer,
                wtnp_references,
                wtnp_referenece_ids,
                wtnp_completion_tokens,
                wtnp_total_tokens,
                wot_answer,
                wot_completion_tokens,
                wot_total_tokens,
            ) = unpacked_result
            
            # Safely log the results
            try:
                print(
                    "wtp_thought_process", bool(wtp_answer), "\n",
                    "wtp_answer", bool(wtp_answer), "\n",
                    "wtp_references", bool(wtp_references), "\n",
                    "wtp_referenece_ids", len(wtp_referenece_ids) if wtp_referenece_ids and hasattr(wtp_referenece_ids, '__len__') else -1, "\n",
                    "wtp_completion_tokens", wtp_completion_tokens or -1, "\n",
                    "wtp_total_tokens", wtp_total_tokens or -1, "\n",
                    "formatted_TP_tokens", formatted_TP_tokens or -1, "\n",
                    "clean_TP_tokens", clean_TP_tokens or -1, "\n",
                    "drop_tokens_after_preprocessing", drop_tokens_after_preprocessing or -1, "\n",
                    "wtnp_answer", bool(wtnp_answer), "\n",
                    "wtnp_references", bool(wtnp_references), "\n",
                    "wtnp_referenece_ids", len(wtnp_referenece_ids) if wtnp_referenece_ids and hasattr(wtnp_referenece_ids, '__len__') else -1, "\n",
                    "wtnp_completion_tokens", wtnp_completion_tokens or -1, "\n",
                    "wtnp_total_tokens", wtnp_total_tokens or -1, "\n",
                    "wot_answer", bool(wot_answer), "\n",
                    "wot_completion_tokens", wot_completion_tokens or -1, "\n",
                    "wot_total_tokens", wot_total_tokens or -1,
                )
            except Exception as log_e:
                print(f"[WARNING] Failed to log results: {log_e}")
            
            # Safely assign values with JSON parsing
            rec["wtp_thought_process"] = safe_json_parse(wtp_thought_process) if isinstance(wtp_thought_process, str) else (wtp_thought_process or None)
            rec["wtp_answer"] = wtp_answer or None
            rec["wtp_references"] = wtp_references or None
            rec["wtp_referenece_ids"] = safe_json_parse(wtp_referenece_ids) if isinstance(wtp_referenece_ids, str) else (wtp_referenece_ids or None)
            
            rec["wtp_completion_tokens"] = wtp_completion_tokens or None
            rec["wtp_total_tokens"] = wtp_total_tokens or None
            rec["formatted_TP_tokens"] = formatted_TP_tokens or None
            rec["clean_TP_tokens"] = clean_TP_tokens or None
            rec["drop_tokens_after_preprocessing"] = drop_tokens_after_preprocessing or None

            rec["wtnp_answer"] = wtnp_answer or None
            rec["wtnp_references"] = wtnp_references or None
            rec["wtnp_referenece_ids"] = safe_json_parse(wtnp_referenece_ids) if isinstance(wtnp_referenece_ids, str) else (wtnp_referenece_ids or None)

            rec["wtnp_completion_tokens"] = wtnp_completion_tokens or None
            rec["wtnp_total_tokens"] = wtnp_total_tokens or None

            rec["wot_answer"] = wot_answer or None
            rec["wot_completion_tokens"] = wot_completion_tokens or None
            rec["wot_total_tokens"] = wot_total_tokens or None
            
        except Exception as e:
            print(f"[ERROR] Error in get_answers for row {safe_get_item(row, 'id')}: {e}")
            existing_error = safe_get_item(rec, "error", "")
            if existing_error and existing_error not in [None, "", "None"]:
                rec["error"] = f"{existing_error} | get_answers - {type(e).__name__}: {e}"
            else:
                rec["error"] = f"get_answers - {type(e).__name__}: {e}"

    return rec


def save_results_to_pickle(results_df, pkl_path):
    """Save the entire results dataframe to pickle file with error handling"""
    try:
        # Create backup of existing file
        if os.path.exists(pkl_path):
            backup_path = pkl_path + ".backup"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(pkl_path, backup_path)
        
        with open(pkl_path, "wb") as out_f:
            pickle.dump(results_df, out_f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Successfully saved results to {pkl_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        # Try to restore backup
        backup_path = pkl_path + ".backup"
        if os.path.exists(backup_path):
            os.rename(backup_path, pkl_path)
            print(f"[INFO] Restored backup file")


def load_results_from_pickle(pkl_path):
    """Load results dataframe from pickle file if it exists with error handling"""
    if not os.path.exists(pkl_path):
        return None
    
    try:
        with open(pkl_path, "rb") as in_f:
            df = pickle.load(in_f)
        print(f"[INFO] Successfully loaded existing results from {pkl_path}")
        return df
    except (pickle.PickleError, EOFError, Exception) as e:
        print(f"[WARNING] Could not load existing results from {pkl_path}: {e}")
        
        # Try backup file
        backup_path = pkl_path + ".backup"
        if os.path.exists(backup_path):
            try:
                with open(backup_path, "rb") as in_f:
                    df = pickle.load(in_f)
                print(f"[INFO] Successfully loaded backup results from {backup_path}")
                return df
            except Exception as backup_e:
                print(f"[WARNING] Could not load backup either: {backup_e}")
        
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM server and wrap it in a ChatOpenAI client"
    )
    
    parser.add_argument(
        "--main_ip", "-ip",
        required=False,
        default="localhost",
        help="IP of the server where the model is hosted"
    )
    
    parser.add_argument(
        "--model-path", "-m",
        required=True,
        help="Local checkpoint directory or HF repo ID (e.g. Qwen/Qwen3-1.7B)"
    )
    
    parser.add_argument(
        "--job-id", "-j",
        required=False,
        default=0,
        help="Job ID for logging"
    )
    
    parser.add_argument(
        "--workspace", "-w",
        default="/data/horse/ws/chja176b-sysbio_cj2025/chja176b-sysbio_cj-1737284518/git_repos/UI_SYSBIO_Sol_Lib",
        help="Path to the workspace directory"
    )
    
    parser.add_argument(
        "--num-few-shot", "-nfs",
        required=False,
        default=3,
        help="Num of few shot examples"
    )

    parser.add_argument(
        "--dataset", "-d",
        required=True,
        default="my_dataset",
        help="Choose between my_dataset or bioasq datasets"
    )
    
    parser.add_argument(
        "--api-key", "-a",
        required=True,
        help="API key to use"
    )
    
    args = parser.parse_args()
    MODEL_NAME = args.model_path
    MAIN_IP = args.main_ip
    JOB_ID = args.job_id
    WORKSPACE = args.workspace
    FEW_SHOT_EXAMPLES = int(args.num_few_shot)
    print("nfs: ", type(FEW_SHOT_EXAMPLES))
    DATASET = args.dataset
    API_KEY = args.api_key

    clean_model_name = re.sub(r"[./]", "", MODEL_NAME)
    
    output_dir = f"{WORKSPACE}/script_logs/python_logs"
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(output_dir, f"logs_{JOB_ID}_{clean_model_name}_{current_time}.log")
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    try:
        if DATASET == "my_dataset":
            file = f"{WORKSPACE}/all_datasets/my_dataset/SB_60Q.csv"
        elif DATASET == "bioasq":
            file = f"{WORKSPACE}/all_datasets/BioASQ-training12b/complete_summary_test_set.csv"
        else:
            raise Exception("The dataset name is invalid. Check again")
        
        df = pd.read_csv(file)
        print(f"[INFO] Loaded dataset with {len(df)} rows")
        
        # Validate DataFrame
        print(f"[INFO] DataFrame columns: {df.columns.tolist()}")
        print(f"[INFO] DataFrame shape: {df.shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    results_columns = [
        "id", "Name", "category", "question", "original_TP",
        "wtp_thought_process", "wtp_answer", "wtp_references", "wtp_referenece_ids",
        "wtp_completion_tokens", "wtp_total_tokens", "total_intermediate_steps",
        "unique_tool_count", "unique_tools_used", "all_tools_list", "selected_few_shot_questions", "original_TP_completion_tokens",
        "original_TP_total_tokens", "formatted_TP_tokens", "clean_TP_tokens",
        "drop_tokens_after_preprocessing", "wtnp_thought_process", "wtnp_answer",
        "wtnp_references", "wtnp_referenece_ids", "wtnp_completion_tokens",
        "wtnp_total_tokens", "wot_answer", "wot_completion_tokens",
        "wot_total_tokens", "error",
    ]

    # Set up the single output file path
    if FEW_SHOT_EXAMPLES > 0:
        pkl_path = f"{WORKSPACE}/all_outputs/{DATASET}/DPP_corrected_few_shot_{FEW_SHOT_EXAMPLES}/{MODEL_NAME.replace('/', '_')}_few_shot_results.pkl"
    else:
        pkl_path = f"{WORKSPACE}/all_outputs/{DATASET}/direct/{MODEL_NAME.replace('/', '_')}_results.pkl"
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    
    # Try to load existing results
    existing_results = load_results_from_pickle(pkl_path)
    if existing_results is not None:
        results_df = existing_results.copy()
        print(f"[INFO] Loaded existing results with {len(results_df)} records")
    else:
        results_df = pd.DataFrame(columns=results_columns)
        # Set column types
        for col in ["unique_tools_used", "all_tools_list", "original_TP", "wtp_thought_process", "wtnp_thought_process", 
                   "wtnp_referenece_ids", "wtp_referenece_ids"]:
            results_df[col] = results_df[col].astype(object)

    # Initialize pipeline with error handling
    try:
        if (MODEL_NAME == "meta-llama/Llama-3.3-70B-Instruct" or MODEL_NAME == "gpt-4.1-2025-04-14"):
            react_pipeline = react.CustomReactPipeline(
                num_few_shot_example=FEW_SHOT_EXAMPLES,
                prompts_style="",
                MODEL_NAME=MODEL_NAME,
                main_ip=MAIN_IP,
                api_key=API_KEY,
                dynamic=False,
                DPP=True
            )
        else:
            raise Exception("Configure model name correctly")
        print(f"[INFO] Successfully initialized pipeline for {MODEL_NAME}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        return

    # Use the full dataset (or filter to specific IDs for testing)
    df_to_process = df.copy()  # Process full dataset
    # df_to_process = df[df["id"].isin([13])]  # Uncomment this line for testing with specific IDs
    
    # Initial processing - process rows that haven't been processed yet
    try:
        processed_ids = set(results_df["id"].dropna().astype(int)) if len(results_df) > 0 else set()
        rows_to_process = df_to_process[~df_to_process["id"].isin(processed_ids)]
        print(f"[INFO] Processing {len(rows_to_process)} new rows...")
    except Exception as e:
        print(f"[WARNING] Error determining processed IDs: {e}, processing all rows")
        rows_to_process = df_to_process
    
    for idx, row in rows_to_process.iterrows():
        try:
            row_id = safe_get_item(row, "id", f"unknown_{idx}")
            print(f"[INFO] Processing row {row_id} (index {idx})")
            
            rec = process_row(row, react_pipeline, results_columns, DATASET, existing_rec=None)
            
            if rec is None:
                print(f"[ERROR] Failed to process row {row_id}, skipping")
                continue
            
            # Add or update the record in results_df
            existing_idx = results_df[results_df["id"] == safe_get_item(row, "id")].index
            if len(existing_idx) > 0:
                # Update existing record
                for col in results_columns:
                    safe_assign_to_dataframe(results_df, existing_idx[0], col, safe_get_item(rec, col))
                print(f"[INFO] Updated existing record for row {row_id}")
            else:
                # Add new record
                new_idx = len(results_df)
                results_df = pd.concat([results_df, pd.DataFrame([rec])], ignore_index=True)
                # Ensure list columns are properly set as object dtype
                for col in ["unique_tools_used", "all_tools_list", "wtp_referenece_ids", "wtnp_referenece_ids", "original_TP", "wtp_thought_process", "wtnp_thought_process"]:
                    if rec and col in rec and rec[col] is not None and isinstance(rec[col], list):
                        safe_assign_to_dataframe(results_df, new_idx, col, rec[col])
                print(f"[INFO] Added new record for row {row_id}")
                    
            # Save after each record
            save_results_to_pickle(results_df, pkl_path)
            print(f"[INFO] Record {row_id} saved to pickle file.")
            time.sleep(10)
            
        except Exception as e:
            print(f"[ERROR] Unexpected error processing row {idx}: {e}")
            continue

    # Retry loop for errors with enhanced error handling
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Check for errors
            error_mask = results_df["error"].notna() & (results_df["error"] != "") & (results_df["error"] != "None")
            error_rows = results_df[error_mask]
            
            if len(error_rows) == 0:
                print("[INFO] No errors found, proceeding to reference check...")
                break
                
            print(f"[INFO] Retry {retry_count + 1}: Found {len(error_rows)} rows with errors, retrying...")
            
            for idx, row in error_rows.iterrows():
                try:
                    row_id = safe_get_item(row, "id", f"unknown_{idx}")
                    print(f"[INFO] Retrying row {row_id} due to error: {safe_get_item(row, 'error')}")
                    
                    # Find original row data from df_to_process
                    original_rows = df_to_process[df_to_process["id"] == safe_get_item(row, "id")]
                    if len(original_rows) == 0:
                        print(f"[WARNING] Could not find original row for {row_id}")
                        continue
                    
                    original_row = original_rows.iloc[0]
                    existing_rec = row.to_dict()
                    rec = process_row(original_row, react_pipeline, results_columns, DATASET, existing_rec=existing_rec)
                    
                    if rec is None:
                        print(f"[ERROR] Failed to retry row {row_id}")
                        continue
                    
                    # Update the record in results_df
                    for col in results_columns:
                        safe_assign_to_dataframe(results_df, idx, col, safe_get_item(rec, col))
                    
                    # Save after each retry
                    save_results_to_pickle(results_df, pkl_path)
                    print(f"[INFO] Retry for record {row_id} completed and saved.")
                    time.sleep(10)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to retry row {idx}: {e}")
                    continue
            
            retry_count += 1
            
        except Exception as e:
            print(f"[ERROR] Error in retry loop: {e}")
            break

    if retry_count >= max_retries:
        print(f"[WARNING] Max retries ({max_retries}) reached. Some errors may still exist.")

    # Check for missing references and reprocess with enhanced error handling
    reference_retry_count = 0
    max_reference_retries = 3
    
    while reference_retry_count < max_reference_retries:
        try:
            # Check for missing wtp_references (empty, None, or NaN)
            missing_ref_mask = (
                results_df["wtp_references"].isna() | 
                (results_df["wtp_references"] == "") | 
                (results_df["wtp_references"] == "None") |
                (results_df["wtp_references"].astype(str) == "nan")
            ) & (
                # Only check rows that don't have errors
                results_df["error"].isna() | 
                (results_df["error"] == "") | 
                (results_df["error"] == "None")
            )
            
            missing_ref_rows = results_df[missing_ref_mask]
            
            if len(missing_ref_rows) == 0:
                print("[INFO] All rows have references or errors. Processing complete!")
                break
                
            print(f"[INFO] Reference retry {reference_retry_count + 1}: Found {len(missing_ref_rows)} rows with missing references, retrying...")
            
            for idx, row in missing_ref_rows.iterrows():
                try:
                    row_id = safe_get_item(row, "id", f"unknown_{idx}")
                    print(f"[INFO] Retrying row {row_id} due to missing references")
                    
                    # Find original row data from df_to_process
                    original_rows = df_to_process[df_to_process["id"] == safe_get_item(row, "id")]
                    if len(original_rows) == 0:
                        print(f"[WARNING] Could not find original row for {row_id}")
                        continue
                    
                    original_row = original_rows.iloc[0]
                    existing_rec = row.to_dict()
                    rec = process_row(original_row, react_pipeline, results_columns, DATASET, existing_rec=existing_rec)
                    
                    if rec is None:
                        print(f"[ERROR] Failed to retry row {row_id}")
                        continue
                    
                    # Update the record in results_df
                    for col in results_columns:
                        safe_assign_to_dataframe(results_df, idx, col, safe_get_item(rec, col))
                    
                    # Save after each retry
                    save_results_to_pickle(results_df, pkl_path)
                    print(f"[INFO] Reference retry for record {row_id} completed and saved.")
                    time.sleep(10)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to retry row {idx}: {e}")
                    continue
            
            reference_retry_count += 1
            
        except Exception as e:
            print(f"[ERROR] Error in reference retry loop: {e}")
            break

    if reference_retry_count >= max_reference_retries:
        print(f"[WARNING] Max reference retries ({max_reference_retries}) reached. Some rows may still have missing references.")

    # Final save
    save_results_to_pickle(results_df, pkl_path)
    print(f"[INFO] Final results saved to {pkl_path}")
    
    # Print summary with error handling
    try:
        total_rows = len(results_df)
        error_rows = len(results_df[results_df["error"].notna() & (results_df["error"] != "") & (results_df["error"] != "None")])
        missing_ref_rows = len(results_df[
            (results_df["wtp_references"].isna() | (results_df["wtp_references"] == "") | (results_df["wtp_references"] == "None")) &
            (results_df["error"].isna() | (results_df["error"] == "") | (results_df["error"] == "None"))
        ])
        
        success_rate = ((total_rows - error_rows - missing_ref_rows) / total_rows * 100) if total_rows > 0 else 0
        
        print(f"""
        Processing Summary:
        - Total rows processed: {total_rows}
        - Rows with errors: {error_rows}
        - Rows with missing references (no errors): {missing_ref_rows}
        - Success rate: {success_rate:.1f}%
        """)
        
        # Additional debugging info
        if error_rows > 0:
            print("\nError breakdown:")
            error_types = results_df[results_df["error"].notna() & (results_df["error"] != "") & (results_df["error"] != "None")]["error"].value_counts()
            for error_type, count in error_types.head(10).items():
                print(f"  - {error_type}: {count} occurrences")
                
    except Exception as e:
        print(f"[ERROR] Failed to generate summary: {e}")


if __name__ == "__main__":
    main()