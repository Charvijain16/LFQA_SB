import time
import os
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import pandas as pd
import tiktoken
import re
from typing import Dict, List, Tuple
import pickle


def initialize_judge(judge_name):
    if judge_name == "openai/gpt-oss-120b" or judge_name == "llama-3-3-70B-instruct":
        llm = ChatOpenAI(
            model_name=judge_name,
            base_url="https://llm.scads.ai/v1",
            openai_api_key=os.environ.get("SCADS_API_KEY"),
            temperature=0,
            timeout=300,
            extra_body={"disable_fallbacks": True}
        )
    elif "Qwen" in judge_name:
        llm = ChatOpenAI(
            model_name=judge_name,
            base_url="https://llm.scads.ai/v1",
            openai_api_key=os.environ.get("SCADS_API_KEY"),
            temperature=0,
            timeout=300,
            extra_body={"disable_fallbacks": True}
        )  
    elif "gpt" in judge_name:
        llm = ChatOpenAI(model=judge_name, api_key=os.environ.get("OPENAI_API_KEY"))
    elif "claude" in judge_name:
        llm = ChatAnthropic(model=judge_name, temperature=0, api_key=os.environ.get("ANTHROPIC_API_KEY"))
    elif "gemini" in judge_name:
        llm = ChatGoogleGenerativeAI(
            model=judge_name,#"gemini-2.5-pro",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0,
            timeout=300,
        )
    else:
        llm = ChatOpenAI(
            model_name=judge_name,
            base_url="http://localhost:6000/v1",
            api_key="EMPTY")
        
    return llm


def parse_score(output: str, num_answers: int) -> List[int]:
    """
    Parse a string containing answer scores and return a list of scores.
    Handles both 2-answer and 3-answer cases dynamically.
    
    Args:
        output (str): String containing answer scores
        num_answers (int): Number of answers to expect (2 or 3)
    
    Returns:
        List[int]: List of scores in order [A, B] or [A, B, C]
    """
    result_dict = {}
    expected_letters = ['A', 'B', 'C'][:num_answers]
    
    # Try regex pattern first (handles embedded text)
    pattern = r'Answer ([A-C]):\s*(\d+)'
    matches = re.findall(pattern, output)
    
    if matches:
        # Convert to dictionary and ensure we have the expected letters
        for answer_letter, score in matches:
            if answer_letter in expected_letters:
                result_dict[answer_letter] = int(score)
        
        # Return scores in A, B, (C) order
        scores = []
        for letter in expected_letters:
            if letter in result_dict:
                scores.append(result_dict[letter])
            else:
                print(f"Warning: Missing score for Answer {letter}")
                return []  # Return empty if any score is missing
        
        return scores
    else:
        # Fallback parsing
        try:
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            for line in lines:
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        key_part = parts[0].strip()
                        value_part = parts[1].strip()
                        
                        # Extract answer letter
                        answer_match = re.search(r'Answer ([A-C])', key_part)
                        if answer_match and answer_match.group(1) in expected_letters and value_part.isdigit():
                            result_dict[answer_match.group(1)] = int(value_part)
            
            # Return in A, B, (C) order
            if len(result_dict) == num_answers:
                return [result_dict.get(letter, 0) for letter in expected_letters]
                
        except Exception as e:
            print(f"Error in fallback parsing: {e}")
    
    return []


def create_randomized_prompt(question: str, answers: Dict[str, str], criterion_prompt: str, 
                           seed: int = None) -> Tuple[str, Dict[str, str]]:
    """
    Create a prompt with randomized answer order to reduce position bias.
    Handles both 2-answer and 3-answer cases dynamically.
    
    Args:
        question: The question being evaluated
        answers: Dict mapping answer types to their content (filters out empty answers)
        criterion_prompt: The evaluation criterion prompt
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (formatted_prompt, mapping_dict) where mapping_dict shows which 
        answer type got assigned to which position (A, B, C)
    """
    if seed is not None:
        random.seed(seed)
    
    # Filter out empty or None answers
    valid_answers = {k: v for k, v in answers.items() if v and v.strip()}
    
    if len(valid_answers) < 2:
        raise ValueError(f"Need at least 2 valid answers, got {len(valid_answers)}")
    
    # Create list of answer items and shuffle
    answer_items = list(valid_answers.items())
    random.shuffle(answer_items)
    
    # Create mapping from position to answer type
    position_to_type = {}
    positions = ['A', 'B', 'C']
    num_answers = len(answer_items)
    
    prompt_parts = [criterion_prompt, "", f"# Question:", question, "", ""]
    
    for i, (answer_type, answer_content) in enumerate(answer_items):
        position = positions[i]
        position_to_type[position] = answer_type
        
        prompt_parts.extend([
            f"# Answer {position}:",
            answer_content or "[No answer provided]",
            "",
            "-" * 50,
            ""
        ])
    
    # Dynamic evaluation instructions based on number of answers
    prompt_parts.append("# Evaluation Instructions:")
    prompt_parts.append("Please evaluate each answer using the criteria above and provide scores in this exact format:")
    
    for i in range(num_answers):
        prompt_parts.append(f"Answer {positions[i]}: [score 1-5]")
    
    prompt_parts.extend([
        "",
        "Provide only the scores, no additional explanation."
    ])
    
    formatted_prompt = "\n".join(prompt_parts)
    
    # Return reverse mapping (position -> answer_type)
    return formatted_prompt, position_to_type


def get_score_with_mapping(llm, content: str, mapping: Dict[str, str], retry: int) -> Dict[str, int]:
    """
    Get scores from LLM and map them back to answer types.
    Handles both 2-answer and 3-answer cases dynamically.
    
    Args:
        llm: Language model instance
        content: Prompt content
        mapping: Dict mapping positions (A, B, C) to answer types
        retry: Number of retries remaining
        
    Returns:
        Dict mapping answer types to their scores
    """
    num_answers = len(mapping)
    expected_positions = ['A', 'B', 'C'][:num_answers]
    
    try:
        responses = llm.invoke(content)
        position_scores = parse_score(responses.content, num_answers)
        
        print(f"Raw LLM response: {responses.content}")
        print(f"Parsed position scores: {position_scores}")
        print(f"Position mapping: {mapping}")
        
        if len(position_scores) == num_answers:
            # Map position scores back to answer types
            result = {}
            for i, score in enumerate(position_scores):
                position = expected_positions[i]
                answer_type = mapping[position]
                result[answer_type] = score
            
            print(f"Final mapped scores: {result}")
            return result
            
        elif retry > 0:
            print(f"Incomplete scores received. Retrying... ({retry} attempts left)")
            
            # Create retry instructions based on number of answers
            format_lines = [f"Answer {pos}: [number]" for pos in expected_positions]
            format_instruction = "\n".join(format_lines)
            
            retry_content = f"""The previous response was incomplete. Please provide exactly {num_answers} scores.

{content}

IMPORTANT: Respond with exactly this format:
{format_instruction}"""
            
            return get_score_with_mapping(llm, retry_content, mapping, retry - 1)
        else:
            print("No more retries left, returning default scores")
            return {answer_type: -10 for answer_type in mapping.values()}
            
    except Exception as e:
        print(f"Error in get_score_with_mapping: {e}")
        if retry > 0:
            print(f"Exception occurred, retrying... ({retry} attempts left)")
            time.sleep(2)
            return get_score_with_mapping(llm, content, mapping, retry - 1)
        else:
            print("No more retries left after exception")
            return {answer_type: -10 for answer_type in mapping.values()}


def criterion_prompts(criterion: list, use_comparative_instruction: bool = True, 
                      use_anchoring: bool = True, use_domain_specific: bool = True) -> dict:
    """
    Load criterion prompts from files and optionally add additional instructions.
    
    Args:
        criterion: List of criteria to evaluate
        use_comparative_instruction: Add comparative evaluation guidance
        use_anchoring: Add reference anchoring examples
        use_domain_specific: Add domain-specific instructions for biomedical content
    """
    
    # Base answers data prompt (same as before)
    answers_data_prompt = """
    # Question:
    {{question}}

    -----------

    # Answer A:
    {{answer_A}}

    -----------

    # Answer B:
    {{answer_B}}

    -----------

    # Answer C:
    {{answer_C}}

    -----------
    """
    
    # Additional instruction components
    comparative_instruction = """
    
    # Additional Evaluation Guidelines:
    
    Before scoring, consider:
    - How does this answer compare to what an expert would provide?
    - What would a perfect answer look like for this specific question?
    - Are you being appropriately critical rather than generous?
    - Does this answer meet the standards of high-quality academic or professional work?
    
    Remember: Average answers should receive average scores (2-3), not high scores.
    Be demanding in your evaluation - most answers have room for improvement.
    """
    
    anchoring_examples = """
    
    # Scoring Reference Points:
    
    For reference when scoring:
    - Score 5: Answer quality suitable for publication in a top-tier journal or expert-level response
    - Score 4: High-quality answer with minor flaws, suitable for professional use  
    - Score 3: Acceptable answer for undergraduate assignment, but with noticeable limitations
    - Score 2: Basic answer that addresses the question but needs significant improvement
    - Score 1: Poor answer that requires major revision before being useful
    
    Most answers fall in the 2-3 range. Scores of 4-5 should be reserved for truly exceptional responses.
    """
    
    domain_specific = """
    
    # Domain-Specific Considerations (Biomedical/Scientific Content):
    
    For scientific and biomedical content, also evaluate:
    - Scientific accuracy and precision of statements
    - Appropriate use of technical terminology
    - Logical flow of biological/medical concepts
    - Consideration of relevant mechanisms, pathways, or processes
    - Clinical relevance and implications where appropriate
    - Proper contextualization within the broader scientific literature
    
    Be especially critical of oversimplifications or inaccuracies in scientific content.
    """
    
    # Load base criterion prompts from files
    prompts_dict = {}
    for item in criterion:
        try:
            with open(f"src/evaluation/eval_prompts_dec16/{item}.txt", "r") as f:
                base_prompt = f.read().strip()
                prompts_dict[item] = base_prompt
        except FileNotFoundError:
            print(f"Warning: Prompt file for {item} not found")
            prompts_dict[item] = f"Evaluate the {item} of the answers on a scale of 1-5."
    
    # Build enhanced prompts
    enhanced_prompts = {}
    for prompt_name, base_prompt in prompts_dict.items():
        
        # Start with base prompt
        full_prompt = base_prompt
        
        # Add additional instructions based on parameters
        if use_comparative_instruction:
            full_prompt += comparative_instruction
            
        if use_anchoring:
            full_prompt += anchoring_examples
            
        if use_domain_specific:
            full_prompt += domain_specific
        
        # Add the answers data section
        full_prompt += answers_data_prompt
        
        # Add final evaluation instructions
        full_prompt += """
    # Final Instructions:
    
    Evaluate each answer using ALL the criteria above and provide scores in this exact format:
    Answer A: [score 1-5]
    Answer B: [score 1-5]
    Answer C: [score 1-5]
    
    Be strict and critical in your evaluation. Provide only the numerical scores, no additional explanation.
    """
        
        enhanced_prompts[prompt_name] = full_prompt
    
    return enhanced_prompts


# Alternative: Add as separate configuration
class EvaluationConfig:
    """Configuration class for evaluation parameters"""
    
    def __init__(self):
        self.use_strict_scoring = True
        self.use_comparative_instruction = True
        self.use_anchoring = False
        self.use_domain_specific = True
        self.expected_score_distribution = "Most answers should score 2-3"
        

def get_enhanced_criterion_prompts(criterion: list, config: EvaluationConfig = None) -> dict:
    """
    Alternative function that uses configuration object for more flexibility
    """
    if config is None:
        config = EvaluationConfig()
    
    return criterion_prompts(
        criterion=criterion,
        use_comparative_instruction=config.use_comparative_instruction,
        use_anchoring=config.use_anchoring,
        use_domain_specific=config.use_domain_specific
    )


def load_existing_scores(cache_file_path: str) -> Dict:
    """
    Load existing scores from cache file.
    
    Args:
        cache_file_path: Path to the cache file
        
    Returns:
        Dict with existing scores or empty dict if file doesn't exist
    """
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, FileNotFoundError):
            print(f"Could not load cache from {cache_file_path}, starting fresh")
            return {}
    return {}


def save_scores_to_cache(cache_file_path: str, scores_cache: Dict):
    """
    Save scores to cache file.
    
    Args:
        cache_file_path: Path to save the cache file
        scores_cache: Dictionary containing all scores
    """
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    with open(cache_file_path, 'wb') as f:
        pickle.dump(scores_cache, f)


def generate_cache_key(row_id: str, answers: Dict[str, str], judges: List[str], criteria: List[str]) -> str:
    """
    Generate a unique cache key based on the evaluation parameters.
    
    Args:
        row_id: ID of the row being evaluated
        answers: Dictionary of answers
        judges: List of judge models
        criteria: List of evaluation criteria
        
    Returns:
        Unique cache key string
    """
    # Sort to ensure consistent ordering
    sorted_answer_keys = sorted(answers.keys())
    sorted_judges = sorted(judges)
    sorted_criteria = sorted(criteria)
    
    # Create a hash of the content to detect changes
    import hashlib
    content_hash = hashlib.md5(
        str(sorted([answers[k] for k in sorted_answer_keys])).encode()
    ).hexdigest()[:8]
    
    cache_key = f"{row_id}_{'-'.join(sorted_answer_keys)}_{'-'.join(sorted_judges)}_{'-'.join(sorted_criteria)}_{content_hash}"
    return cache_key


def metric_llm_judgement(row: pd.Series, llm_judges: List[str], criteria: List[str], 
                        cache_file_path: str = None) -> dict:
    """
    Improved LLM judgement metric with enhanced prompts and smart caching.
    Now handles both 2-answer and 3-answer cases dynamically and avoids recomputation.
    
    Args:
        row: Pandas Series containing the data row
        llm_judges: List of judge model names
        criteria: List of evaluation criteria
        cache_file_path: Optional path to cache file for storing/loading results
        
    Returns:
        Dictionary of scores
    """
    final_scores = {}
    
    # Load existing scores from cache if provided
    scores_cache = {}
    if cache_file_path:
        scores_cache = load_existing_scores(cache_file_path)
    
    # Option 1: Use the enhanced prompts with all additional instructions
    prompts_dict = criterion_prompts(
        criteria, 
        use_comparative_instruction=True,
        use_anchoring=False, 
        use_domain_specific=True  # Set to False if not evaluating scientific content
    )
    
    # Prepare answers - filter out empty ones
    all_possible_answers = {
        'wot': row.get("wot_answer", ""),
        'wtnp': row.get("wtnp_answer", ""), 
        'wtp': row.get("wtp_answer", "")
    }
    
    # Filter to only include non-empty answers
    answers = {k: v for k, v in all_possible_answers.items() if v and str(v).strip()}
    
    # Validate that we have at least 2 answers
    if len(answers) < 2:
        print(f"Warning: Need at least 2 answers, found {len(answers)} for row {row.get('id', 'unknown')}")
        return {f"{answer_type}_{judge}_{criterion}": -10 
                for answer_type in all_possible_answers.keys() 
                for judge in llm_judges 
                for criterion in criteria}
    
    print(f"Processing {len(answers)} answers: {list(answers.keys())}")
    
    question = row.get('question_gt', row.get('question', ''))
    if not question:
        print(f"Warning: No question found for row {row.get('id', 'unknown')}")
    
    # Generate cache key for this evaluation
    row_id = str(row.get('id', 'unknown'))
    cache_key = generate_cache_key(row_id, answers, llm_judges, criteria)
    
    # Check if we already have results for this exact configuration
    if cache_key in scores_cache:
        print(f"Loading cached results for row {row_id}")
        cached_scores = scores_cache[cache_key]
        
        # Ensure we have scores for all expected combinations
        expected_keys = {f"{answer_type}_{judge}_{criterion}" 
                        for answer_type in all_possible_answers.keys() 
                        for judge in llm_judges 
                        for criterion in criteria}
        
        if expected_keys.issubset(set(cached_scores.keys())):
            return cached_scores
        else:
            print(f"Cached results incomplete for row {row_id}, recomputing...")
    
    # Use row ID as seed for consistent randomization
    base_seed = hash(str(row.get('id', 0))) % 10000
        
    for judge_idx, judge in enumerate(llm_judges):
        print(f"Initializing judge: {judge}")
        try:
            active_judge = initialize_judge(judge)
            print(f"Successfully initialized judge: {judge}")
        except Exception as e:
            print(f"Failed to initialize judge {judge}: {e}")
            for criterion in criteria:
                # Set scores for all possible answer types, not just valid ones
                for answer_type in all_possible_answers.keys():
                    final_scores[f"{answer_type}_{judge}_{criterion}"] = -10
            continue
        
        for crit_idx, criterion in enumerate(criteria):
            print(f"Evaluating criterion: {criterion}")
            
            if criterion not in prompts_dict:
                print(f"Warning: No prompt found for criterion {criterion}")
                continue
            
            # Create unique seed for this judge-criterion combination
            seed = base_seed + judge_idx * 1000 + crit_idx * 100
            
            try:
                # Create randomized prompt using the enhanced prompt
                prompt, position_mapping = create_randomized_prompt(
                    question=question,
                    answers=answers,  # Only pass valid answers
                    criterion_prompt=prompts_dict[criterion],
                    seed=seed
                )
                
                print(f"Position mapping for row {row.get('id')}: {position_mapping}")
                
                # Get scores with proper mapping
                mapped_scores = get_score_with_mapping(
                    llm=active_judge,
                    content=prompt,
                    mapping=position_mapping,
                    retry=3
                )
                
                # Store scores in final results for evaluated answers
                for answer_type, score in mapped_scores.items():
                    final_scores[f"{answer_type}_{judge}_{criterion}"] = score
                
                # Set default score for non-evaluated answers (empty/missing ones)
                for answer_type in all_possible_answers.keys():
                    score_key = f"{answer_type}_{judge}_{criterion}"
                    if score_key not in final_scores:
                        final_scores[score_key] = -10
                
                print(f"Completed evaluation for {judge}-{criterion}")
                
            except Exception as e:
                print(f"Error evaluating {judge}-{criterion}: {e}")
                for answer_type in all_possible_answers.keys():
                    final_scores[f"{answer_type}_{judge}_{criterion}"] = -10
    
    # Cache the results if cache file path is provided
    if cache_file_path:
        scores_cache[cache_key] = final_scores
        save_scores_to_cache(cache_file_path, scores_cache)
        print(f"Saved results to cache for row {row_id}")
    
    return final_scores