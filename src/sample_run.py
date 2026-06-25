import json
import argparse
from src.agents import react

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
        default="",
        help="Path to the workspace directory"
    )
    
    parser.add_argument(
        "--num-few-shot", "-nfs",
        required=False,
        default=0,
        help="Num of few shot examples"
    )

    parser.add_argument(
        "--dataset", "-d",
        required=False,
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
    # JOB_ID = args.job_id
    # WORKSPACE = args.workspace
    FEW_SHOT_EXAMPLES = int(args.num_few_shot)
    # print("nfs: ", type(FEW_SHOT_EXAMPLES))
    DATASET = args.dataset
    API_KEY = args.api_key


    question="What is the role of Myosin-X in spindle morphogenesis in the mouse oocyte?"

    react_pipeline = react.CustomReactPipeline(
                num_few_shot_example=FEW_SHOT_EXAMPLES,
                prompts_style="",
                MODEL_NAME=MODEL_NAME,
                main_ip=MAIN_IP,
                api_key=API_KEY,
                dynamic=False,
                DPP=False
            )

    thought_process, answer, references = react_pipeline.execute_agent(input=question, prompt_type="")

    print("#########################################################################################")
    print("Question: ", question)
    print("#########################################################################################")
    print("Reasoning: ", json.dumps(thought_process, indent=4))
    print("#########################################################################################")
    print("Final Report: ", answer)
    print("#########################################################################################")
    print("References: ", references)
    print("#########################################################################################")


if __name__ == "__main__":
    main()