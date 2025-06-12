import os, torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
print(f"[DEBUG] Hello from RANK={os.environ.get('RANK')} | LOCAL_RANK={os.environ.get('LOCAL_RANK')} | CUDA: {torch.cuda.current_device()}")
print("device", torch.cuda.current_device())
print("device name", torch.cuda.get_device_name(0))

####################################################################################################################################
import os  
import re 
from reward import total_reward
from datetime import datetime  
from dataclasses import dataclass, field  
from typing import Optional  
from datasets import load_dataset, load_from_disk  # HuggingFace datasets 관련 함수
from datasets import Dataset, DatasetDict  
from transformers import Qwen2VLForConditionalGeneration  
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified  
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config  
####################################################################################################################################

# GRPO 스크립트 실행용 인자 정의
@dataclass # GRPOScript를 dataclass(GRPOScript)의 인자로 넣겠다는 말. #즉, GRPOScript = dataclass(GRPOScript)
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    # reward function들의 리스트를 지정합니다. 기본값은 ["accuracy", "format"]입니다.
    # field는 dataclass의 필드속성을 세밀하게 조정 -> 각 instance마다 새로운 리스트를 만들어서 버그 예방.
    reward_funcs: list[str] = field(
        default_factory=lambda: ["total"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    # 이미지의 최대 픽셀 수를 지정합니다.
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    # 이미지의 최소 픽셀 수를 지정합니다.
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    # temporal GRPO 사용 여부를 지정합니다.
    temporal: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using temporal GRPO"},
    )
    # length reward 사용 여부를 지정합니다.
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )



reward_funcs_registry = {
    # "accuracy": accuracy_reward,
    # "format": format_reward,
    "total": total_reward
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
import torch.distributed as dist

def init_distributed_if_needed():
    if not dist.is_initialized() and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        print(f"[init] Rank={dist.get_rank()}, WorldSize={dist.get_world_size()}")


def main(script_args, training_args, model_args):
    init_distributed_if_needed()
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]


    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }]
            }
        
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("trainer_cls using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    # TrlParser는 인자를 받아서 GRPOScriptArguments, GRPOConfig, ModelConfig라는 @dataclass들에 매핑할 준비를 하는 애,# 즉 어떤 인자들이 어떤 클래스로 가야 할지를 결정하는 형틀 -> 즉, 분배규칙표를 만듦
    
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    # GRPOScriptArguments:→ “데이터를 어떻게 준비할지, 어떤 보상 방식으로 실행할지 등 GRPO-specific 설정들” -> 스크립트 실행시 사용하는 커스텀 설정중심.,# GRPOConfig:→ “학습을 어떻게 할 건지 (학습률, KL, 보상 가중치, generation 관련 설정 등)” -> 학습세부설정중심
    
    print("파서 생성 완료, 인자 파싱 시작")
    # 실제로 인자를 받아서 나눠주는 실행 함수 -> 분배작업을 실행해 실제 각 dataclass에 채워 넣는 함수.
    script_args, training_args, model_args = parser.parse_args_and_config()
    print("📌script_args:", script_args)
    print("📌training_args:", training_args)
    print("📌model_args:", model_args)
    print("📌실행시간: ", datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
    main(script_args, training_args, model_args)
