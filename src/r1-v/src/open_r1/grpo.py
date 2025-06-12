# 표준 라이브러리 import
import os  
import re  
from datetime import datetime  
from dataclasses import dataclass, field  
from typing import Optional  

# 외부 라이브러리 import
from datasets import load_dataset, load_from_disk  # HuggingFace datasets 관련 함수
from transformers import Qwen2VLForConditionalGeneration  # Qwen2VL 모델 import

# 내부 trainer 모듈 import -> 폴더에 있는거임. 
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified  # GRPO 트레이너 클래스

# TRL 라이브러리 import (HuggingFace RL 트레이닝 툴킷)
# trl은 Transformer reinforcement Leaning으로 huggingface에서 제공하는 강화학습 프레임워크
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config  

# 데이터셋 타입 import
from datasets import Dataset, DatasetDict  

# 평가 지표 관련 라이브러리 import
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # BLEU 점수 계산
from rouge_score import rouge_scorer  # ROUGE 점수 계산

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
        default_factory=lambda: ["accuracy", "format"],
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
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    # length reward 사용 여부를 지정합니다.
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )



def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        # <answer>와 </answer> 태그 사이의 내용을 추출하기 위한 정규표현식 패턴을 정의합니다.
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        # re.search는 text에서 pattern과 일치하는 부분을 찾아서 match 객체를 반환합니다.
        # re.DOTALL(Dot matches All_) 옵션은 줄바꿈 문자(\n)가 있어도 .이 모든 문자와 매치되도록 합니다.
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # match.group(1)은 첫 번째 괄호(캡처 그룹)에 해당하는 부분(즉, <answer>와 </answer> 사이의 실제 답변 텍스트)을 반환합니다.
            # strip()은 앞뒤 공백을 제거합니다.
            return match.group(1).strip()
        # 만약 패턴이 일치하지 않으면 빈 문자열을 반환합니다.
        return ""

    # 문자열로 표현된 숫자를 실수(float)로 변환하는 함수.
    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    # wer 함수는 Word Error Rate(단어 오류율)를 계산하는 함수입니다.
    # reference(정답 문장)와 hypothesis(예측 문장)를 받아서,
    # 두 문장 간의 최소 편집 거리를 단어 단위로 계산한 뒤,
    # 정답 문장의 단어 수로 나누어 오류율을 반환합니다.
    # 이 값은 0에 가까울수록 예측이 정답과 유사함을 의미합니다.
    def wer(reference, hypothesis):  # Word Error Rate(단어 오류율)를 계산하는 함수 정의
        ref_words = reference.split()  # reference(정답 문장)를 단어 단위로 분할
        hyp_words = hypothesis.split()  # hypothesis(예측 문장)를 단어 단위로 분할
        m = len(ref_words)  # 정답 문장의 단어 수
        n = len(hyp_words)  # 예측 문장의 단어 수
        d = [[0]*(n+1) for _ in range(m+1)]  # 편집 거리 계산을 위한 2차원 배열 초기화
        for i in range(m+1):  # 첫 번째 열 초기화 (삽입 연산)
            d[i][0] = i
        for j in range(n+1):  # 첫 번째 행 초기화 (삭제 연산)
            d[0][j] = j
        for i in range(1, m+1):  # 동적 프로그래밍을 이용한 편집 거리 계산
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:  # 단어가 같으면 추가 비용 없음
                    d[i][j] = d[i-1][j-1]
                else:  # 단어가 다르면 삽입, 삭제, 교체 중 최소 비용 선택
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)  # 편집 거리를 정답 단어 수로 나누어 오류율 반환 (0으로 나누는 것 방지)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        """
        use_stemmer는 rouge 점수를 계산할 때 어간 추출(stemming)을 사용할지 여부를 결정하는 인자입니다.
        True로 설정하면 단어의 어간만 비교하여 좀 더 유연하게 유사도를 평가하고,
        False로 설정하면 단어의 원형 그대로 비교합니다.
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    # accuracy_reward 함수의 주요 동작을 설명합니다.
    # 이 함수는 문제 유형에 따라 모델의 출력(completions)과 정답(solution)을 비교하여 보상을 계산합니다.
    # 각 문제 유형별로 보상 계산 방식이 다릅니다.

    # 1. 문제 유형을 받아옵니다.
    question_type = kwargs['problem_type'][0]
    print("question_type: ", question_type)
    
    # 2. completions에서 실제 답변 텍스트만 추출합니다.
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    # 3. 각 답변(content)과 정답(sol)에 대해 반복하며 보상을 계산합니다.
    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)  # 모델이 생성한 답변에서 <answer> 태그 안의 값을 추출
            gt_ans = extract_answer(sol)         # 정답에서 <answer> 태그 안의 값을 추출
            print("gt_ans: ", gt_ans)
            print("\noutput_ans: ", output_ans)

            # 문제 유형별로 보상 계산
            if question_type == "multiple choice":
                # 객관식: 정답과 예측이 정확히 일치하면 1.0, 아니면 0.0
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0

            elif question_type == "numerical":
                # 숫자형: 소수점(혹은 콤마) 포함 여부가 다르면 0점
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    # 변환 실패 시 0점, 아니면 소수 둘째자리까지 반올림 후 일치하면 1점
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0

            elif question_type == "OCR":
                # OCR: 정답과 예측의 단어 오류율(Word Error Rate, WER)을 계산하여 1에서 빼서 보상으로 사용
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))  # 0~1 사이로 보정

            elif question_type == "free-form":
                # 자유서술형: rouge 점수(정답과 예측의 유사도)를 계산하여 0~1 사이로 보상
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))

            elif question_type == "regression":
                # 회귀: 정답과 예측의 상대 오차(relative difference)를 계산하여 1에서 빼서 보상
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff

            else:
                # 정의되지 않은 문제 유형은 0점
                reward = 0.0

        except Exception as e:
            # 예외 발생 시 0점, 에러 메시지 출력
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        # DEBUG_MODE가 true면 로그 파일에 결과를 기록
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    # 각 샘플별로 계산된 보상 리스트를 반환
    return rewards


def format_reward(completions, **kwargs):
    """
    이 함수는 주어진 completions(모델의 출력) 리스트에서 각 completion이 특정 포맷을 따르는지 확인하여 보상을 부여하는 함수입니다.

    - 포맷 기준: <think> ... </think> 와 <answer> ... </answer> 태그가 각각 한 번씩 등장하고, 그 사이에 어떤 내용이든 올 수 있어야 합니다.
    - 각 completion의 첫 번째 turn(대화의 첫 메시지)의 "content" 필드를 추출합니다.
    - 정규표현식(re.fullmatch)을 사용해 위 포맷과 정확히 일치하는지 검사합니다.
    - 일치하면 1.0, 아니면 0.0의 보상 점수를 반환합니다.
    - 반환값은 completions의 각 항목에 대해 계산된 보상 점수의 리스트입니다.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs: ", reward_funcs)

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
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
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
    # TrlParser는 인자를 받아서 GRPOScriptArguments, GRPOConfig, ModelConfig라는 @dataclass들에 매핑할 준비를 하는 애
    # 즉 어떤 인자들이 어떤 클래스로 가야 할지를 결정하는 형틀 -> 즉, 분배규칙표를 만듦
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    # GRPOScriptArguments:→ “데이터를 어떻게 준비할지, 어떤 보상 방식으로 실행할지 등 GRPO-specific 설정들” -> 스크립트 실행시 사용하는 커스텀 설정중심.
    # GRPOConfig:→ “학습을 어떻게 할 건지 (학습률, KL, 보상 가중치, generation 관련 설정 등)” -> 학습세부설정중심


    # 실제로 인자를 받아서 나눠주는 실행 함수 -> 분배작업을 실행해 실제 각 dataclass에 채워 넣는 함수.
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
