# -----------------------------------------------------------------------------
#  ██████╗ ██████╗  ██████╗  ██████╗          HuggingFace GRPO Trainer
#  Author : HF Team (Apache‑2.0)
# -----------------------------------------------------------------------------
# This file implements GRPO (Group Relative Policy Optimization) for Qwen‑2‑VL models.
# - Loads multi-modal data (images/videos + text prompts)
# - Uses reference model for KL regularization
# - Applies custom reward functions (accuracy, format, etc.)
# - Optimizes via RL loss combining rewards and per-token KL
# -----------------------------------------------------------------------------

# -------------------- 표준 라이브러리 import --------------------
import os  
import textwrap  
from collections import defaultdict  
from typing import Any, Callable, Optional, Union  
import random  

# -------------------- 파이토치 및 트랜스포머 관련 import --------------------
import torch 
import torch.utils.data  
import transformers  

# -------------------- 데이터셋 관련 import --------------------
from datasets import Dataset, IterableDataset  
from packaging import version  

# -------------------- 트랜스포머 모델 및 유틸 import --------------------
from transformers import (
    AriaForConditionalGeneration,  # 
    AriaProcessor,  
    AutoModelForCausalLM,  # 사전학습된 causal language model(GPT류)을 config에 따라 자동으로 로드하는 클래스.
    AutoModelForSequenceClassification,  
    AutoProcessor,  # 이미지+텍스트 등 다양한 입력 포맷을 전처리할 수 있도록 자동 구성되는 통합 processor (예: CLIPProcessor).
    AutoTokenizer,  # 모델 config에 맞춰 알맞은 tokenizer(BPE, WordPiece 등)를 자동 로딩하는 클래스.
    GenerationConfig, 
    PreTrainedModel,  
    PreTrainedTokenizerBase,  # 사전학습 토크나이저 베이스
    Qwen2VLForConditionalGeneration,  # Qwen2-VL 모델
    Qwen2_5_VLForConditionalGeneration,  # Qwen2.5-VL 모델
    Trainer,  # 트레이너 베이스
    TrainerCallback,  # 콜백
    is_wandb_available,  # wandb 사용 가능 여부
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled 
from transformers.utils import is_peft_available 

# -------------------- TRL(Transformer Reinforcement Learninig강화학습) 관련 import --------------------
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template  
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation  
from trl.trainer.grpo_config import GRPOConfig  
from trl.trainer.utils import generate_model_card, get_comet_experiment_url  

# -------------------- Qwen-VL 유틸 import --------------------
from qwen_vl_utils import process_vision_info 

# -------------------- 기타 --------------------
import copy  # 딥카피

# PEFT (Parameter-Efficient Fine-Tuning) 지원 여부 확인
if is_peft_available():
    from peft import PeftConfig, get_peft_model

# Weights & Biases 통합 여부 확인
if is_wandb_available():
    import wandb

# RewardFunc 타입 정의: 문자열(model id), 사전학습 모델, 또는 커스텀 함수
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
## Callable 은 함수라는 뜻으로, 위에는 list,list 를 2개의 인자로 받고 output으로 list[float]가 나온다는 의미.


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```
    #trl은 Transformer Reinforcement Learning으로 huggingface 에서 만들 강화학습 framework.
    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,# 데이터 전처리에 사용되는 토크나이저(Processing class)를 지정합니다.
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype) # getattr은 torch의 torch_dtype를 가져오는 속성함수
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            # gradient_checkpointing=True일 땐 반드시 use_cache=False로 설정해야 안전합니다.
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            ) 

            # model_id에 따라 적절한 모델 클래스를 선택하여 로드합니다.
            # Qwen2-VL, Qwen2.5-VL, Aria 등 다양한 모델 이름에 따라 분기 처리합니다.
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                # Aria 모델의 경우 use_cache 인자를 제거해야 합니다.
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                # 기본적으로 Qwen2.5-VL 모델을 사용합니다.
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        #self.ref_model = None
        # Reference model
        # deepspeed란, Ms에서 개발한 오픈소스 딥러닝 라이브러리(메모리최적화 -> 모델학습 가능하게)
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                # self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class -> Optional이며 이것을 통해서, 텍스트를 모델이 이해할 수 있는 형식으로 바꾼다. 
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # 리워드 프로세싱 클래스 설정
        # reward_processing_classes가 None이면, reward_funcs의 개수만큼 None으로 채운 리스트를 생성
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        # 리스트가 아니면 리스트로 변환
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            # reward_processing_classes와 reward_funcs의 길이가 다르면 에러 발생
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        # 각 리워드 함수와 그에 대응하는 프로세싱 클래스를 순회
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        # 최종적으로 reward_processing_classes를 인스턴스 변수로 저장
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temporal = script_args.temporal # => video_r1에서 temporal 추가한 것임.
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,  # 생성할 최대 토큰 수
            do_sample=True,                            # 샘플링 사용
            top_p=0.95,                                # nucleus sampling에서 누적 확률 임계값 (0.95면 상위 95% 확률 내에서 샘플링
            temperature=1,                             # 샘플링 온도
            num_return_sequences=self.num_generations, # 생성할 시퀀스 개수
            pad_token_id=pad_token_id,                 # 패딩 토큰 id
        )
        # sequence의 절반으로 줄인거를 shuffled_num_generation이라고 부른 것 뿐임.
        self.shuffled_num_generations = self.num_generations // 2 
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = 0 # self.beta는 0으로 초기화(KL divergence 가 없기 때문.)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                # prepare_model은 딥러닝 모델을 효율적으로 학습(훈련)할 수 있도록 도와주는 util 함수임.
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # signature column이란, 모델이 forward할때 실제로 받는 입력값의 컬럼명
        # 만약 self.args.remove_unused_columns가 True라면, signature(시그니처)가 아닌 컬럼들은 제거된다.
        # 기본적으로, 이 메서드는 self._signature_columns를 모델이 기대하는 입력값(시그니처 컬럼)으로 설정한다.
        # 하지만 GRPOTrainer에서는 데이터를 사전 처리(preprocess)하기 때문에, 모델의 시그니처 컬럼을 사용하는 것이 맞지 않는다.
        # 대신, training_step 메서드에서 기대하는 컬럼들로 시그니처를 설정해야 하므로, 이 메서드를 오버라이드(재정의)한다.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B:BatchSize, L-1:Sequence_length, V:Vocabularly), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # return_outputs 이 True 이면 안되게 명시
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

                
        
        input_copy = copy.deepcopy(inputs[0]['prompt'])
        
        input_copy = self.remove_none_from_data(input_copy)
        
        if inputs[0]['data_type'] == 'image':
            input_copy[0]['content'][0]['image'] = os.getcwd() + "/data" + inputs[0]['path'][1:] 
        elif inputs[0]['data_type'] == 'video':
            input_copy[0]['content'][0]['video'] = os.getcwd() + "/data" + inputs[0]['path'][1:] 
            
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            if inputs[0]['data_type'] == 'image':
                input_copy[0]['content'][0]['image'] = os.getcwd() + "/data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
            elif inputs[0]['data_type'] == 'video':
                input_copy[0]['content'][0]['video'] = os.getcwd() + "/data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'
                
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        
        
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)


        # fix prompt_inputs["input_ids"] length issue
        if self.max_prompt_length is not None:
            # prompt_inputs["input_ids"]는 (배치, 시퀀스길이) 형태의 텐서임
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :] # 각 배치(행마다) 뒤에서 max_prompt_length개만 남기고 앞부분은 잘라냄
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]# attention mask도 위와 동이라헥

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        
        # Generate completions
        # with는 context manager 사용할때 쓰는것으로, 블록이 끝나면 자동으로 닫힘
        # 즉, with는 특정 상황에서만 임시로 뭔가를 하고 끝나면 자동으로 정리.
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
            if self.temporal:
                
                if video_inputs:
            
                    shuffled_prompt_completion_ids = unwrapped_model.generate(**shuffled_prompt_inputs, generation_config=self.shuffled_generation_config)
                    shuffled_prompt_length = shuffled_prompt_ids.size(1)
                    shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                    shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
                    shuffled_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)
                    
                else:
                    
                    shuffled_prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.dummy_generation_config)

        
        print('path:', input_copy[0]['content'][0][inputs[0]['data_type']])   
        print('problem_id:', inputs[0]['problem_id'])       
        print('prompt_length:', prompt_length)
                
        
        
        
        # Mask everything after the first EOS(End of Sequence) token
        # 처음등장하느 EOS(End of Sequence) 토큰 이후의 모든 토큰을 마스킹(무시)하기 위한 마스크를 만드는 과정.
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        # 각 배치(문장)별로 EOS토큰이 하나라고 있으면 True 없으면 False
        # 각 배치별로 처음 등장하는 EOS토큰의 인덱스 반환 , EOS토큰이 있는 문장에 대해서만 인덱스를 실제 EOS위치로 바꿔줌
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        #시퀀으 인덱스를 만들어줌 [0,1,2,....L-1]을 배치 크기만큼 복제해서 (B,L)텐서로 만듦
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        
        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        # import pdb; pdb.set_trace()
        

        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]
                # prompt_inputs["second_per_grid_ts"] = torch.tensor(prompt_inputs["second_per_grid_ts"]).repeat(len(prompt_completion_ids), 1)
        
        
        
        ### 보상 기반의 강화학습(GRPO)으로 텍스트 생성 모델을 정교하게 미세조정하는 손실 함수 계산 로직###
        try:
            # 1) per-token log-probs 계산 (학습 모델)
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            # prompt 부분 제외하고 completion 부분만 슬라이싱
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
        except Exception as e:
            # 에러 시 fallback: 기본 호출로 로그확률 재계산
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            # per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device, requires_grad=True)
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
        
        torch.inference_mode = torch.no_grad  
        with torch.inference_mode():
            try:
                # 2) reference 모델의 per-token log-probs 계산(정책함수)
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    model_unwrapped = self.accelerator.unwrap_model(model)
                    if hasattr(model_unwrapped, "disable_adapter"):
                        with model_unwrapped.disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                    else:
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)

                # prompt 제외한 부분만
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                model_unwrapped = self.accelerator.unwrap_model(model)
                if hasattr(model_unwrapped, "disable_adapter"):
                    with model_unwrapped.disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                else:
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]


        # Compute the KL divergence between the model and the reference model
        # 3) KL divergence approximation: clamp 후 exp(x)-x-1
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10) 
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        if self.temporal and video_inputs:
            # 4) temporal 모드: 셔플된 completions 준비
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                 # 대화형이면 role/content 포맷으로 래핑
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion in shuffled_completions]
                
            # Compute the rewards
            # prompt 반복 생성
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            # 보상 함수별 결과를 저장할 tensor 초기화
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            
            # 5) 각 reward function 실행
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                # 보상 함수 호출
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32, device=device)

        
        # Decode the generated completions
        # 6) 최종 생성 completions 디코딩 & 보상 계산
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        

        
        
        if self.temporal and video_inputs:
             # 7) temporal 모드 보상 조정
            temporal_rewards_per_func = rewards_per_func.clone()
            
            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()

            if acc_mean >= 0.8 * shuffled_acc_mean:
                mask = temporal_rewards_per_func[:, 0] > 0.1
                temporal_rewards_per_func[mask, 0] = temporal_rewards_per_func[mask, 0] + 0.3
                temporal_rewards = torch.tensor([1.0]).to('cuda')
            else:
                temporal_rewards = torch.tensor([0.0]).to('cuda')
        else:
            temporal_rewards =  torch.tensor([0.5]).to('cuda')
        
        # Sum the rewards from all reward functions
        # 8) Temporal 보상 합산
        if self.temporal and video_inputs:
            rewards = temporal_rewards_per_func.sum(dim=1)
        else:
            rewards = rewards_per_func.sum(dim=1)
    
        
        if self.len_control:
             # 9) 길이 제어: reward_funcs[:,0] 기준 mask            
            mem_rewards = [0] * self.num_generations
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()

            if len(selected_indices) > 1:     
                for idx in selected_indices:
                    if 320 <= lenth_list[idx] <= 512:
                        rewards[idx] += 0.2
        
        print(rewards)
        print(completion_mask.sum(1))

        # Compute grouped-wise rewards
        # 11) 그룹별 통계 (mean/std) 및 어드밴티지 계산
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        

       # 12) per-token 손실 계산 
       # 여기 부분이 min(clip )부분임. 그런데 clip으로 안쓰고 구현에서는 PPOstyle 정책비율과 어드밴티지를 곱해주는 per-token loss 구성
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -per_token_loss      # ← KL 항 제거
        # per_token_loss = -per_token_loss
        # policy ratio * advantage =  exp(log pi - log pi_{stopgrad})*A
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # 수정
        ## [기여점](1) 분모를 생성된 총 토큰 수로 변경.
        tokens_in_batch = completion_mask.sum()             # 전체 토큰 개수
        loss = (per_token_loss * completion_mask).sum() / tokens_in_batch
       
            


        # ——— Log the metrics ———
        ### metric logging 이 하는 일 ###
        # 1) completion length: 디바이스별 토큰 개수 합을 모아 평균
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        # 2) reward_per_func: 함수별 보상 평균
        #    rewards_per_func.shape = [(batch_size*num_gen), num_funcs]
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            # 모델인지 함수인지에 따라 이름 추출
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        # 3) 전체 rewards 모아서 디바이스별 구조로 변환
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        num_devices = gathered_rewards.size(0) // self.num_generations 
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        
        # 4) all_wrong: 한 디바이스의 모든 gen 보상이 ≤ 1 인 비율
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        self._metrics["all_wrong"].append(wrong_ratio)
        
        # 5) all_correct: 모든 gen 보상이 ≥ 2 인 비율
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices       
        self._metrics["all_correct"].append(correct_ratio)
        
        # 6) temporal 모드 추가 보상 로깅
        if self.temporal:
            temporal_rewards_list = self.accelerator.gather_for_metrics(temporal_rewards)
            self._metrics["temporal_rewards"].append(self.accelerator.gather_for_metrics(temporal_rewards_list).mean().item())
        
        # 7) 전체 평균 보상, 표준편차, KL
        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        
        # 8) per-token KL 평균
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
