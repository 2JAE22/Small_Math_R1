#################################### 
# Reward funciton 을 적은 공간입니다.#
####################################

import re
import os
from datetime import datetime




# 1) Format 보상: r_0 -> 맞으면 0.5,틀리면 0(형식이 틀리면 정답이 맞아도 무조건 마이너스로 패널티줌.)
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
    return [0.5 if match else 0.0 for match in matches]

# 2) Length 보상: LR  <think>안의 길이선에 따른 보상. -> min(1,Len/ML) *r1(r1은 사용자가 정해줌. 여기선 0.5) 해서 최대 0.5점 받게함.
def length_reward(completions, **kwargs):
    """
    Length Reward (r₁):
      LR = min(1, Len/ML) * r1
    여기서
      - Len: <think>…</think> 태그 안의 문자 수
      - ML: 최대 보상 한도에 대응하는 길이 (자유롭게 조정 가능)
      - r1: 최대 보상값 (예: 0.5)
    """
    r1 = kwargs.get("r1", 0.5)   # 최대 보상값
    ML = kwargs.get("ML", 512)   # 최대 길이 한도
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        matches = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        think_len = len(matches.group(1)) if matches else 0

        # 비율 계산 후 r1 곱하기
        ratio = think_len / ML
        LR = min(1.0, ratio) * r1
        rewards.append(LR)

    return rewards

# 3) accuracy 보상: r_2 맞으면 1 ,틀리면0
# def accuracy_reward(completions, solution, **kwargs):
    
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

   
    # accuracy_reward 함수의 주요 동작을 설명합니다.
    # 이 함수는 문제 유형에 따라 모델의 출력(completions)과 정답(solution)을 비교하여 보상을 계산합니다.
    # 각 문제 유형별로 보상 계산 방식이 다릅니다.

    # 1. 문제 유형을 받아옵니다.
    question_type = kwargs['problem_type'][0]
    print("question_type: ", question_type)
    
    # 2. completions에서 실제 답변 텍스트만 추출합니다.
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    # 3. 각 답변(content)과 정답(sol)에 대해 반복하며 보상을 계산합니다.
    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)  # 모델이 생성한 답변에서 <answer> 태그 안의 값을 추출
            gt_ans = extract_answer(sol)         # 정답에서 <answer> 태그 안의 값을 추출

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
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # 연도-월-일-시-분-초
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    # 각 샘플별로 계산된 보상 리스트를 반환
    return rewards

#4) 향후 사용할 partial_reward 스텁(주석 처리)
# def partial_reward(completions, solution, **kwargs):
#     """
#     형식은 지켰지만 답이 틀린 경우에 주는 보상.
#     현재 구체적인 reward 방식은 중간 추론 과정을 사람이 계속 feedback 주는 방안 생각중..
#   """
#     
import math

def accuracy_reward(completions, solution, **kwargs):
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            # 빈 문자열 등 예외 처리
            if num_str == "" or num_str is None:
                return None
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None
    question_type = kwargs['problem_type'][0]
    print("question_type: ", question_type)
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            # 방어 코드 추가: 둘 다 공란이면 0점
            if output_ans == "" or gt_ans == "":
                reward = 0.0
            elif question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    # 둘 중 하나라도 변환 실패
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
        # NaN 방어
        if reward is None or (isinstance(reward, float) and (math.isnan(reward) or math.isinf(reward))):
            reward = 0.0
        rewards.append(float(reward))
        # DEBUG_MODE 로그 등은 그대로 두면 됨
    return rewards

# 4) 총보상 계산 함수
def total_reward(completions, solution, **kwargs):
    """
    FR = format_reward(r_0) + length_reward(r_1)
    AR = accuracy_reward(r_2)
    R  = { AR+FR  (FR>0 & AR>0)
         { -FR    (FR>0 & AR==0)
         { - (r0_max + r1_max + r2_max)  (FR==0)
    """

    # ① 세 부분 보상 리스트 구하기
    fr_list = format_reward(completions, **kwargs)                # r0 값 (0.5 또는 0)
    lr_list = length_reward(completions, **kwargs)                # 0 ~ 0.5
    ar_list = accuracy_reward(completions, solution, **kwargs)    # 1.0 또는 0

    # ② 최대값(상한) –  penalty 계산에 필요
    r0_max = 0.5                    # format_reward returns 최대 0.5
    r1_max = kwargs.get("r1", 0.5)  # length_reward의 상한
    r2_max = 1.0                    # accuracy_reward의 상한

    rewards = []
    for fr, lr, ar in zip(fr_list, lr_list, ar_list):
        FR = fr + lr   # 형식 + 길이
        AR = ar        # 정확도

        if FR > 0 and AR > 0:          # ✅ 형식 OK & 정답
            R = FR + AR
        elif FR > 0 and AR == 0:       # ⚠️ 형식 OK & 오답
            R = -FR
        else:                          # ❌ 형식 자체 오류
            R = -(r0_max + r1_max + r2_max)
        rewards.append(R)

    return rewards

