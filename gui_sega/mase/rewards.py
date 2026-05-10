import re
import ast
import json
import math
import difflib
from Levenshtein import ratio as levenshtein_ratio


def determine_swipe_direction(coords):
    if len(coords) != 2 or any(len(point) != 2 for point in coords):
        return "Invalid input: two 2D coordinate points required"

    x1, y1 = coords[0][0], coords[0][1]
    x2, y2 = coords[1][0], coords[1][1]

    dx = x2 - x1
    dy = y2 - y1

    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance < 1e-5:
        return "None"

    if abs(dx) > abs(dy):
        direction = "Right" if dx > 0 else "Left"
    elif abs(dy) > abs(dx):
        direction = "Down" if dy > 0 else "Up"
    else:
        if dx > 0:
            direction = "Right"
        else:
            direction = "Left"

    return direction


def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str = predicted_str.replace("[", "").replace("]", "")
    ground_truth_str = ground_truth_str.replace("[", "").replace("]", "")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens) == 1 and len(ground_truth_tokens) == 1:
        predicted_token = list(predicted_tokens)[0]
        ground_truth_token = list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    precision = len(common_tokens) / len(predicted_tokens) if len(predicted_tokens) > 0 else 0
    recall = len(common_tokens) / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def _extract_answer_from_completion(completion_text):
    if "</thinking>" in completion_text:
        t = completion_text.split("</thinking>")[-1]
    else:
        t = completion_text

    if "<answer>" in t and "</answer>" in t:
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_index = t.find(start_tag)
        end_index = t.rfind(end_tag)
        extracted = t[start_index + len(start_tag): end_index].strip()
        return extracted, True
    return None, False


def _extract_answer_from_solution(solution_text):
    if "</thinking>" in solution_text:
        sol_answer = solution_text.split("</thinking>")[-1]
    else:
        sol_answer = solution_text

    start_tag = "<answer>"
    end_tag = "</answer>"
    start_index = sol_answer.find(start_tag)
    end_index = sol_answer.rfind(end_tag)
    if start_index != -1 and end_index != -1:
        return sol_answer[start_index + len(start_tag): end_index].strip(), True
    return None, False


def _is_malformed_json(extracted_answer):
    return (
        extracted_answer.startswith("{{")
        or extracted_answer.endswith("}}")
        or not extracted_answer.startswith("{")
        or not extracted_answer.endswith("}")
    )


def _safe_eval_dict(s):
    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return result, True
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return result, True
    except (ValueError, SyntaxError):
        pass
    return None, False


def format_reward_func(completions, **kwargs):
    pattern = r"^<thinking>.*?</thinking>.*?<answer>.*?</answer>$"
    matches = [re.match(pattern, content[0]["content"], re.DOTALL) for content in completions]
    return [0.1 if match else 0.0 for match in matches]


def thinking_reward_func(completions, solution, **kwargs):
    rewards = []
    for completion, solution in zip(completions, solution):
        sol_match = re.search(r"<thinking>(.*?)</thinking>", solution, re.DOTALL)
        reference_text = sol_match.group(1).strip() if sol_match else solution.strip()

        raw_content = completion[0]["content"]
        comp_match = re.search(r"<thinking>(.*?)</thinking>", raw_content, re.DOTALL)
        if comp_match:
            candidate_text = comp_match.group(1).strip()
            similarity = difflib.SequenceMatcher(None, reference_text, candidate_text).ratio()
            rewards.append(similarity * 0.2)
        else:
            rewards.append(0.0)

    return rewards


def accuracy_reward_type(completions, solution, **kwargs):
    res = []
    for completion, solution in zip(completions, solution):
        completion_text = completion[0]["content"]
        extracted, has_answer = _extract_answer_from_completion(completion_text)
        if not has_answer:
            res.append(0.0)
            continue

        sol_extracted, sol_has = _extract_answer_from_solution(solution)
        if not sol_has:
            res.append(0.0)
            continue

        sol, sol_ok = _safe_eval_dict(sol_extracted)
        if not sol_ok or _is_malformed_json(extracted):
            res.append(0.0)
            continue

        extracted_dict, ext_ok = _safe_eval_dict(extracted)
        if not ext_ok:
            res.append(0.0)
            continue

        res.append(levenshtein_ratio(extracted_dict["action_type"].lower(), sol["action_type"].lower()) * 0.3)

    return res


def accuracy_reward_action(completions, solution, **kwargs):
    res = []
    for completion, solution in zip(completions, solution):
        completion_text = completion[0]["content"]
        extracted, has_answer = _extract_answer_from_completion(completion_text)
        if not has_answer:
            res.append(0.0)
            continue

        sol_extracted, sol_has = _extract_answer_from_solution(solution)
        if not sol_has:
            res.append(0.0)
            continue

        sol, sol_ok = _safe_eval_dict(sol_extracted)
        if not sol_ok or _is_malformed_json(extracted):
            res.append(0.0)
            continue

        extracted_dict, ext_ok = _safe_eval_dict(extracted)
        if not ext_ok:
            res.append(0.0)
            continue

        if extracted_dict["action_type"] != sol["action_type"]:
            res.append(0.0)
            continue

        action_type = extracted_dict["action_type"]
        pred_info = extracted_dict["action_info"]
        sol_info = sol["action_info"]

        if action_type == "SCROLL":
            if all(isinstance(x, (int, float)) for x in pred_info):
                res.append(0.0)
            else:
                direction_sol = determine_swipe_direction(sol_info)
                direction_pred = pred_info
                res.append(0.4 if direction_sol == direction_pred else 0.0)
        elif action_type == "CLICK" and all(isinstance(x, list) for x in sol_info):
            if all(isinstance(x, (int, float)) for x in pred_info):
                sol_x1, sol_y1 = sol_info[0][0], sol_info[0][1]
                sol_x2, sol_y2 = sol_info[1][0], sol_info[1][1]
                answer_x, answer_y = float(pred_info[0]), float(pred_info[1])
                res.append(0.4 if sol_x1 < answer_x < sol_x2 and sol_y1 < answer_y < sol_y2 else 0.0)
            elif pred_info == sol_info:
                res.append(0.4)
            else:
                res.append(0.0)
        else:
            res.append(levenshtein_ratio(pred_info, sol_info) * 0.4)

    return res
