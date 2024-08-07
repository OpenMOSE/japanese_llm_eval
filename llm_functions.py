import os

from datasets import Dataset
import openai
import time


# === 評価生成関数群 ===
def get_response_from_openai(messages: list, model_name: str) -> str:
    #client = OpenAI(
    #    api_key=os.environ.get("OPENAI_API_KEY")
    #)
    api_key=os.environ.get("OPENAI_API_KEY")
    openai.api_key = api_key
    #openai.api_base = base_url
    time.sleep(5)

    evaluation_temperature = 0
    evaluation_max_tokens = 1024

    response = openai.ChatCompletion.create(
        messages=messages,
        model=model_name,
        temperature=evaluation_temperature,
        max_tokens=evaluation_max_tokens
    )
    return response.choices[0].message.content


def get_response_func(model_name: str) -> callable:
    if "gpt" in model_name:
        return get_response_from_openai
    else:
        """
        他のモデルで評価する場合は関数、分岐をここに追加
        """
        raise NotImplementedError(f"Model {model_name} is not supported")


def get_model_response(messages: list, model_name: str) -> str:
    answer_function = get_response_func(model_name)
    return answer_function(messages, model_name)


# === 回答生成関数群 ===
def get_answer(question: str, model_name: str):
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if api_key == "EMPTY":
        base_url = "http://localhost:9000/v1"
    else:
        base_url = None

    #client = OpenAI(
    #    api_key=api_key,
    #    base_url=base_url,
    #)
    print(base_url)
    openai.api_key = api_key
    openai.api_base = base_url

    generation_temperature = 0
    generation_max_tokens = 2048

    response = openai.ChatCompletion.create(
        messages=[{"role": "user", "content": question}],
        model=model_name,
        temperature=generation_temperature,
        max_tokens=generation_max_tokens
    )
    return response.choices[0].message.content


def get_answerer(model_name: str) -> callable:
    """OpenAIとvLLM以外のモデルを使う場合はここに追加する"""
    return get_answer


def get_model_answer(dataset: Dataset,
                     model_name: str,
                     batch_size: int) -> Dataset:
    answer_function = get_answerer(model_name)
    print(answer_function)
    dataset = dataset.map(
        lambda x: {"ModelAnswer": answer_function(x['Question'], model_name)},
        num_proc=batch_size
    )
    return dataset
