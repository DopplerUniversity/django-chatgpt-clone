from typing import Callable

from django.conf import settings
from duckduckgo_search import DDGS
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def web_search(query):
    results = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, region="us-en"):
            results.append(result)

    return results


def chat(message, conversation, model, system_message, web_access=False) -> Callable:
    def stream(response):
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_message}
    ]

    if web_access:
        search_results = web_search(message["content"])[:5]
        search_messages = [
            f"[{index}] \"{result['body']}\" URL:{result['href']}" for index, result in enumerate(search_results)
        ]
        search_messages += [
            "Current date: {datetime.now().strftime('%Y-%m-%d')",
            "Instructions: Using the provided web search results, write a comprehensive reply to the next user query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refers to multiple subjects with the same name, write separate answers for each subject. Ignore your previous response if any.",  # noqa: E501 pylint: disable=line-too-long
        ]
        messages.append({"role": "user", "content": "\n\n".join(search_messages)})

    if conversation:
        messages += conversation

    messages.append(message)

    response = client.chat.completions.create(
        messages=messages, model=model, temperature=settings.OPENAI_TEMPERATURE, max_tokens=500, stream=True
    )
    return lambda: stream(response)
