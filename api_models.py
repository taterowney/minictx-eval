import os, re
from openai import AzureOpenAI
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
from anthropic import Anthropic
load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant who is an expert in the Lean theorem prover. All code that you write will be in Lean 4, and will be placed inside ```lean  ``` blocks."

API_MODELS = ["gpt-o4-mini", "gemini", "claude"]

azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def prompt_model(prompt, model, temperatures, num_samples, max_tokens=256):
    # print("Calling api...")

    if model == "gpt-o4-mini":
        texts = complete_o4_mini(prompt, temperatures[0], num_samples, max_tokens)
    elif model == "gemini":
        texts = complete_gemini(prompt, temperatures[0], num_samples, max_tokens)

    scores = [0]*len(texts)

    texts, scores = _unique_sorted(texts, scores)
    for i, text in enumerate(texts):
        print(f"Response {i+1}:")
        print(text)
        print()
    return texts, scores

def complete_o4_mini(prompt, temp=0.7, num_samples=1, max_tokens=2048):
    texts = []
    # print(os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment = "o4-mini"

    # Prepare API request parameters
    responses = azure_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "user", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        # max_completion_tokens=max_tokens,
        # temperature=temperature,
        n=num_samples,
    )
    for choice in responses.choices:
        content = choice.message.content
        texts.append(content)
    return texts

def complete_gemini(prompt, temp=0.7, num_samples=1, max_tokens=2048):
    # print(prompt)
    contents = [
        # types.Content(role='system', parts=[types.Part.from_text(text=SYSTEM_PROMPT)]),
        types.Content(role='user', parts=[types.Part.from_text(text=prompt)])
    ]
    responses = google_client.models.generate_content(
        # model="gemini-2.5-pro-preview-03-25",
        model="gemini-2.5-flash-preview-05-20",
        contents=contents,
        config=GenerateContentConfig(
            system_instruction=[SYSTEM_PROMPT],
            temperature=temp,
            candidate_count=num_samples,
            # max_output_tokens=max_tokens,
        )
    )
    # print(f"Gemini response: {response}")
    # return [response]
    texts = []
    for c in responses.candidates:
        has_found = False
        if c.content is not None and c.content.parts is not None:
            for part in c.content.parts:
                if part.text:
                    texts.append(part.text)
                    has_found = True
                    break
        else:
            print(f"Empty response detected. Reason: {c.finish_reason.name}")
        if not has_found:
            # print("NO RESPONSE DETECTED")
            texts.append("")
    # print(f"Gemini response: {texts[0]}")
    return texts

def complete_claude(prompt, temp, num_samples, max_tokens=2048):
    texts = []
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temp,
    )
    for choice in response.choices:
        content = choice.text
        texts.append(content)
    return texts

def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_

def process_responses(responses):
    processed_responses = []
    for response in responses[0]:
        pattern = re.compile(r'```lean(.*?)```', re.DOTALL | re.IGNORECASE)
        match = pattern.search(response)
        if match:
            processed_responses.append(match.group(1).strip())
        else:
            processed_responses.append(response)
    return processed_responses

if __name__ == "__main__":
    prompt = "Please provide a simple Lean theorem and its proof."
    # model = "o4-mini"
    # temperatures = [0.7]
    # num_samples = 1
    # max_tokens = 256
    #
    # texts, scores = prompt_model(prompt, model, temperatures, num_samples, max_tokens)
    # processed_responses = process_responses((texts, scores))
    complete_gemini(prompt)

    # for i, response in enumerate(processed_responses):
    #     print(f"Processed Response {i+1}:")
    #     print(response)
    #     print()