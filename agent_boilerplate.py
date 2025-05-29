import os, re
import openai
from openai.types import chat
from concurrent.futures import ThreadPoolExecutor


def get_model_vendor(model_name, is_api=False):
    if model_name.lower().startswith("gpt") or re.match("^o[0-9].*$", model_name.lower()) is not None:
    # if model_name in models.list():
        return "openai"
    elif model_name.lower().startswith("gemini"):
        return "google"
    elif model_name.startswith("claude"):
        return "anthropic"
    else:
        if is_api:
            raise ValueError(f"Unsupported API model name: {model_name}")
        return "vllm"


    # params = vllm.SamplingParams(
    #     n=num_samples,
    #     temperature=temperature,
    #     use_beam_search=(temperature==0.0 and num_samples>1),
    #     max_tokens=max_tokens,
    #     stop=["\n\n\n", "---", "[/TAC]"],
    # )

class Client:
    def __init__(self, model_name, model_source=None, endpoint="http://localhost:8000/v1", **kwargs):
        if model_source is None:
            model_source = get_model_vendor(model_name)
        elif model_source == "api":
            model_source = get_model_vendor(model_name, is_api=True)

        self.model_source = model_source
        self.model_name = model_name
        if self.model_source == "vllm":
            from vllm import LLM
            self.client = LLM(
                model=model_name,
                tensor_parallel_size=kwargs.get("tp_degree", 1),
                dtype='bfloat16',
            )
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            return

        elif self.model_source == "vllm-online":
            self.client = openai.OpenAI(
                api_key="EMPTY",
                base_url=endpoint
            )
            return

        from dotenv import load_dotenv
        try:
            load_dotenv()
        except:
            pass

        if self.model_source == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        elif self.model_source == "azure":
            if get_model_vendor(model_name) != "openai":
                raise ValueError(f"Azure only supports OpenAI models, and {model_name} is not one.")
            from openai import AzureOpenAI
            if not os.getenv("AZURE_OPENAI_API_KEY"):
                raise EnvironmentError("AZURE_OPENAI_API_KEY environment variable not set.")
            if not os.getenv("AZURE_OPENAI_ENDPOINT"):
                raise EnvironmentError("AZURE_OPENAI_ENDPOINT environment variable not set.")
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )

        elif self.model_source == "google":
            if not os.getenv("GEMINI_API_KEY"):
                raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
            self.client = openai.OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

        elif self.model_source == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
            self.client = openai.OpenAI(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url="https://api.anthropic.com/v1/"
            )
        else:
            raise ValueError(f"Unsupported model vendor: {self.model_source}. Supported vendors are: openai, azure, google, anthropic, vllm, vllm-online.")

    def get_raw_response(self, messages, **kwargs):
        if self.model_source == "vllm-online" or self.model_source == "openai" or self.model_source == "azure" or self.model_source == "google":
            try:
                kwargs = self.to_OpenAI_format(kwargs)
                kwargs["model"] = self.model_name
                kwargs["messages"] = messages
                # API can't handle a ton of requests at once for some reason :(
                if "n" in kwargs and type(kwargs["n"]) == int and kwargs["n"] > 8:
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        requests = kwargs["n"]
                        while requests > 0:
                            batch_size = min(requests, 8)
                            kwargs["n"] = batch_size
                            futures.append(executor.submit(self.client.chat.completions.create, **kwargs))
                            requests -= batch_size
                        all_results = [future.result() for future in futures]
                        res = all_results[0]
                        res.choices = sum([f.choices for f in all_results], [])
                    return res
                res = self.client.chat.completions.create(**kwargs)
                return res
            except openai.BadRequestError as e:
                if "'code': 'unknown_model'" in e.message:
                    raise ValueError(f"\"{self.model_name}\" does not appear to be a valid OpenAI model. Visit https://platform.openai.com/docs/models for a list of available models, or specify the model source manually. If you're using Azure, make sure you have setup a deployment for the model in your Azure OpenAI resource, and that you are using the deployment name as the model name.")
                else:
                    raise e
        elif self.model_source == "vllm":
            params = self.to_vLLM_format(kwargs)
            outputs = self.client.chat(
                messages,
                sampling_params=params,
                use_tqdm=False,
            )
            return self.from_vLLM_format(outputs)
        else:
            assert False

    def get_response(self, messages, **kwargs):
        if type(messages) is str:
            messages = [{"role": "user", "content": messages}]
        res = self.get_raw_response(messages, **kwargs)
        return res.choices[0].message.content

    def to_OpenAI_format(self, kwargs):
        """
        Convert the kwargs to OpenAI format.
        """
        flags = ['messages', 'model', 'audio', 'frequency_penalty', 'function_call', 'functions', 'logit_bias', 'logprobs', 'max_completion_tokens', 'max_tokens', 'metadata', 'modalities', 'n', 'parallel_tool_calls', 'prediction', 'presence_penalty', 'reasoning_effort', 'response_format', 'seed', 'service_tier', 'stop', 'store', 'stream', 'stream_options', 'temperature', 'tool_choice', 'tools', 'top_logprobs', 'top_p', 'user', 'web_search_options', 'extra_headers', 'extra_query', 'extra_body', 'timeout']
        for kw in kwargs:
            if kw not in flags:
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"][kw] = kwargs[kw]
        return kwargs

    def to_vLLM_format(self, kwargs):
        from vllm import SamplingParams
        return SamplingParams(kwargs)

    def from_vLLM_format(self, outputs):
        """
        Convert vLLM SamplingParams to OpenAI format.
        """
        res = openai.ChatCompletion()
        for i, output in enumerate(outputs[0].outputs):
            text = output.text.replace(self.tokenizer.eos_token, '')
            res.choices.append(
                chat.chat_completion.Choice(
                    finish_reason="stop",
                    index=i,
                    message=chat.ChatCompletionMessage(
                        role="assistant",
                        content=text
                    )
                )
            )
        return res


if __name__ == "__main__":
    client = Client("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    print(client.get_response(messages, temperature=0.7, max_tokens=50, n=1))

