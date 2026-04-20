from typing import Sequence, List, cast

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.reasoning import ReasoningParserManager
from vllm import LLM


def get_parser_class(model_name):
    grp = ReasoningParserManager.get_reasoning_parser
    lowered = model_name.lower()

    # Hack for now--looks like deepseek's distill-llama uses </think> and <end_of_sentence> syntax
    # This also has a couple unnecessary newlines after the </think>
    if "deepseek" in lowered and "distill-llama" in lowered:
        return grp("qwen3")
    if "llama" in lowered:
        return None

    if "gpt-oss" in lowered:
        return grp("openai_gptoss")
    # Need to address the special <end_of_sentence> token here
    if "qwen" in lowered:
        return grp("qwen3")
    if "deepseek" in lowered and "v3" in lowered:
        return grp("deepseek_v3")

    # Also used the </think> and <end_of_sentence> syntax
    if "deepseek" in lowered and "r1" in lowered:
        return grp("deepseek_r1")
    if "mistral" in lowered or "ministral" in lowered:
        return grp("mistral")


class Model:
    def __init__(self, model):
        self.__llm = LLM(model=model)
        self.__tokenizer = self.__llm.get_tokenizer()
        self.__pad_token_id = self.__tokenizer.pad_token_id

        sampling_params = self.__llm.get_default_sampling_params()
        sampling_params.max_tokens = 2048
        self.__sampling_params = sampling_params

        parser_class = get_parser_class(self.__llm.model_config.model)
        if parser_class is not None:
            reasoning_parser = parser_class(self.__tokenizer)
        else:
            reasoning_parser = None
        self.__reasoning_parser = reasoning_parser

    def __extract_content(self, token_ids: Sequence[int]) -> str:
        token_ids = cast(List[int], token_ids)
        reasoning_parser = self.__reasoning_parser
        pad_token_id = self.__pad_token_id
        if not reasoning_parser:
            trimmed_ids = token_ids
        else:
            trimmed_ids = reasoning_parser.extract_content_ids(token_ids)
        trimmed_ids = [x for x in trimmed_ids if x != pad_token_id]
        trimmed_text = self.__tokenizer.decode(trimmed_ids).lstrip("\n")
        return trimmed_text

    def __call__(self, prompts: Sequence[str] | str) -> List[str] | str:
        if isinstance(prompts, str):
            prompts = [prompts]
            scalar_in = True
        else:
            scalar_in = False
        llm = self.__llm
        sampling_params = self.__sampling_params

        batch_inputs: List[List[ChatCompletionMessageParam]] = [
            [{"role": "user", "content": c}] for c in prompts
        ]
        batch_outputs = llm.chat(batch_inputs, sampling_params=sampling_params)
        batch_texts = list(
            map(lambda x: self.__extract_content(x.outputs[0].token_ids), batch_outputs)
        )
        if scalar_in:
            return batch_texts[0]
        return batch_texts
