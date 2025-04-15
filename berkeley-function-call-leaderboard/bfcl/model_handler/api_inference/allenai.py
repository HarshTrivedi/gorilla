import json
import os

from bfcl.model_handler.api_inference.openai import OpenAIHandler
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import func_doc_language_specific_pre_processing, system_prompt_pre_processing_chat_model
from openai import OpenAI


class AllenAIHandler(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        host = os.getenv("VLLM_ENDPOINT")  # reusing VLLM env vars without creating new ones.
        port = os.getenv("VLLM_PORT")  # set them in .env
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1")
        self.is_fc_model = False

    def decode_ast(self, result, language="Python"):
        result = result.replace("<function_calls>", "").replace("</function_calls>", "")
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            try:
                parsed = eval(result)
            except Exception:  # Temporary hack
                return super().decode_ast(result)
        if isinstance(parsed, list) and all("name" in item and "arguments" in item  for item in parsed):
            return [{item["name"]: item["arguments"]} for item in parsed]
        return super().decode_ast(result)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )
        content = test_entry["question"][0][0]["content"]
        original_format = "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]"
        updated_format = (
            '[{"name": "func_name1", "arguments": {"params_name1": params_value1, "params_name2": params_value2, ...}}, '
            '{"name": "func_name2", "arguments": {"params_name1": params_value1, ...}}]'
        )
        content = content.replace(original_format, updated_format)
        content = content.replace(
            "You SHOULD NOT include any other text in the response.",
            "Make sure to also include module name as part of the function name when applicable. E.g., triangle_properties.get instead of just get.\n"
            "You SHOULD NOT include any other text in the response."
        )
        test_entry["question"][0][0]["content"] = content
        return {"message": []}
