import json
import time
from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    retry_with_backoff,
)
from bfcl_eval.utils import contain_multi_turn_interaction
from openai import RateLimitError
from overrides import override


class AllenAIHandler(OSSHandler):
    """
    Handler for AllenAI models served via a local vLLM server.

    Inherits from OSSHandler for server lifecycle management (spin up / shut down
    vLLM, tokenizer loading, etc.) but replaces the prompting-only completions API
    with the full OpenAI Chat Completions API, including native tool-call (FC) support,
    mirroring the behaviour of OpenAICompletionsHandler.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, dtype=dtype, **kwargs)
        # Use the chat-completions model style so that tool conversion uses the
        # correct schema (identical to OpenAICompletionsHandler).
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        # The OpenAI client pointing at the local vLLM server is already
        # initialised by OSSHandler.__init__ (self.client).

    @property
    def _model_id(self) -> str:
        """Return the model identifier to pass to the API.

        After spin_up_local_server is called, self.model_path_or_id is set to
        the local path or HuggingFace model ID.  Fall back to self.model_name
        for deployments that skip server setup (e.g. a pre-existing endpoint).
        """
        return self.model_path_or_id or self.model_name

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()
        return api_response, end_time - start_time

    # ------------------------------------------------------------------
    # Inference dispatch  (override OSSHandler which is prompting-only)
    # ------------------------------------------------------------------

    @override
    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        is_multi_turn = contain_multi_turn_interaction(test_entry["id"])
        if self.is_fc_model:
            if is_multi_turn:
                return self.inference_multi_turn_FC(
                    test_entry, include_input_log, exclude_state_log
                )
            else:
                return self.inference_single_turn_FC(test_entry, include_input_log)
        else:
            raise NotImplementedError("This handler currently doesn't support non-FC mode.")

    # ------------------------------------------------------------------
    # Decode  (override OSSHandler's prompting-only defaults)
    # ------------------------------------------------------------------

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        if self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        if self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "messages": message,
            "model": self._model_id,
            "temperature": self.temperature,
        }

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        inference_data["tools"] = tools
        return inference_data

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [
                func_call.id for func_call in api_response.choices[0].message.tool_calls
            ]
        except Exception:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)
        return inference_data
