import json
import re
import uuid

from src.llm.inference import InferenceModule


class Conversation:
    def __init__(self, sys_prompt: str, inference_module: InferenceModule) -> None:
        self.inference_module = inference_module
        self.messages = self.init_message(sys_prompt)

    def init_message(self, sys_prompt: str) -> None:
        return [
            {
                "role": "system",
                "content": sys_prompt,
            },
        ]

    def add_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_tool_call(self, content: str) -> None:
        content = self.__post_process_message(content)
        function_output = self.__call_function(content)

        tool_call_id = uuid.uuid4()
        self.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": tool_call_id, "type": "function", "function": content}
                ],
            }
        )
        self.__add_tool_response(tool_call_id, content["name"], function_output)

    def __post_process_message(self, message: dict) -> dict:
        output = message.replace("\n", "")
        try:
            json_str = re.search(r"<tool_call>(.*?)</tool_call>", output.strip()).group(
                1
            )
            json_str = json.loads(json_str)
        except AttributeError as e:
            print(f"Output: {output} | \n Error: {e}")
            print(e)
            json_str = {
                "arguments": {"location": "New York, USA", "unit": "fahrenheit"},
                "name": "get_current_temperature",
            }

        return json_str

    def __call_function(self, tool_call: dict) -> str:
        function_name = tool_call["name"]
        arguments = tool_call["arguments"]

        func = self.inference_module.tools_dict.get(function_name)
        return func(**arguments)

    def __add_tool_response(
        self, tool_call_id: str, name: str, function_output: str
    ) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": function_output,
            }
        )
