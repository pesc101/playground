import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class InferenceModule:
    def __init__(
        self,
        checkpoint: str,
        revision: str = "pr/13",
        max_new_tokens: int = 128,
    ) -> None:
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "35GB"},
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self.init_tools()
        self.tools_dict = {func.__name__: func for func in self.tools}

    def generate_response(self, messages: list, chat_template: str) -> list[str]:
        template = None
        tools = None
        if chat_template == "tool":
            template = "tool_use"
            tools = self.tools

        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tools=tools,
        )
        tokenized_chat = {k: v.to(self.model.device) for k, v in tokenized_chat.items()}
        out = self.model.generate(**tokenized_chat, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(out[0][len(tokenized_chat["input_ids"][0]) :])

    def init_tools(self):
        def get_current_temperature(location: str, unit: str) -> float:
            """
            Get the current temperature at a location.

            Args:
                location: The location to get the temperature for, in the format "City, Country"
                unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
            Returns:
                The current temperature at the specified location in the specified units, as a float.
            """
            return random.randint(0, 20)

        def get_current_wind_speed(location: str) -> float:
            """
            Get the current wind speed in km/h at a given location.

            Args:
                location: The location to get the temperature for, in the format "City, Country"
            Returns:
                The current wind speed at the given location in km/h, as a float.
            """
            return random.randint(0, 20)

        self.tools = [get_current_temperature, get_current_wind_speed]
