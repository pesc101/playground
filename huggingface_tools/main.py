import logging
import re

import hydra
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

from config import DBQRConfig
from src.data.pre_processing import DBQR
from src.llm.conversation import Conversation
from src.llm.inference import InferenceModule

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="evaluation_ragas", node=DBQRConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DBQRConfig) -> None:
    inference_module = InferenceModule(
        cfg.model_name, revision=None, max_new_tokens=cfg.max_tokens
    )
    dbqr = DBQR("/ltstorage/home/strich/dbqr/raw")
    df = dbqr.df.head(10)
    prompts = df["raw_prompt"].tolist()
    ids = df["id"].tolist()
    with open("system_prompt.txt", "r") as f:
        system_prompt = f.read()

    outputs = {}
    for id, prompt in tqdm(
        zip(ids, prompts), total=len(prompts), desc="Processing prompts"
    ):
        conversation = Conversation(system_prompt, inference_module)
        conversation.add_message(prompt)
        output = inference_module.generate_response(conversation.messages, None)
        outputs[id] = output
        print(output)

    def calculate_exact_match(df):
        exact_matches = df.apply(lambda row: row["answer"] == row["llm_answer"], axis=1)
        exact_match_ratio = exact_matches.mean()
        return exact_match_ratio

    def extract_answer(text):
        match = re.search(r"\[Answer START\](.*?)\[Answer END\]", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    answers = {id: extract_answer(output) for id, output in outputs.items()}

    df["llm_answer"] = df["id"].map(answers)
    df[["id", "answer", "llm_answer"]].to_csv("results.csv", index=False)

    exact_match_ratio = calculate_exact_match(df)
    print(f"Exact Match Ratio: {exact_match_ratio:.2%}")


if __name__ == "__main__":
    main()


# conversation = Conversation(system_prompt, inference_module)
# conversation.add_message("user")
# output = inference_module.generate_response(conversation.messages, "tool")
# conversation.add_tool_call(output)
# output = inference_modul.generate_response(conversation.messages, "tool"
