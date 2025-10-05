import os
from mistralai import Mistral
from rich.console import Console

# import dspy


class Solver:
    def __init__(self):
        # register the reasoning model vendor
        return

    def retrive_information(self):
        return

    def solve_problem(self):
        return


class WebAgent:
    def __init__(self):
        return

    def search_info(self, prompt: str):
        return


class SciTracer:
    def __init__(self, config=None):
        self.model = "codestral-latest"
        self.console = Console()
        self.llm_service = Mistral(api_key=os.environ["MISTRAL_API"])
        with open(config.prompt) as f:
            d = f.read()
            prompt = "".join(d.split("\n")[1:-1])
        self.system_prompt = prompt

    def _search_external(self):
        return

    def _web_search(self):
        return

    def problem_finding(self):
        return

    def solution_finding(self, problem: "str"):
        return

    def novelty_estimation(self):
        return

    def __call__(self, text):
        for t in text:
            response = self.llm_service.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": t + self.system_prompt,
                    },
                ],
            )
            # TODO store the graph to the database
            yield response
