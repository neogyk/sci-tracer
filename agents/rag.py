import dspy
import sys

sys.path.append("/Users/leonid/Desktop/FusionAI/SciTracer/tools")
from pub_search import arxiv_search, scolars_search, file_reader
import tqdm
import chromadb

# 1. Using the question find the papers
# 2. Parse the papers and decompose it into the sequence of the problem -> solution -> results/conclustion
# 3. Having the list of the trajectories. Create own instructions
# 4. Follow the instructions to perform the task.


class Trajectory(dspy.Signature):
    question = dspy.InputField(
        desc="The problem or question that is necessary to solve"
    )
    paragraph = dspy.InputField(desc="Part of the reseach article")
    problem: str = dspy.OutputField(
        desc="the formulation of problem that is being solve in the paragraph or article"
    )
    method: str = dspy.OutputField(
        desc="The instuction that author or other scientist or people proceeded in order to solve this problem. In could be the methodology or algorithm of list of instruction"
    )
    result: str = dspy.OutputField(desc="The result obtained by solving this problems")


class ResearchPlan(dspy.Signature):
    question = dspy.InputField(
        desc="The problem or question that is necessary to solve"
    )
    previous_research_plan = dspy.InputField(desc="Previouse Version of research plan")
    paragraph = dspy.InputField(desc="Part of the reseach article")
    research_plan = dspy.OutputField(
        desc="Research plan that is necessary to proceed inorder to resolve the problem"
    )


class Executor(dspy.Signature):
    task = dspy.InputField(desc="The task that is necessary to resolve")
    plan = dspy.InputField(
        desc="Research plan that is necessary to proceed inorder to resolve the problem"
    )
    answer = dspy.OutputField(desc="Final answer")


class Researcher(dspy.Module):
    """
    Generate the reaserch plan based on the

    :param dspy: _description_
    :type dspy: _type_
    """

    def __init__(self):
        self.researcher = dspy.ChainOfThought(ResearchPlan)
        return

    def forward(self, prompt, trajectory):
        previous_research_plan = ""
        for paragraph in trajectory:
            res = self.researcher(
                question=prompt,
                paragraph=paragraph,
                previous_research_plan=previous_research_plan,
            )
            previous_research_plan = res.research_plan
            print(previous_research_plan)
        return previous_research_plan


class RAG(dspy.Module):
    """
    The RAG Module provides the discovery trajectories for given idea, question or hypothesis.
    Please chunk the text of the document and decompose it into the sequence of the
    problem - method of the problem - solution and cosequences. This discovery trajectory for each of the paper
    will be used for the research planning.
    """

    def __init__(self, collection=None):
        self.trajectories = dspy.Predict(Trajectory)
        self.search_tools = [arxiv_search, scolars_search]
        self.researcher = dspy.ChainOfThought("Trajectory->Plan")
        self.collection = collection
        return

    def build_rag(self, prompt):
        papers_url = []
        for i in range(len(self.search_tools)):
            results = self.search_tools[i](prompt=prompt, n=2)
            for r in results:
                papers_url.append(r[0])
        return papers_url

    def sample_trajectory_points(self, starting_point, n_results=5):
        """Sample trajectory points (research plan) from embedding vectors.

        Retrieves similar trajectory points from the collection using the
        starting point text as a query. Returns the problem and method
        for each matched trajectory, excluding the completion (result).

        Args:
              starting_point: The input text to query similar trajectories.
              n_results: Number of similar results to retrieve.

        Returns:
              List of dicts with 'problem', 'method', and 'next_id' keys,
              representing the research plan steps without final answers.
        """
        results = self.collection.query(
            query_texts=[starting_point], n_results=n_results
        )

        # Extract base trajectory IDs (strip the a/b/c suffix)
        trajectory_ids = []
        seen = set()
        for doc_id in results["ids"][0]:
            base_id = doc_id[:-1]
            if base_id not in seen:
                seen.add(base_id)
                trajectory_ids.append(base_id)

        # For each trajectory, retrieve problem and method but NOT result
        plan_points = []
        for base_id in trajectory_ids:
            problem_id = base_id + "a"
            method_id = base_id + "b"

            trajectory_data = self.collection.get(ids=[problem_id, method_id])

            if trajectory_data and len(trajectory_data["documents"]) == 2:
                point = {
                    "problem": trajectory_data["documents"][0],
                    "method": trajectory_data["documents"][1],
                    "next_id": trajectory_data["metadatas"][0].get("next_id"),
                }
                plan_points.append(point)

        return plan_points

    def forward(self, prompt: str):
        tr_list = []
        papers_url = self.build_rag(prompt=prompt)
        # Form the trajectory
        for url in tqdm.tqdm(papers_url):
            if url is not None:
                try:
                    paper = file_reader(url)
                except Exception as e:
                    print(f"E: {e} | {url}")
                else:
                    for section in tqdm.tqdm(paper.split("##")):
                        if len(section) == 0:
                            continue
                        tr_list.append(
                            self.trajectories(paragraph=section, question=prompt)
                        )
        return tr_list
