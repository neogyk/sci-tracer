import dspy
from agents import research_manager, rag, subagent
import uuid
from agents.rag import RAG, Researcher
from agents.rag import Executor
from dotenv import load_dotenv
load_dotenv()
class SciTracer(dspy.Module):
    """
    """
    def __init__(self):
#        os.environ["MISTRAL_API_KEY"] = "sBEjabQFOosPqTWOrBkJKKXFQK9RHCha"
        try:
            lm = dspy.LM("mistral/ministral-8b-2512")
            dspy.configure(lm=lm)
        except Exception: 
            pass
        rag_agent:dspy.Module = RAG()
        researcher:dspy.Module = Researcher()
        executor:dspy.Module = dspy.ChainOfThought(Executor)
        self.problem_text: str|None = None
        self.plan:str:None = None
        return
    
    def make_plan(self, problem_text:str) -> List[str]:
        literature = rag_agent.forward(problem_text)
        collection = chroma_client.create_collection(name=f"scibench_{i}")
        index = str(uuid.uuid4())
        points = []
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        for traj_id in range(len(literature)):
            trajectory = literature[traj_id]
            if traj_id == len(literature) - 1:
                next_id = None
            index = str(uuid.uuid4())
            next_id = str(uuid.uuid4())
            state = trajectory.problem
            action = trajectory.method
            result = trajectory.result
            collection.add(
                ids=[index + "a", index + "b", index + "c"],
                embeddings=[
                    list(model.embed(state))[0],
                    list(model.embed(action))[0],
                    list(model.embed(result))[0],
                ],
                documents=[
                    state,
                    action,
                    result,
                ],
                metadatas=[
                    {"next_id": next_id},
                    {"next_id": next_id},
                    {"next_id": next_id},
                ],
            )
        results = collection.query(query_texts=[problem_text], n_results=10)

        for paragraph in results:
            plan = researcher.forward(problem_text, paragraph)

        return plan
    
    def validate_plan(self) -> bool:
        return
    
    def verify_plan(self) -> List[str]:
        return
    
    def _create_executer(self) -> dspy.Module:
        return

    def execute_plan(self, plan:str) -> :
        return

    
