<img width="400" height="300" alt="SciTracer" src="https://github.com/user-attachments/assets/03fb9c10-70f0-47e8-b122-6100a1390956" />
 
### Sci-Tracer provable and verifiable scientific discovery.
Sci-Tracer is an agentic systems for automatic, efficient scientific discovery that utilize the plan formalization and logic verification.



### Knowledge Constructor:

```
from storage.vector_db import TrajectoryStore
from storage.knowledge_graph import KnowledgeGraph
from agents.knowledge_constructor import KnowledgeConsructor
from agents.rag import RAG
from agents.researcher import Researcher
from agents.formalizer import Formalizer
from agents.verifier import Verifier
from agents.executor import Executor
from agents.evolution import EvolutionJudge


question = "TYPE HERE research question"

vector_db     = TrajectoryStore()
graph_storage = KnowledgeGraph()

kc = KnowledgeConsructor(
        question = question,
        storage  = vector_db,
        kg       = graph_storage,
    )


```


### Plan definition:

```
rag        = RAG(vector_db, graph_storage)
researcher = Researcher(rag=rag)

plan = researcher.forward(prompt)
```

### Plan formalization:

```
formalizer = Formalizer()
formalized_plan = formalizer.forward(plan)
```


### Plan verification:

```
verifier   = Verifier()
verification_results = verifier.forward(formalized_plan)
```

### Plan execution:

```
executor   = Executor()
refined     = verification_results.refined_plan or plan
code_result = executor.forward(refined, research_plan=plan)
exec_output = code_result.output

```
