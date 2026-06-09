from agents.knowledge_constructor import KnowledgeConsructor
from agents.rag import RAG
from storage.vector_db import TrajectoryStore
from storage.knowledge_graph import KnowledgeGraph
from core.lm_setup import _setup_lm

import dspy
import pdb;
if __name__ == "__main__":
    prompt="""Fix $n=2$ and take $\epsilon \in (0,1)$. For $i = 1,\dots,4$, let $B_i \subset \R^{2n}$ denote the closed unit ball centered at $(3i-3,0,\dots,0)$.
            Your goal is to find an algorithm that takes $\epislon$ as input and produces as output an explicit smooth Hamiltonian function $H_\epsilon: \R^{2n} \times [0,1] \rightarrow \R$ whose time-$1$ Hamiltonian flow $\phi: \R^{2n} \ra \R^{2n}$ satisfies $\phi(\sqcup_{i=1}^{4}B_i) \subset B^{2n}(R)$, where $B^{2n}(R) \subset \R^{2n}$ is the closed ball centered at the origin of radius $R$,  and where we require the volume ratio of $\bigsqcup\limits_{i=1}^{4} B_i$ to $B^{2n}(R)$ to satisfy $\dfrac{4}{R^{2n}} > 1 - \epsilon$.
            Present the algorithm as a Python program that takes $\epislon$ as input. The algorithm's output must itself be a Python program that implements the Hamiltonian for the given $\epislon$. This program should be in a file called Hamiltonian.py which implements a function Hamiltonian(Q, P, t) that has the following properties.
                - Takes inputs Q (positions), P (momenta), and t (time).
                - Must be implemented using PyTorch operations to support automatic differentiation.
                - Inputs Q and P will be `torch.Tensor`s of shape `(N, n)` (where N is the size of a batch size of points sampled from the boundaries of the input balls).
                - Input t will be a scalar float.
                - Returns a `torch.Tensor` of shape `(N,)` representing the Hamiltonian value for each point in the batch.
                - Is smooth, to ensure existence and uniqueness of flow.
            A solution to this problem is meant to rest on finding a family of embeddings that has a natural algorithmic dependence on $\epsilon$. As such, the algorithm that produces the Hamiltonians for a given $\epsilon$ must finish running in under an hour on a typical laptop. The Hamiltonian function itself will be called multiple times during solution verification and must return within ten seconds on each call.
            """
    pdb.set_trace()
    vector_db = TrajectoryStore()
    graph_storage = KnowledgeGraph()
    lm = _setup_lm()
    dspy.configure(lm=lm) 
    
    from agents.knowledge_constructor import KnowledgeConsructor
    kc = KnowledgeConsructor(storage=vector_db, kg=graph_storage, question=prompt)
    kc.collect_literature(prompt=prompt)

    #Define the rag with the 
    rag = RAG()
    l1_res = rag.query(prompt=prompt, level=0)
    l2_res = rag.query(prompt=prompt, level=1)
    l2_res = rag.query(prompt=prompt, level=2)


    
