import pdb
from agents.researcher import Researcher
from agents.formalizer import Formalizer
from agents.verifier import Verifier 
from agents.rag import RAG
from agents.executor import Executor
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
    prompt = """
            Please solve this problem it's from field of Quantum Mechanics (A 3-State QM Problem)
            The Hamiltonian of a three-level system is given as $H = \begin{pmatrix} E_a & 0 & A \\ 0 & E_b & 0 \\ A & 0 & E_a \\ \end{pmatrix}$ where $A$ is real. The state of the system at time $t=0$ is (in this basis) $\psi(t=0) = \frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ 1\\ 0\end{pmatrix}$ What is the expectation value of the energy at time $t$?
            Requirements:
                Provide the answer in the form of \texttt{python} code. Implement the following function \begin{python} def expectation_value(A: float, E_a:float, E_b:float, t:float) -> float: pass \end{python}
            """
    
    prompt = """
            Please solve this problem it's form the field of the cosmology
            In cosmology, large-scale cosmological dark-matter halo fields are biased tracers of the underlying Gaussian matter density $\delta_m$. Assume we have a sample $\delta_m$. We simulate a halo number density field by taking $n(\mathbf{x}) = \bar{n}\max(0,1+b\delta_m(\mathbf{x}))$, where bare number density $\bar{n}$ and bare bias $b$ are specified constants. What is the bias of the sampled halo field? Derive an equation to evaluate the bias which depends on the bare bias and the variance in each pixel.
            Provide the answer in the form of the \texttt{python} code. Implement the following function.
            \begin{python}
            #let b_in stand for bare bias
            def b_eff(sigma: float, b_in:float) -> float:   
            pass
            \end{python}
            """
    vector_db = TrajectoryStore()
    graph_storage = KnowledgeGraph()
    lm = _setup_lm()
    dspy.configure(lm=lm) 

#    agents.knowledge_constructor

    #TODO add the RAG;

    #Define the rag with the 
    rag = RAG(vector_db, graph_storage)
    researcher = Researcher(rag=rag)

    plan = researcher.forward(prompt)
    
    
    formalizer = Formalizer()
    executor = Executor()
    formalized_plan = formalizer.forward(plan)
    
    verifier = Verifier()
    #TODO add the verification results:
    verification_results = verifier.forward(formalized_plan)#, kripkes_structure)
    print(verification_results)

    refined_plan = verification_results.refined_plan
    code = executor.forward(plan, research_plan=plan)

    #TODO add the summarizer that write the plan/ solutions ..
    

    
