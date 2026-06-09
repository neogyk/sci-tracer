from typing import Any, Dict, Literal, TypedDict


class Step2Result(TypedDict):
    graph_model: Literal["path-forest"]
    multiplicativity: bool


def step_2(context: Dict[str, Any]) -> Step2Result:
    """
    Translate the cousin-counting problem into perfect matchings on a disjoint union of paths
    (a path-forest) and assert multiplicativity across components.

    Precondition:
      - From step_1: S is a disjoint union of singleton and doubleton blocks on Z.
      - Define N(S) = {s-1, s+1 : s in S}. Build a bipartite graph with left part S and right part N(S),
        and edges between s in S and t in N(S) if |s - t| = 1.

    Claim:
      - This graph is a disjoint union of paths ordered along Z (a path-forest).
      - A perfect matching saturating S corresponds bijectively to a valid cousin set T (disjoint from S),
        giving the desired pairing with |s - t| = 1.
      - Because the graph is a disjoint union of components, the total number of such matchings
        (hence the number of cousin sets) is the product of the counts over its connected components
        (multiplicativity).

    Returns:
      - graph_model: "path-forest" to indicate the structural model.
      - multiplicativity: True to indicate independent component-wise multiplication of counts.
    """
    return Step2Result(graph_model="path-forest", multiplicativity=True)