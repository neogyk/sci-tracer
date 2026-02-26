from dspy.evaluate import Evaluate

# 1. Your Devset (List of dspy.Example objects)
devset = [
    dspy.Example(
        question="How does CommCP improve coordination?",
        answer="By using conformal prediction to calibrate messages.",
    ).with_inputs("question"),
    # ... add more examples
]

# 2. Configure the Evaluator
evaluator = Evaluate(
    devset=devset,
    metric=qarag_metric,
    num_threads=4,
    display_progress=True,
    display_table=5,  # Shows the first 5 results in a pretty table
)

# 3. Run it on your Multi-Agent Program
# agent_system = YourMultiAgentModule()
results = evaluator(agent_system)
print(f"Overall Score: {results}")
