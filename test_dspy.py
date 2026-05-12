import dspy

lm = dspy.LM(
            'ollama_chat/lfm2.5-thinking:1.2b',  # Replace with your specific model name
            api_base='http://localhost:11434',
            #logprobs=True,
            #max_tokens=100
            )

dspy.configure(lm=lm)
prompt = "What is the meaning of non-sense"


#Test Convert the Paper2Hoare Triplets:
prompt = """Natural Gradient Descent, a second-degree optimization method motivated by
the information geometry, makes use of the Fisher Information Matrix instead of
the Hessian which is typically used. However, in many cases, the Fisher Information Matrix is equivalent to the Generalized Gauss-Newton Method, that both
approximate the Hessian. It is an appealing method to be used as an alternative to
stochastic gradient descent, potentially leading to faster convergence. However, being a second-order method makes it infeasible to be used directly in problems with
a huge number of parameters and data. This is evident from the community of deep
learning sticking with the stochastic gradient descent method since the beginning.
In this paper, we look at the different perspectives on the natural gradient method,
study the current developments on its efficient-scalable empirical approximations,
and finally examine their performance with extensive experiments."""
instruction = "Convert this paragram in the instruction level triplets: Preposition , Conditioning, Prediction: "
#Test Convert the Paper2Hoare Triplets:
result = lm(instruction + prompt, temperature=0.7)
last_interaction = lm.history[-1]
print("result: ", result)

#Test Convert the Paper2Hoare Triplets:


