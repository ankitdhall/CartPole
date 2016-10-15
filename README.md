# CartPole
Solving CartPole environment on OpenAI Gym using a linear approx Q-function using TensorFlow.

The algorithm learns a single 4x2 transformation matrix to map observed state values to Q-value approximation for actions.
M: states -> Q-value approximation for actions

The linear mapping solves both the [v0](https://gym.openai.com/evaluations/eval_JJQ797NXTs2VQk8oxREGhg) and [v1](https://gym.openai.com/evaluations/eval_PgCVjqrTCObX2gHpEQdeQ) of CartPole and the evaluations can be found on the OpenAI gym.
