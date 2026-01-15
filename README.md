# Connect4-RL-Bot

An agent trained using Monte Carlo Tree Search based on an Upper Confidence Bound Policy to play Connect4. 
In this code, the agent can play against a human player or a random agent. With further modification, it can also play with other agents.
The agent is constructed such that it updates the Monte Carlo Tree used to make moves upon encountering new board configurations as it plays against an external agent, so it learns while playing.

# MCTS trees

mcts_tree_10: The tree developed from the agent playing 10 games with itself 

mcts_tree_25: The tree developed from the agent playing 25 games with itself

mcts_tree_50: The tree developed from the agent playing 50 games with itself

mcts_tree: mcts_tree_50 updated over a few additional games against a human
