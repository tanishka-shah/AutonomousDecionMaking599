# Autonomous Decision-Making

## Setup

The code requires [Anaconda](https://www.anaconda.com/download).

Please create a virtual environment before running the code (see documentation for [Visual Code](https://code.visualstudio.com/docs/python/environments))

To install all dependencies run the following commands in a terminal:
```
cd code
pip install -r requirements.txt
```

## Available Maps

All available maps are provided in the folder `code/layouts` and listed in the table below.

| Map   		| File                      |
|---------------|---------------------------|
| `easy_0`      | `code/layouts/easy_0.txt` |
| `easy_1`      | `code/layouts/easy_1.txt` |
| `medium_0`    | `code/layouts/medium_0.txt` |
| `medium_1`    | `code/layouts/medium_1.txt` |
| `hard_0`      | `code/layouts/hard_0.txt` |
| `hard_1`      | `code/layouts/hard_1.txt` |


## Usage

Run agent using the following commands in a terminal (`map-name` is provided in the "Map"-column of the table above):
```
cd code
python main.py <map-name>
```
After running the agent with provided map. A menu will be shown to select a bandit for the q-learning agent. According to my analysis Q-Learning agent works best with Epsilon-Greedy bandit. Epsilon Greedy and UCB1 bandits solve both medium maps and both hard maps.

The menu provides options as follows
1. Q-Learning agent with Epsion Greedy Bandit
2. Q-Learning agent with UCB1 Bandit
3. Q-Learning agent with Boltzmann Bandit
4. Compare all three listed above
