Expense 8 Puzzle Problem
Name: Oluwajomiloju Okuwobi
Programming Language: Python

Overview
This project involves building an agent to solve a modified version of the 8-puzzle problem using state space search and graph search to prevent repeated states. The goal is to transform a 3x3 grid containing tiles numbered 1 through 8 and a blank (represented by 0) into a desired configuration. The twist is that each tile's number represents the cost of moving that tile. This project implements several search algorithms to solve the puzzle efficiently.

Code Structure
Libraries Used:
deque from the collections library for implementing BFS and DFS.
heappush and heappop from the heapq library for priority queue operations.
sys for handling system-specific parameters and functions.
logging for debugging and tracking the search algorithms.
Class and Function Definitions
Class: PuzzleState
This class models the current state of the 8-puzzle.

Attributes:
board: A 3x3 grid representing the puzzle state (e.g., [[1,2,3], [4,5,6], [7,0,8]]).
blank_position: Tracks the location of the blank tile (0).
depth: The depth of the current node in the search tree.
cost: The cost of moving a certain tile.
path: Records the sequence of moves leading to the current state.
Methods:
to_string(): Converts the board into a string representation for comparison and hashing.
generate_new_states(): Generates all possible new states from the current state.
__gt__(), __lt__(), __eq__(): Comparison methods for priority queue operations based on cost or heuristic values.
Additional Functions
read_grid(filename): Reads the puzzle grid from a file and converts it into a list.
find_blank(grid): Identifies the position of the blank tile (0) in the grid.
Search Algorithms
Each algorithm is designed to search for a solution starting from the initial grid to the goal grid. They all accept the following arguments: start_grid, goal_grid, and an optional log_enabled flag for logging output to a file.

1. Breadth-First Search (BFS):

Uses a deque as the fringe.
Nodes are processed in FIFO order (using popleft()).

2. Depth-First Search (DFS):

Similar to BFS, but nodes are processed in LIFO order (stack-based).

3. Uniform Cost Search (UCS):

Uses the cost associated with each state.
Nodes are popped from the fringe based on their cumulative cost.

4. Greedy Best-First Search:

Utilizes the Manhattan distance heuristic.
Nodes are processed based on their lowest heuristic value.

5. A Search:*

Combines both the cost (g(n)) and heuristic (h(n)) to choose nodes with the lowest total f(n) = g(n) + h(n).
Uses the Manhattan distance heuristic for h(n).
Fringes for Greedy Best-First Search and A* Search are implemented as priority queues.

How to Run the Code
The code can be run from the command line with the following format:

python3 expense_8_puzzle.py <start_file> <goal_file> <algorithm> [log_enabled]

Alternatively:

python expense_8_puzzle.py <start_file> <goal_file> <algorithm> [log_enabled]

Supported Algorithms:
BFS: bfs
DFS: dfs
UCS: ucs
Greedy Best-First Search: greedy
A Search*: "a*" (quotations required)

python3 expense_8_puzzle.py start.txt goal.txt bfs
python3 expense_8_puzzle.py start.txt goal.txt "a*"


Input Files:
start.txt: File containing the initial puzzle configuration.
goal.txt: File containing the goal puzzle configuration.
Optional Flags:
log_enabled: If set to True, logs the output (including node cost, depth, and f(n) values) to dump.log for debugging purposes.
Default values:
algorithm: "a*"
log_enabled: False
Directory Structure
expense_8_puzzle.py: Main program file.
readme.md: This README file.
start.txt: File containing the initial puzzle state.
goal.txt: File containing the goal puzzle state.
dump.log: Log file for debugging (generated when log_enabled=True).

