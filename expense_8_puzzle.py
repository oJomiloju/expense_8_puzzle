from collections import deque 
from heapq import heappush, heappop
import sys
import logging
'''
    1. will repreent the 3 x 3 grid as a list of lists.
    blank tile represented by
    2. There are four possible moves: Up, Down, Left, Right.
    3. Moving a Tile incurs the cost equivalent to the number on the tile
    4. Implementing a BFS algorithm to explore the nodes, tracking costs and states,
    and adjust for the minimum cost 
    5. read the start and goal states from files, and output the search results as specified
'''
'''
GOAL - Implement BFS - DONE 
       Implement UCS 
       Implement Greedy Search 
       Implement A* search 
'''

logging.basicConfig(filename="dump.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class PuzzleState:
    def __init__(self,board,blank_pos,depth=0,cost=0,path=None):
        self.board = board # 3 x 3 grid 
        self.blank_pos = blank_pos
        self.depth = depth
        self.cost = cost
        self.path = path if path is not None else [] # This stores the path of moves needed to take to reach this state

        # Method to generate a unique string representation for the state 
    def to_string(self):
        return ''.join(str(num) for row in self.board for num in row) # 123456780 (an example)
        
    # Method to generate all possible next states from the current state
    
    def generate_new_states(self):
        x, y = self.blank_pos  # current position of the blank tile
       

        directions = {
            'Down': (-1, 0),
            'Up': (1, 0),
            'Right': (0, -1),
            'Left': (0, 1)
        }

        new_states = []
        for direction, (dx, dy) in directions.items():
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_board = [row[:] for row in self.board]
                tile_moved = new_board[new_x][new_y]  # The tile value being moved into the blank position
                new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
                new_blank_pos = (new_x, new_y)
                new_cost = self.cost + tile_moved
                new_path = self.path + [(direction, tile_moved)]  # Append (direction, cost) to path
                new_state = PuzzleState(new_board, new_blank_pos, self.depth + 1, new_cost, new_path)
                new_states.append(new_state)

                

        return new_states
    
    # Comparison methods for priority queue
    def __gt__(self, other):
        return self.cost > other.cost
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, other):
        return self.cost == other.cost

# Function to read the grid from a file and turn it in to a list 
def read_grid(file_path):
    with open(file_path,'r') as file:
        grid = [list(map(int,line.split())) for line in file.readlines()]
    return grid 

# function to find the blank tile in the grid
def find_blank(grid):
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 0:
                return (i, j) # exact position of the blank tile

# Function to perform Breadth First Search
def bfs(start_grid, goal_grid,log_enabled=False):

    if log_enabled:
        logging.info("Method selected: BFS")
        logging.info("Running BFS")

    start_blank = find_blank(start_grid)
    goal_state = PuzzleState(goal_grid, find_blank(goal_grid))
    
    # initialize BFS Fringe (queue) and visited states since we are implementing a graph
    fringe = deque([PuzzleState(start_grid, start_blank)]) # start with the initial state in the queue 
    visited = set() # where all visited states will be stored 
    visited.add(fringe[0].to_string())

    nodes_popped = 0
    nodes_expanded = 0 
    nodes_generated = 0 
    max_fringe_size = 0

    # BFS LOOP 
    '''
    if fringe is empty then return failure 
    node <- Remove-Front(fringe)
    if Goal-Test(problem,STATE[node]) then return node 
    if STATE[node] not in visited then
        Add(STATE[node]) to visited 
        fringe <- Insert-All(Expand(problem, STATE[node]), fringe)
    '''
    while fringe: # loop goes on until the fringe is empty 
        max_fringe_size = max(max_fringe_size, len(fringe)) # update the max fringe size
        current_state = fringe.popleft() # remove the first element from the fringe (BFS IS FIFO)
        nodes_popped += 1

        if log_enabled:
            logging.info(f"Current state: {current_state.to_string()}")


        # CHECK IF GOAL STATE IS REACHED
        if current_state.to_string() == goal_state.to_string():
            print(f"Nodes Popped: {nodes_popped}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at Depth: {current_state.depth} with cost of {current_state.cost}")
            print("Solution Path:")
            for direction, cost in current_state.path:
                print(f"Move {cost} {direction}")  # Print the direction and the tile cost
            return
        
        # IF GOAL STATE IS NOT REACHED, EXPAND THE CURRENT STATE
        nodes_expanded += 1
        if log_enabled:
            logging.info("Visited so far:")
            logging.info(f" closed = [{visited}]")

        for new_state in current_state.generate_new_states():
            state_str = new_state.to_string()
            if state_str not in visited:
                fringe.append(new_state)
                visited.add(state_str)
                nodes_generated += 1

                if log_enabled:
                    count = 0
                    count+= 1
                    logging.info(f"Generated new state: {new_state.to_string()}")
        
        if log_enabled:
            logging.info(f"Current fringe size: {len(fringe)}")
            logging.info("Currrent fringe states:")
            for state in fringe:
                logging.info(f"{state.to_string()}, Move {state.path[-1][1]} {state.path[-1][0]}")

def dfs(start_grid, goal_grid,log_enabled=False):
    if log_enabled:
        logging.info("Method selected: DFS")
        logging.info("Running DFS")
    
    start_blank = find_blank(start_grid)
    goal_state = PuzzleState(goal_grid, find_blank(goal_grid))
    
    # initialize BFS Fringe (queue) and visited states since we are implementing a graph
    fringe = deque([PuzzleState(start_grid, start_blank)]) # start with the initial state in the queue 
    visited = set() # where all visited states will be stored 
    visited.add(fringe[0].to_string())

    nodes_popped = 0
    nodes_expanded = 0 
    nodes_generated = 0 
    max_fringe_size = 0

    # BFS LOOP 
    '''
    if fringe is empty then return failure 
    node <- Remove-Front(fringe)
    if Goal-Test(problem,STATE[node]) then return node 
    if STATE[node] not in visited then
        Add(STATE[node]) to visited 
        fringe <- Insert-All(Expand(problem, STATE[node]), fringe)
    '''
    while fringe: # loop goes on until the fringe is empty 
        max_fringe_size = max(max_fringe_size, len(fringe)) # update the max fringe size
        current_state = fringe.pop() # remove the last element from the fringe (DFS IS LIFO)
        nodes_popped += 1

        if log_enabled:
            logging.info(f"Current state: {current_state.to_string()}")


        # CHECK IF GOAL STATE IS REACHED
        if current_state.to_string() == goal_state.to_string():
            print(f"Nodes Popped: {nodes_popped}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at Depth: {current_state.depth} with cost of {current_state.cost}")
            print("Solution Path:")
            for direction, cost in current_state.path:
                print(f"Move {cost} {direction}")  # Print the direction and the tile cost
            return
        
        # IF GOAL STATE IS NOT REACHED, EXPAND THE CURRENT STATE
        nodes_expanded += 1
        if log_enabled:
            logging.info("Visited so far:")
            logging.info(f" closed = [{visited}]")

        for new_state in current_state.generate_new_states():
            state_str = new_state.to_string()
            if state_str not in visited:
                fringe.append(new_state)
                visited.add(state_str)
                nodes_generated += 1
            
                if log_enabled:
                    logging.info(f"Generated new state: {new_state.to_string()}")
        if log_enabled:
            logging.info(f"Current fringe size: {len(fringe)}")
            logging.info("Currrent fringe states:")
            for state in fringe:
                logging.info(f"{state.to_string()}, Move {state.path[-1][1]} {state.path[-1][0]}")


def ucs(start_grid, goal_grid,log_enabled=False):
    
    if log_enabled:
        logging.info("Method selected: UCS")
        logging.info("Running UCS")

    start_blank = find_blank(start_grid)
    goal_state = PuzzleState(goal_grid, find_blank(goal_grid))

    # Fringe is implemented as a priority queue
    fringe = [(0, PuzzleState(start_grid, start_blank))] # (cost, state)
    visited = set()
    visited.add(fringe[0][1].to_string()) # Add initial state to visited 

    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 0

    while fringe:
        max_fringe_size = max(max_fringe_size,len(fringe))
        current_cost, current_state = heappop(fringe) # Remove the state with the lowest cost from the fringe
        nodes_popped += 1

        if log_enabled:
            logging.info(f"Current state: {current_state.to_string()}, Cost: {current_cost}")


        # check if current state matches the goal state 
        if current_state.to_string() == goal_state.to_string():
            print(f"Nodes Popped: {nodes_popped}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at Depth: {current_state.depth} with cost of {current_state.cost}")
            print("Solution Path:")
            for direction, cost in current_state.path:
                print(f"Move {cost} {direction}")  # Print the direction and the tile cost
            return
        
        # Expand the current state
        nodes_expanded += 1
        if log_enabled:
            logging.info("Visited so far:")
            logging.info(f" closed = [{visited}]")
        for new_state in current_state.generate_new_states():
            state_str = new_state.to_string()
            if state_str not in visited:
                heappush(fringe, (new_state.cost, new_state)) # Add the new state to the fringe with its cost
                visited.add(state_str)
                nodes_generated += 1

                if log_enabled:
                    logging.info(f"Generated new state: {new_state.to_string()}, with Cost: {new_state.cost}")
        if log_enabled:
            logging.info(f"Current fringe size: {len(fringe)}")
            logging.info("Currrent fringe states:")
            for cost, state in fringe:
                logging.info(f"{state.to_string()}, Cost: {cost}, , action = Move {state.path[-1][1]} {state.path[-1][0]}")


    # If no solution is found
    print(f"Nodes Popped: {nodes_popped}")
    print(f"Nodes Expanded: {nodes_expanded}")
    print(f"Nodes Generated: {nodes_generated}")
    print(f"Max Fringe Size: {max_fringe_size}")
    print("No solution found")

def greedy_search(start_grid, goal_grid, log_enabled=False):

    if log_enabled:
        logging.info("Method selected: Greedy")
        logging.info("Running Greedy")


    start_blank = find_blank(start_grid)
    goal_state = PuzzleState(goal_grid, find_blank(goal_grid))

    # FRINGE IS A PRIORITY QUEUE BASED ON HEURISTIC VALUES 
    #fringe = [(misplaced_heuristic(PuzzleState(start_grid,start_blank),goal_state),PuzzleState(start_grid,start_blank))]
    fringe = [(manhattan_distance(PuzzleState(start_grid,start_blank),goal_state),PuzzleState(start_grid,start_blank))]
    visited = set()
    visited.add(fringe[0][1].to_string())

    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 0

    while fringe:
        max_fringe_size = max(max_fringe_size, len(fringe))
        current_heuristic, current_state = heappop(fringe)
        nodes_popped += 1
        
        if log_enabled:
            logging.info(f"Current state: {current_state.to_string()}, Heuristic: {current_heuristic}")

        if current_state.to_string() == goal_state.to_string():
            print(f"Nodes Popped: {nodes_popped}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at Depth: {current_state.depth} with cost of {current_state.cost}")
            print("Solution Path:")
            for direction, cost in current_state.path:
                print(f"Move {cost} {direction}")  # Print the direction and the tile cost
            return

        for new_state in current_state.generate_new_states():
            nodes_expanded += 1
            if log_enabled:
                logging.info("Visited so far:")
                logging.info(f" closed = [{visited}]")
            state_str = new_state.to_string()
            if state_str not in visited:
                heappush(fringe, (misplaced_heuristic(new_state, goal_state), new_state))
                visited.add(state_str)
                nodes_generated += 1 
                if log_enabled:
                    logging.info(f"Generated new state: {new_state.to_string()}, Heuristic: {misplaced_heuristic(new_state, goal_state)}")
        if log_enabled:
            logging.info(f"Current fringe size: {len(fringe)}")
            logging.info("Currrent fringe states:")
            for heuristic, state in fringe:
                logging.info(f"{state.to_string()}, Heuristic: {heuristic}, , action = Move {state.path[-1][1]} {state.path[-1][0]}")

    print("No solution found")  


def a_star_search(start_grid, goal_grid,log_enabled=False):

    if log_enabled:
        logging.info("Method selected: a*")
        logging.info("Running a*")

    start_blank = find_blank(start_grid)
    goal_state = PuzzleState(goal_grid, find_blank(goal_grid))

    # Fringe is a priority queue based on the sum of cost and heuristic value f(n) = h(n) + g(n)
    start_state = PuzzleState(start_grid, start_blank)
    fringe = [(manhattan_distance(start_state, goal_state), 0, PuzzleState(start_grid, start_blank))]
    #fringe = [(misplaced_heuristic(PuzzleState(start_grid, start_blank), goal_state), 0, PuzzleState(start_grid, start_blank))]
    visited = set()
    visited.add(fringe[0][2].to_string())

    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 0


    while fringe:
        max_fringe_size = max(max_fringe_size,len(fringe))    
        f_n, g_n, current_state = heappop(fringe)
        nodes_popped+= 1

        if log_enabled:
            logging.info(f"Popped node with state: {current_state.to_string()}, g(n) = {g_n}, h(n) = {manhattan_distance(current_state, goal_state)}, f(n) = {f_n}")

        # Check if the goal state is reached
        

        if current_state.to_string() == goal_state.to_string():
            print(f"Nodes Popped: {nodes_popped}")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Generated: {nodes_generated}")
            print(f"Max Fringe Size: {max_fringe_size}")
            print(f"Solution Found at Depth: {current_state.depth} with cost of {current_state.cost}")
            print("Solution Path:")
            for direction, cost in current_state.path:
                print(f"Move {cost} {direction}")  # Print the direction and the tile cost
            return
        
        nodes_expanded += 1

        if log_enabled:
            logging.info("Visited so far:")
            logging.info(f" closed = [{visited}]")

        for new_state in current_state.generate_new_states():
            state_str = new_state.to_string()
            if state_str not in visited:
                g_new = g_n + new_state.cost
                h_new = manhattan_distance(new_state, goal_state)
                f_new = g_new + h_new
                heappush(fringe, (f_new, g_new, new_state))
                visited.add(state_str)
                nodes_generated += 1
        
                if log_enabled:
                    logging.info(f"Generated new state: {new_state.to_string()}, g(n) = {g_new}, h(n) = {h_new}, f(n) = {f_new}")

        
        if log_enabled:
            logging.info(f"Fringe size after expansion: {len(fringe)}")
            logging.info("Current fringe states:")
            for f_value, g_value, state in fringe:
                logging.info(f"  State: {state.to_string()}, g(n) = {g_value}, h(n) = {manhattan_distance(state, goal_state)}, f(n) = {f_value}, action = Move {state.path[-1][1]} {state.path[-1][0]}")
        
    print(f"Nodes Popped: {nodes_popped}")
    print(f"Nodes Expanded: {nodes_expanded}")
    print(f"Nodes Generated: {nodes_generated}")
    print(f"Max Fringe Size: {max_fringe_size}")
    print("No solution found")
        



# DEVELOPING A HEURISTIC FOR A* AND GREEDY SEARCH USING MISPLACED TILES AND MANHATTAN DISTANCE 
def misplaced_heuristic(state, goal_state):
    start = [num for row in state.board for num in row] 
    goal = [num for row in goal_state.board for num in row]
    # For every tile out of place add 1 
    misplaced_count = sum(1 for i in range(9) if start[i] != goal[i] and start[i] != 0)
    return misplaced_count

'''
def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state.board[i][j]
            if tile != 0:  # Ignore the blank tile
                goal_x, goal_y = [(x, y) for x in range(3) for y in range(3) if goal_state.board[x][y] == tile][0]
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance
'''
def manhattan_distance(state, goal_state):
    # Create a mapping of the goal positions for each tile
    goal_positions = {}
    for i in range(3):
        for j in range(3):
            tile = goal_state.board[i][j]
            goal_positions[tile] = (i, j)

    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state.board[i][j]
            if tile != 0:  # Ignore the blank tile
                goal_x, goal_y = goal_positions[tile]
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python expense_8_puzzle.py <start.txt> <goal.txt> [<method>] [true]")
        sys.exit(1)

    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    
    # Default method is 'a*' if no method is provided
    method = 'a*' if len(sys.argv) < 4 else sys.argv[3].lower()  # Convert method to lowercase to handle case sensitivity

    # Check if logging is enabled (if 'true' is passed as the last argument)
    log_enabled = (len(sys.argv) == 5 and sys.argv[4].lower() == 'true')

    # Read the start and goal grids from files
    start_grid = read_grid(start_file)
    goal_grid = read_grid(goal_file)

    # Execute the appropriate search algorithm
    if method == 'bfs':
        print("Running Breadth-First Search (BFS)...")
        bfs(start_grid, goal_grid, log_enabled=log_enabled)
    elif method == 'ucs':
        print("Running Uniform Cost Search (UCS)...")
        ucs(start_grid, goal_grid, log_enabled=log_enabled)
    elif method == 'greedy':
        print("Running Greedy Search...")
        greedy_search(start_grid, goal_grid, log_enabled=log_enabled)
    elif method == 'dfs':
        print("Running Depth First Search...")
        dfs(start_grid, goal_grid, log_enabled=log_enabled)
    elif method == 'a*':
        print("Running A* Search...")
        a_star_search(start_grid, goal_grid, log_enabled=log_enabled)
    else:
        print("Invalid method specified. Supported methods are: bfs, ucs, greedy, dfs, a*.")
        sys.exit(1)
