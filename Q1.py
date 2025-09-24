"""
Homework 1 Solutions
Authors: Santiago Defelice and Santiago Rubero
Course: Algorithm Techniques
Date: 9/24/2025



"""

from collections import deque
import heapq
from typing import List, Tuple, Optional

# =============================================================================
# QUESTION 1: NUMBER OF ISLANDS
# =============================================================================

def num_islands(grid: List[List[str]]) -> int:
    """
    Number of Islands
    
    PROBLEM ANALYSIS:
    This is a classic connected components problem. We need to find all connected
    regions of '1's (land) in the grid. We can use DFS or BFS to explore each
    island completely when we encounter an unvisited land cell.
    
    ALGORITHM:
    1. Iterate through each cell in the grid
    2. When we find a '1' that hasn't been visited, increment island count
    3. Use DFS to mark all connected '1's as visited
    4. Continue until all cells are processed
    
    COMPLEXITY ANALYSIS:
    - Time Complexity: O(m × n) where m is rows and n is columns
      Each cell is visited at most once during the DFS traversal
    - Space Complexity: O(min(m, n)) for the recursive call stack in worst case
      In the worst case (all cells are land), the recursion depth can be m × n,
      but typically it's much smaller
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r: int, c: int) -> None:
        """Mark all connected land cells as visited"""
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0'):
            return
        
        # Mark as visited by changing to '0'
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r + 1, c)  # down
        dfs(r - 1, c)  # up
        dfs(r, c + 1)  # right
        dfs(r, c - 1)  # left
    
    # Check each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)  # Mark entire island as visited
    
    return islands

# =============================================================================
# QUESTION 2: SHORTEST PATH IN A GRID WITH OBSTACLE ELIMINATION
# =============================================================================

def shortest_path_with_elimination(grid: List[List[int]], k: int) -> int:
    """
    PROBLEM ANALYSIS:
    This is a shortest path problem with a twist - we can eliminate obstacles.
    We need to track not just position, but also how many eliminations we've used.
    This creates a 3D state space: (row, col, eliminations_used).
    
    ALGORITHM:
    1. Use BFS since we want the shortest path (minimum steps)
    2. State: (row, col, eliminations_used, steps)
    3. For each position, we can move to adjacent cells if:
       - Cell is empty (value 0), or
       - Cell is obstacle (value 1) and we have eliminations left
    4. Track visited states to avoid cycles
    
    COMPLEXITY ANALYSIS:
    - Time Complexity: O(m × n × k) where m×n is grid size, k is max eliminations
      Each cell can be visited at most k+1 times (with 0 to k eliminations used)
    - Space Complexity: O(m × n × k) for the visited set and BFS queue
    """
    if not grid or not grid[0]:
        return -1
    
    rows, cols = len(grid), len(grid[0])
    
    # Edge case: start or end is blocked and no eliminations
    if grid[0][0] == 1 and k == 0:
        return -1
    if grid[rows-1][cols-1] == 1 and k == 0:
        return -1
    
    # BFS: (row, col, eliminations_used, steps)
    queue = deque([(0, 0, 0, 0)])  # Start at (0,0) with 0 eliminations and 0 steps
    visited = set()
    visited.add((0, 0, 0))  # (row, col, eliminations_used)
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    
    while queue:
        row, col, eliminations_used, steps = queue.popleft()
        
        # Check if we reached the destination
        if row == rows - 1 and col == cols - 1:
            return steps
        
        # Try all 4 directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
                continue
            
            new_eliminations = eliminations_used
            
            # If the cell is an obstacle, we need to eliminate it
            if grid[new_row][new_col] == 1:
                if eliminations_used >= k:
                    continue  # Can't eliminate more obstacles
                new_eliminations = eliminations_used + 1
            
            state = (new_row, new_col, new_eliminations)
            if state not in visited:
                visited.add(state)
                queue.append((new_row, new_col, new_eliminations, steps + 1))
    
    return -1  # No path found

# =============================================================================
# QUESTION 3: EIGHT PUZZLE GAME
# =============================================================================

def solve_eight_puzzle(initial_state: List[List[int]], goal_state: List[List[int]]) -> List[List[List[int]]]:
    """
    PROBLEM ANALYSIS:
    This is a state-space search problem where each state is a 3×3 grid configuration.
    We need to find the shortest sequence of moves from initial to goal state.
    We'll use A* search with Manhattan distance heuristic for efficiency.
    
    ALGORITHM:
    1. Use A* search with priority queue
    2. State representation: flatten 3×3 grid to tuple for hashing
    3. Heuristic: Manhattan distance (sum of distances each tile is from goal position)
    4. Generate successor states by moving tiles into empty space
    5. Track path to reconstruct solution
    
    COMPLEXITY ANALYSIS:
    - Time Complexity: O(b^d) where b is branching factor (~3) and d is solution depth
      In practice, much better due to A* heuristic pruning
    - Space Complexity: O(b^d) for storing states in priority queue and visited set
    """
    
    def state_to_tuple(state: List[List[int]]) -> tuple:
        """Convert 2D state to tuple for hashing"""
        return tuple(tuple(row) for row in state)
    
    def tuple_to_state(state_tuple: tuple) -> List[List[int]]:
        """Convert tuple back to 2D state"""
        return [list(row) for row in state_tuple]
    
    def find_empty_pos(state: List[List[int]]) -> Tuple[int, int]:
        """Find position of empty cell (0)"""
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return (i, j)
        return (-1, -1)
    
    def manhattan_distance(state: List[List[int]], goal: List[List[int]]) -> int:
        """Calculate Manhattan distance heuristic"""
        distance = 0
        goal_pos = {}
        
        # Map each number to its goal position
        for i in range(3):
            for j in range(3):
                if goal[i][j] != 0:
                    goal_pos[goal[i][j]] = (i, j)
        
        # Calculate distance for each tile
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    goal_i, goal_j = goal_pos[state[i][j]]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        
        return distance
    
    def get_neighbors(state: List[List[int]]) -> List[List[List[int]]]:
        """Generate all possible next states"""
        neighbors = []
        empty_i, empty_j = find_empty_pos(state)
        
        # Possible moves: up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for di, dj in moves:
            new_i, new_j = empty_i + di, empty_j + dj
            
            # Check bounds
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                # Create new state by swapping empty cell with adjacent cell
                new_state = [row[:] for row in state]  # Deep copy
                new_state[empty_i][empty_j], new_state[new_i][new_j] = \
                    new_state[new_i][new_j], new_state[empty_i][empty_j]
                neighbors.append(new_state)
        
        return neighbors
    
    # A* search implementation
    initial_tuple = state_to_tuple(initial_state)
    goal_tuple = state_to_tuple(goal_state)
    
    if initial_tuple == goal_tuple:
        return [initial_state]
    
    # Priority queue: (f_score, g_score, state_tuple, path)
    heap = [(manhattan_distance(initial_state, goal_state), 0, initial_tuple, [initial_state])]
    visited = set([initial_tuple])
    
    while heap:
        f_score, g_score, current_tuple, path = heapq.heappop(heap)
        current_state = tuple_to_state(current_tuple)
        
        # Generate neighbors
        for neighbor_state in get_neighbors(current_state):
            neighbor_tuple = state_to_tuple(neighbor_state)
            
            if neighbor_tuple == goal_tuple:
                return path + [neighbor_state]
            
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                new_g_score = g_score + 1
                h_score = manhattan_distance(neighbor_state, goal_state)
                new_f_score = new_g_score + h_score
                
                heapq.heappush(heap, (new_f_score, new_g_score, neighbor_tuple, path + [neighbor_state]))
    
    return []  # No solution found

# =============================================================================
# TEST CASES AND MAIN FUNCTION
# =============================================================================

def print_grid(grid: List[List], title: str = ""):
    """Helper function to print grids nicely"""
    if title:
        print(f"\n{title}:")
    for row in grid:
        print(row)

def test_question1():
    """Test cases for Number of Islands"""
    print("=" * 60)
    print("QUESTION 1: NUMBER OF ISLANDS - TEST CASES")
    print("=" * 60)
    
    # Test Case 1: Example from problem
    print("\n--- Test Case 1 ---")
    grid1 = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    print_grid(grid1, "Input Grid")
    result1 = num_islands([row[:] for row in grid1])  # Deep copy to preserve original
    print(f"Number of islands: {result1}")
    print("Expected: 3")
    
    # Test Case 2: No islands
    print("\n--- Test Case 2 ---")
    grid2 = [
        ["0","0","0"],
        ["0","0","0"],
        ["0","0","0"]
    ]
    print_grid(grid2, "Input Grid")
    result2 = num_islands([row[:] for row in grid2])
    print(f"Number of islands: {result2}")
    print("Expected: 0")
    
    # Test Case 3: One large island
    print("\n--- Test Case 3 ---")
    grid3 = [
        ["1","1","1"],
        ["1","0","1"],
        ["1","1","1"]
    ]
    print_grid(grid3, "Input Grid")
    result3 = num_islands([row[:] for row in grid3])
    print(f"Number of islands: {result3}")
    print("Expected: 1")

    
    # Test Case 1: Simple one-move solution
    print("\n--- Test Case 1 ---")
    initial1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 0, 8]
    ]
    goal1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    print_grid(initial1, "Initial State")
    print_grid(goal1, "Goal State")
    
    solution1 = solve_eight_puzzle(initial1, goal1)
    if solution1:
        print(f"Solution found in {len(solution1) - 1} moves:")
        for i, state in enumerate(solution1):
            print(f"Step {i}:")
            print_grid(state)
    else:
        print("No solution found")
    
    # Test Case 2: Already solved
    print("\n--- Test Case 2 ---")
    initial2 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    goal2 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    print_grid(initial2, "Initial State")
    print_grid(goal2, "Goal State")
    
    solution2 = solve_eight_puzzle(initial2, goal2)
    print(f"Solution: Already solved! Steps required: {len(solution2) - 1}")
    
    # Test Case 3: More complex puzzle
    print("\n--- Test Case 3 ---")
    initial3 = [
        [1, 2, 3],
        [4, 6, 8],
        [7, 0, 5]
    ]
    goal3 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    print_grid(initial3, "Initial State")
    print_grid(goal3, "Goal State")
    
    solution3 = solve_eight_puzzle(initial3, goal3)
    if solution3:
        print(f"Solution found in {len(solution3) - 1} moves:")
        for i, state in enumerate(solution3):
            print(f"Step {i}:")
            print_grid(state)
    else:
        print("No solution found")

def main():
    """Main function to run all test cases"""
    print("=" * 60)
    
    # Run all test cases
    test_question1()
    
    print("\n" + "=" * 60)
    print("ALL TEST CASES COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()