# =============================================================================
# QUESTION 3: EIGHT PUZZLE GAME
# =============================================================================

from typing import List
from collections import deque
import heapq
from typing import List, Tuple, Optional

def solve_eight_puzzle(initial_state: List[List[int]], goal_state: List[List[int]]) -> List[List[List[int]]]:
    """

    Eight Puzzle Game

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


def test_question3():
    """Test cases for Eight Puzzle Game"""
    print("\n" + "=" * 60)
    print("QUESTION 3: EIGHT PUZZLE GAME - TEST CASES")
    print("=" * 60)
    
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
        print(f"Solution found in {len(solution3) - 1} moves")
        print("(Showing first few steps due to length)")
        for i, state in enumerate(solution3[:min(4, len(solution3))]):
            print(f"Step {i}:")
            print_grid(state)
        if len(solution3) > 4:
            print(f"... {len(solution3) - 4} more steps to reach goal")
    else:
        print("No solution found")

def main():
    """Main function to run all test cases"""
    print("=" * 60)
    
    # Run all test cases
    test_question3()
    
    print("\n" + "=" * 60)
    print("ALL TEST CASES COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()