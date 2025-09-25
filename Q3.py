# =============================================================================
# QUESTION 3: EIGHT PUZZLE GAME - ENHANCED WITH FOUR ALGORITHMS
# =============================================================================

from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import heapq
import time
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Class to store search algorithm results"""
    path: List[List[List[int]]]
    nodes_explored: int
    path_length: int
    runtime: float
    algorithm_name: str

class EightPuzzleSolver:
    """Enhanced Eight Puzzle Solver with four different algorithms"""

    """
    Enhanced Eight Puzzle Solver with Four Search Algorithms

    PROBLEM ANALYSIS:
    This is a state-space search problem where each state represents a 3×3 grid configuration.
    The goal is to find the shortest sequence of moves from an initial state to the goal state.
    We implement four different search algorithms to compare their performance:
    BFS, DFS, Dijkstra's algorithm, and A* with Manhattan distance heuristic.

    ALGORITHMS:
    1. BFS (Breadth-First Search): 
    - Guarantees optimal solution by exploring states level by level
    - Uses queue (FIFO) to process states
    - Time: O(b^d), Space: O(b^d) where b=branching factor, d=depth

    2. DFS (Depth-First Search):
    - Uses stack (LIFO) with depth limit to prevent infinite loops
    - May not find optimal solution but uses less memory
    - Time: O(b^m), Space: O(bm) where m=maximum depth

    3. Dijkstra's Algorithm (Uniform Cost Search):
    - Guarantees optimal solution for weighted graphs (here all moves cost 1)
    - Uses priority queue ordered by path cost
    - Time: O((V+E)logV), Space: O(V) where V=states, E=transitions

    4. A* Search:
    - Uses Manhattan distance heuristic to guide search toward goal
    - Combines actual cost (g) + heuristic cost (h) = f(n) = g(n) + h(n)
    - Guarantees optimal solution if heuristic is admissible
    - Most efficient for this problem due to informed search

    STATE REPRESENTATION:
    - Each state is a 3×3 grid flattened to tuple for hashing
    - Empty cell represented by 0, numbered tiles 1-8
    - Transitions: swap empty cell with adjacent numbered tiles

    PERFORMANCE METRICS:
    - Runtime: Execution time for each algorithm
    - Path length: Number of moves in solution
    - Nodes explored: Number of states examined during search
    - Efficiency: (path_length / nodes_explored) × 100%

    COMPLEXITY ANALYSIS:
    - State space size: 9!/2 = 181,440 reachable states (half of 9! due to parity)
    - Branching factor: ~2.67 average (empty cell has 2-4 possible moves)
    - A* typically explores fewest nodes due to heuristic guidance
    - BFS/Dijkstra explore similar nodes (uniform cost), guarantee optimality
    - DFS may find suboptimal solutions but uses less memory
    """
    
    def __init__(self):
        self.goal_state = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]
    
    def state_to_tuple(self, state: List[List[int]]) -> tuple:
        """Convert 2D state to tuple for hashing"""
        return tuple(tuple(row) for row in state)
    
    def tuple_to_state(self, state_tuple: tuple) -> List[List[int]]:
        """Convert tuple back to 2D state"""
        return [list(row) for row in state_tuple]
    
    def find_empty_pos(self, state: List[List[int]]) -> Tuple[int, int]:
        """Find position of empty cell (0)"""
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return (i, j)
        return (-1, -1)
    
    def manhattan_distance(self, state: List[List[int]]) -> int:
        """Calculate Manhattan distance heuristic for A*"""
        distance = 0
        goal_pos = {}
        
        # Map each number to its goal position
        for i in range(3):
            for j in range(3):
                if self.goal_state[i][j] != 0:
                    goal_pos[self.goal_state[i][j]] = (i, j)
        
        # Calculate distance for each tile
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    goal_i, goal_j = goal_pos[state[i][j]]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        
        return distance
    
    def get_neighbors(self, state: List[List[int]]) -> List[List[List[int]]]:
        """Generate all possible next states"""
        neighbors = []
        empty_i, empty_j = self.find_empty_pos(state)
        
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
    
    def reconstruct_path(self, came_from: Dict, current_tuple: tuple) -> List[List[List[int]]]:
        """Reconstruct path from came_from dictionary"""
        path = []
        while current_tuple is not None:
            path.append(self.tuple_to_state(current_tuple))
            current_tuple = came_from.get(current_tuple)
        return path[::-1]
    
    def bfs(self, initial_state: List[List[int]]) -> SearchResult:
        """Breadth-First Search implementation"""
        start_time = time.time()
        
        initial_tuple = self.state_to_tuple(initial_state)
        goal_tuple = self.state_to_tuple(self.goal_state)
        
        if initial_tuple == goal_tuple:
            return SearchResult([initial_state], 0, 0, time.time() - start_time, "BFS")
        
        queue = deque([initial_tuple])
        visited = {initial_tuple}
        came_from = {initial_tuple: None}
        nodes_explored = 0
        
        while queue:
            current_tuple = queue.popleft()
            current_state = self.tuple_to_state(current_tuple)
            nodes_explored += 1
            
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = self.state_to_tuple(neighbor_state)
                
                if neighbor_tuple == goal_tuple:
                    came_from[neighbor_tuple] = current_tuple
                    path = self.reconstruct_path(came_from, neighbor_tuple)
                    return SearchResult(
                        path, nodes_explored, len(path) - 1, 
                        time.time() - start_time, "BFS"
                    )
                
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    came_from[neighbor_tuple] = current_tuple
                    queue.append(neighbor_tuple)
        
        return SearchResult([], nodes_explored, -1, time.time() - start_time, "BFS")
    
    def dfs(self, initial_state: List[List[int]], max_depth: int = 20) -> SearchResult:
        """Depth-First Search implementation with depth limit"""
        start_time = time.time()
        
        initial_tuple = self.state_to_tuple(initial_state)
        goal_tuple = self.state_to_tuple(self.goal_state)
        
        if initial_tuple == goal_tuple:
            return SearchResult([initial_state], 0, 0, time.time() - start_time, "DFS")
        
        stack = [(initial_tuple, 0)]
        visited = set()
        came_from = {initial_tuple: None}
        nodes_explored = 0
        
        while stack:
            current_tuple, depth = stack.pop()
            
            if current_tuple in visited or depth > max_depth:
                continue
                
            visited.add(current_tuple)
            current_state = self.tuple_to_state(current_tuple)
            nodes_explored += 1
            
            if current_tuple == goal_tuple:
                path = self.reconstruct_path(came_from, current_tuple)
                return SearchResult(
                    path, nodes_explored, len(path) - 1,
                    time.time() - start_time, "DFS"
                )
            
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = self.state_to_tuple(neighbor_state)
                
                if neighbor_tuple not in visited:
                    came_from[neighbor_tuple] = current_tuple
                    stack.append((neighbor_tuple, depth + 1))
        
        return SearchResult([], nodes_explored, -1, time.time() - start_time, "DFS")
    
    def dijkstra(self, initial_state: List[List[int]]) -> SearchResult:
        """Dijkstra's algorithm implementation (uniform cost search)"""
        start_time = time.time()
        
        initial_tuple = self.state_to_tuple(initial_state)
        goal_tuple = self.state_to_tuple(self.goal_state)
        
        if initial_tuple == goal_tuple:
            return SearchResult([initial_state], 0, 0, time.time() - start_time, "Dijkstra")
        
        # Priority queue: (cost, state_tuple)
        heap = [(0, initial_tuple)]
        visited = set()
        came_from = {initial_tuple: None}
        g_score = {initial_tuple: 0}
        nodes_explored = 0
        
        while heap:
            current_cost, current_tuple = heapq.heappop(heap)
            
            if current_tuple in visited:
                continue
                
            visited.add(current_tuple)
            nodes_explored += 1
            
            if current_tuple == goal_tuple:
                path = self.reconstruct_path(came_from, current_tuple)
                return SearchResult(
                    path, nodes_explored, len(path) - 1,
                    time.time() - start_time, "Dijkstra"
                )
            
            current_state = self.tuple_to_state(current_tuple)
            
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = self.state_to_tuple(neighbor_state)
                tentative_g_score = g_score[current_tuple] + 1
                
                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current_tuple
                    g_score[neighbor_tuple] = tentative_g_score
                    heapq.heappush(heap, (tentative_g_score, neighbor_tuple))
        
        return SearchResult([], nodes_explored, -1, time.time() - start_time, "Dijkstra")
    
    def astar(self, initial_state: List[List[int]]) -> SearchResult:
        """A* search with Manhattan distance heuristic"""
        start_time = time.time()
        
        initial_tuple = self.state_to_tuple(initial_state)
        goal_tuple = self.state_to_tuple(self.goal_state)
        
        if initial_tuple == goal_tuple:
            return SearchResult([initial_state], 0, 0, time.time() - start_time, "A*")
        
        # Priority queue: (f_score, g_score, state_tuple)
        heap = [(self.manhattan_distance(initial_state), 0, initial_tuple)]
        visited = set()
        came_from = {initial_tuple: None}
        g_score = {initial_tuple: 0}
        nodes_explored = 0
        
        while heap:
            f_score, current_g, current_tuple = heapq.heappop(heap)
            
            if current_tuple in visited:
                continue
                
            visited.add(current_tuple)
            nodes_explored += 1
            
            if current_tuple == goal_tuple:
                path = self.reconstruct_path(came_from, current_tuple)
                return SearchResult(
                    path, nodes_explored, len(path) - 1,
                    time.time() - start_time, "A*"
                )
            
            current_state = self.tuple_to_state(current_tuple)
            
            for neighbor_state in self.get_neighbors(current_state):
                neighbor_tuple = self.state_to_tuple(neighbor_state)
                tentative_g_score = g_score[current_tuple] + 1
                
                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current_tuple
                    g_score[neighbor_tuple] = tentative_g_score
                    h_score = self.manhattan_distance(neighbor_state)
                    f_score = tentative_g_score + h_score
                    heapq.heappush(heap, (f_score, tentative_g_score, neighbor_tuple))
        
        return SearchResult([], nodes_explored, -1, time.time() - start_time, "A*")
    
    def solve_all_algorithms(self, initial_state: List[List[int]]) -> List[SearchResult]:
        """Solve using all four algorithms and return results"""
        algorithms = [
            ("BFS", self.bfs),
            ("DFS", self.dfs),
            ("Dijkstra", self.dijkstra),
            ("A*", self.astar)
        ]
        
        results = []
        for name, algorithm in algorithms:
            print(f"Running {name}...")
            result = algorithm(initial_state)
            results.append(result)
            
            if result.path_length >= 0:
                print(f"  ✓ Solution found: {result.path_length} steps, "
                      f"{result.nodes_explored} nodes explored, "
                      f"{result.runtime:.4f}s")
            else:
                print(f"  ✗ No solution found, "
                      f"{result.nodes_explored} nodes explored, "
                      f"{result.runtime:.4f}s")
        
        return results
    
    def compare_algorithms(self, results: List[SearchResult]):
        """Compare and analyze algorithm performance"""
        print("\n" + "=" * 80)
        print("ALGORITHM PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Filter successful results
        successful_results = [r for r in results if r.path_length >= 0]
        
        if not successful_results:
            print("No algorithms found a solution.")
            return
        
        print(f"{'Algorithm':<12} {'Steps':<8} {'Nodes':<10} {'Runtime(ms)':<12} {'Efficiency':<12}")
        print("-" * 80)
        
        for result in results:
            if result.path_length >= 0:
                efficiency = (result.path_length / result.nodes_explored * 100) if result.nodes_explored > 0 else 0
                print(f"{result.algorithm_name:<12} {result.path_length:<8} "
                      f"{result.nodes_explored:<10} {result.runtime*1000:<12.2f} {efficiency:<12.2f}%")
            else:
                print(f"{result.algorithm_name:<12} {'N/A':<8} "
                      f"{result.nodes_explored:<10} {result.runtime*1000:<12.2f} {'N/A':<12}")
        
        # Analysis
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        
        if successful_results:
            # Best by different metrics
            best_steps = min(successful_results, key=lambda x: x.path_length)
            best_nodes = min(successful_results, key=lambda x: x.nodes_explored)
            best_time = min(successful_results, key=lambda x: x.runtime)
            
            print(f"Optimal solution length: {best_steps.path_length} steps")
            print(f"Most efficient (fewest nodes explored): {best_nodes.algorithm_name} ({best_nodes.nodes_explored} nodes)")
            print(f"Fastest runtime: {best_time.algorithm_name} ({best_time.runtime*1000:.2f}ms)")
            
            # Check optimality
            optimal_length = best_steps.path_length
            optimal_algorithms = [r.algorithm_name for r in successful_results if r.path_length == optimal_length]
            print(f"Algorithms finding optimal solution: {', '.join(optimal_algorithms)}")


def solve_eight_puzzle(initial_state: List[List[int]], goal_state: List[List[int]]) -> List[List[List[int]]]:
    """
    Original function signature maintained for compatibility
    Uses A* algorithm (best performing) by default
    """
    solver = EightPuzzleSolver()
    solver.goal_state = goal_state
    result = solver.astar(initial_state)
    return result.path


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
    """Test cases for Eight Puzzle Game with all four algorithms"""
    print("\n" + "=" * 60)
    print("QUESTION 3: EIGHT PUZZLE GAME - FOUR ALGORITHMS COMPARISON")
    print("=" * 60)
    
    solver = EightPuzzleSolver()
    
    # Test Case 1: Simple puzzle (few moves)
    print("\n--- Test Case 1: Simple Puzzle ---")
    initial1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 0, 8]
    ]
    print_grid(initial1, "Initial State")
    print_grid(solver.goal_state, "Goal State")
    
    results1 = solver.solve_all_algorithms(initial1)
    solver.compare_algorithms(results1)
    
    # Test Case 2: More complex puzzle
    print("\n--- Test Case 2: Complex Puzzle ---")
    initial2 = [
        [1, 2, 3],
        [4, 6, 8],
        [7, 0, 5]
    ]
    print_grid(initial2, "Initial State")
    print_grid(solver.goal_state, "Goal State")
    
    results2 = solver.solve_all_algorithms(initial2)
    solver.compare_algorithms(results2)
    
    # Test Case 3: Even more complex puzzle
    print("\n--- Test Case 3: Very Complex Puzzle ---")
    initial3 = [
        [2, 6, 3],
        [1, 8, 4],
        [7, 0, 5]
    ]
    print_grid(initial3, "Initial State")
    print_grid(solver.goal_state, "Goal State")
    
    results3 = solver.solve_all_algorithms(initial3)
    solver.compare_algorithms(results3)


def main():
    """Main function to run all test cases"""
    print("=" * 60)
    print("EIGHT PUZZLE SOLVER - FOUR ALGORITHMS")
    print("BFS, DFS, Dijkstra's Algorithm, and A*")
    print("=" * 60)
    
    # Run all test cases
    test_question3()
    
    print("\n" + "=" * 60)
    print("ALL TEST CASES COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()