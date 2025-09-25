# =============================================================================
# QUESTION 2: SHORTEST PATH IN A GRID WITH OBSTACLE ELIMINATION
# =============================================================================

from typing import List
from collections import deque

def shortest_path_with_elimination(grid: List[List[int]], k: int) -> int:
  
    """

    Shortest Path in a Grid with Obstacle Elimination
      
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


def print_grid(grid: List[List], title: str = ""):
    """Helper function to print grids nicely"""
    if title:
        print(f"\n{title}:")
    for row in grid:
        print(row)


def test_question2():
    """Test cases for Shortest Path with Obstacle Elimination"""
    print("\n" + "=" * 60)
    print("QUESTION 2: SHORTEST PATH WITH OBSTACLE ELIMINATION - TEST CASES")
    print("=" * 60)
    
    # Test Case 1: Example 1 from problem
    print("\n--- Test Case 1 ---")
    grid1 = [
        [0,0,0],
        [1,1,0],
        [0,0,0],
        [0,1,1],
        [0,0,0]
    ]
    k1 = 1
    print_grid(grid1, "Input Grid")
    print(f"k = {k1}")
    result1 = shortest_path_with_elimination(grid1, k1)
    print(f"Shortest path length: {result1}")
    
    
    # Test Case 2: Example 2 from problem
    print("\n--- Test Case 2 ---")
    grid2 = [
        [0,1,1],
        [1,1,1],
        [1,0,0]
    ]
    k2 = 1
    print_grid(grid2, "Input Grid")
    print(f"k = {k2}")
    result2 = shortest_path_with_elimination(grid2, k2)
    print(f"Shortest path length: {result2}")
    
    
    # Test Case 3: Simple path with no obstacles
    print("\n--- Test Case 3 ---")
    grid3 = [
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ]
    k3 = 0
    print_grid(grid3, "Input Grid")
    print(f"k = {k3}")
    result3 = shortest_path_with_elimination(grid3, k3)
    print(f"Shortest path length: {result3}")
    

def main():
    """Main function to run all test cases"""
    print("=" * 60)
    
    # Run all test cases
    test_question2()
    
    print("\n" + "=" * 60)
    print("ALL TEST CASES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()