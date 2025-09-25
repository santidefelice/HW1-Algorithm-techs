"""
Homework 1 Solutions
Authors: Santiago Defelice and Santiago Rubero
Course: Algorithm Techniques
Date: 9/24/2025



"""

from typing import List

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