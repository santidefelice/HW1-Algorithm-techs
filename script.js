class PuzzleGame {
    constructor() {
        this.size = 3;
        this.goalState = [1, 2, 3, 4, 5, 6, 7, 8, 0];
        this.currentState = [...this.goalState];
        this.moveCount = 0;
        this.isUsingImage = false;
        this.imageData = null;
        this.solutionSteps = [];
        this.currentSolutionStep = 0;
        
        this.initializeEventListeners();
        this.renderPuzzle();
    }
    
    initializeEventListeners() {
        document.getElementById('imageUpload').addEventListener('change', this.handleImageUpload.bind(this));
        document.getElementById('useNumbersBtn').addEventListener('click', this.useNumbers.bind(this));
        document.getElementById('shuffleBtn').addEventListener('click', this.shufflePuzzle.bind(this));
        document.getElementById('solveBtn').addEventListener('click', this.solvePuzzle.bind(this));
        document.getElementById('resetBtn').addEventListener('click', this.resetPuzzle.bind(this));
        document.getElementById('prevStepBtn').addEventListener('click', this.previousSolutionStep.bind(this));
        document.getElementById('nextStepBtn').addEventListener('click', this.nextSolutionStep.bind(this));
    }
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            const canvas = document.getElementById('imageCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = () => {
                canvas.width = 300;
                canvas.height = 300;
                ctx.drawImage(img, 0, 0, 300, 300);
                
                this.imageData = [];
                const pieceSize = 100;
                
                for (let i = 0; i < 9; i++) {
                    if (i === 8) {
                        this.imageData.push(null); // Empty space
                    } else {
                        const row = Math.floor(i / 3);
                        const col = i % 3;
                        const pieceCanvas = document.createElement('canvas');
                        pieceCanvas.width = pieceSize;
                        pieceCanvas.height = pieceSize;
                        const pieceCtx = pieceCanvas.getContext('2d');
                        
                        pieceCtx.drawImage(canvas, 
                            col * pieceSize, row * pieceSize, pieceSize, pieceSize,
                            0, 0, pieceSize, pieceSize);
                        
                        this.imageData.push(pieceCanvas.toDataURL());
                    }
                }
                
                this.isUsingImage = true;
                this.renderPuzzle();
                this.updateStatus('Image loaded! You can now shuffle and play.');
            };
            
            img.src = URL.createObjectURL(file);
        }
    }
    
    useNumbers() {
        this.isUsingImage = false;
        this.imageData = null;
        this.renderPuzzle();
        this.updateStatus('Switched to number mode.');
    }
    
    renderPuzzle(state = this.currentState, containerId = 'puzzleGrid') {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        state.forEach((value, index) => {
            const tile = document.createElement('div');
            tile.className = 'puzzle-tile';
            
            if (value === 0) {
                tile.classList.add('empty');
            } else {
                if (this.isUsingImage && this.imageData) {
                    const img = document.createElement('img');
                    img.src = this.imageData[value - 1];
                    tile.appendChild(img);
                } else {
                    tile.textContent = value;
                }
                
                if (containerId === 'puzzleGrid') {
                    tile.addEventListener('click', () => this.handleTileClick(index));
                }
            }
            
            container.appendChild(tile);
        });
    }
    
    handleTileClick(index) {
        const emptyIndex = this.currentState.indexOf(0);
        
        if (this.isValidMove(index, emptyIndex)) {
            this.swapTiles(index, emptyIndex);
            this.moveCount++;
            this.updateMoveCount();
            this.renderPuzzle();
            
            if (this.isSolved()) {
                this.updateStatus('Congratulations! Puzzle solved!');
                document.getElementById('puzzleGrid').classList.add('winning-animation');
                setTimeout(() => {
                    document.getElementById('puzzleGrid').classList.remove('winning-animation');
                }, 1000);
            }
        }
    }
    
    isValidMove(clickedIndex, emptyIndex) {
        const clickedRow = Math.floor(clickedIndex / 3);
        const clickedCol = clickedIndex % 3;
        const emptyRow = Math.floor(emptyIndex / 3);
        const emptyCol = emptyIndex % 3;
        
        return (Math.abs(clickedRow - emptyRow) === 1 && clickedCol === emptyCol) ||
               (Math.abs(clickedCol - emptyCol) === 1 && clickedRow === emptyRow);
    }
    
    swapTiles(index1, index2) {
        [this.currentState[index1], this.currentState[index2]] = 
        [this.currentState[index2], this.currentState[index1]];
    }
    
    shufflePuzzle() {
        // Generate a solvable random state
        do {
            this.currentState = this.generateRandomState();
        } while (!this.isSolvable(this.currentState) || this.isSolved());
        
        this.moveCount = 0;
        this.updateMoveCount();
        this.renderPuzzle();
        this.hideSolution();
        this.updateStatus('Puzzle shuffled! Start solving.');
    }
    
    generateRandomState() {
        const state = [...this.goalState];
        for (let i = state.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [state[i], state[j]] = [state[j], state[i]];
        }
        return state;
    }
    
    isSolvable(state) {
        let inversions = 0;
        for (let i = 0; i < state.length - 1; i++) {
            for (let j = i + 1; j < state.length; j++) {
                if (state[i] !== 0 && state[j] !== 0 && state[i] > state[j]) {
                    inversions++;
                }
            }
        }
        return inversions % 2 === 0;
    }
    
    isSolved() {
        return JSON.stringify(this.currentState) === JSON.stringify(this.goalState);
    }
    
    async solvePuzzle() {
        const algorithm = document.getElementById('algorithmSelect').value;
        this.updateStatus(`Solving using ${algorithm.toUpperCase()}...`);
        
        const startTime = performance.now();
        let result;
        
        switch (algorithm) {
            case 'astar':
                result = this.aStar(this.currentState, this.manhattanDistance.bind(this));
                break;
            case 'astar-misplaced':
                result = this.aStar(this.currentState, this.misplacedTiles.bind(this));
                break;
            case 'bfs':
                result = this.bfs(this.currentState);
                break;
            case 'dfs':
                result = this.dfs(this.currentState);
                break;
            default:
                result = this.aStar(this.currentState, this.manhattanDistance.bind(this));
        }
        
        const endTime = performance.now();
        const timeElapsed = endTime - startTime;
        
        if (result) {
            this.solutionSteps = result.path;
            this.currentSolutionStep = 0;
            this.showSolution();
            this.updateAlgorithmStats(algorithm, result, timeElapsed);
            this.updateStatus(`Solution found in ${result.path.length - 1} steps!`);
        } else {
            this.updateStatus('No solution found (this should not happen for a valid puzzle).');
        }
    }
    
    aStar(initialState, heuristic) {
        const openSet = [{ state: initialState, g: 0, h: heuristic(initialState), f: heuristic(initialState), parent: null }];
        const closedSet = new Set();
        let nodesExplored = 0;
        
        while (openSet.length > 0) {
            openSet.sort((a, b) => a.f - b.f);
            const current = openSet.shift();
            const stateStr = JSON.stringify(current.state);
            
            if (closedSet.has(stateStr)) continue;
            closedSet.add(stateStr);
            nodesExplored++;
            
            if (JSON.stringify(current.state) === JSON.stringify(this.goalState)) {
                return {
                    path: this.reconstructPath(current),
                    nodesExplored,
                    pathLength: current.g
                };
            }
            
            for (const neighbor of this.getNeighbors(current.state)) {
                const neighborStr = JSON.stringify(neighbor);
                if (closedSet.has(neighborStr)) continue;
                
                const g = current.g + 1;
                const h = heuristic(neighbor);
                const f = g + h;
                
                openSet.push({
                    state: neighbor,
                    g,
                    h,
                    f,
                    parent: current
                });
            }
        }
        
        return null;
    }
    
    bfs(initialState) {
        const queue = [{ state: initialState, parent: null, depth: 0 }];
        const visited = new Set([JSON.stringify(initialState)]);
        let nodesExplored = 0;
        
        while (queue.length > 0) {
            const current = queue.shift();
            nodesExplored++;
            
            if (JSON.stringify(current.state) === JSON.stringify(this.goalState)) {
                return {
                    path: this.reconstructPath(current),
                    nodesExplored,
                    pathLength: current.depth
                };
            }
            
            for (const neighbor of this.getNeighbors(current.state)) {
                const neighborStr = JSON.stringify(neighbor);
                if (!visited.has(neighborStr)) {
                    visited.add(neighborStr);
                    queue.push({
                        state: neighbor,
                        parent: current,
                        depth: current.depth + 1
                    });
                }
            }
        }
        
        return null;
    }
    
    dfs(initialState, maxDepth = 20) {
        const stack = [{ state: initialState, parent: null, depth: 0 }];
        const visited = new Set();
        let nodesExplored = 0;
        
        while (stack.length > 0) {
            const current = stack.pop();
            const stateStr = JSON.stringify(current.state);
            
            if (visited.has(stateStr) || current.depth > maxDepth) continue;
            visited.add(stateStr);
            nodesExplored++;
            
            if (stateStr === JSON.stringify(this.goalState)) {
                return {
                    path: this.reconstructPath(current),
                    nodesExplored,
                    pathLength: current.depth
                };
            }
            
            for (const neighbor of this.getNeighbors(current.state)) {
                const neighborStr = JSON.stringify(neighbor);
                if (!visited.has(neighborStr)) {
                    stack.push({
                        state: neighbor,
                        parent: current,
                        depth: current.depth + 1
                    });
                }
            }
        }
        
        return null;
    }
    
    getNeighbors(state) {
        const neighbors = [];
        const emptyIndex = state.indexOf(0);
        const emptyRow = Math.floor(emptyIndex / 3);
        const emptyCol = emptyIndex % 3;
        
        const moves = [
            { row: -1, col: 0 }, // up
            { row: 1, col: 0 },  // down
            { row: 0, col: -1 }, // left
            { row: 0, col: 1 }   // right
        ];
        
        for (const move of moves) {
            const newRow = emptyRow + move.row;
            const newCol = emptyCol + move.col;
            
            if (newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3) {
                const newIndex = newRow * 3 + newCol;
                const newState = [...state];
                [newState[emptyIndex], newState[newIndex]] = [newState[newIndex], newState[emptyIndex]];
                neighbors.push(newState);
            }
        }
        
        return neighbors;
    }
    
    manhattanDistance(state) {
        let distance = 0;
        for (let i = 0; i < state.length; i++) {
            if (state[i] !== 0) {
                const currentRow = Math.floor(i / 3);
                const currentCol = i % 3;
                const goalIndex = state[i] - 1;
                const goalRow = Math.floor(goalIndex / 3);
                const goalCol = goalIndex % 3;
                distance += Math.abs(currentRow - goalRow) + Math.abs(currentCol - goalCol);
            }
        }
        return distance;
    }
    
    misplacedTiles(state) {
        let count = 0;
        for (let i = 0; i < state.length; i++) {
            if (state[i] !== 0 && state[i] !== this.goalState[i]) {
                count++;
            }
        }
        return count;
    }
    
    reconstructPath(node) {
        const path = [];
        while (node) {
            path.unshift(node.state);
            node = node.parent;
        }
        return path;
    }
    
    showSolution() {
        document.getElementById('solutionContainer').style.display = 'block';
        this.updateSolutionDisplay();
    }
    
    hideSolution() {
        document.getElementById('solutionContainer').style.display = 'none';
    }
    
    updateSolutionDisplay() {
        if (this.solutionSteps.length > 0) {
            this.renderPuzzle(this.solutionSteps[this.currentSolutionStep], 'solutionGrid');
            document.getElementById('stepInfo').textContent = 
                `Step ${this.currentSolutionStep} of ${this.solutionSteps.length - 1}`;
            
            document.getElementById('prevStepBtn').disabled = this.currentSolutionStep === 0;
            document.getElementById('nextStepBtn').disabled = this.currentSolutionStep === this.solutionSteps.length - 1;
        }
    }
    
    previousSolutionStep() {
        if (this.currentSolutionStep > 0) {
            this.currentSolutionStep--;
            this.updateSolutionDisplay();
        }
    }
    
    nextSolutionStep() {
        if (this.currentSolutionStep < this.solutionSteps.length - 1) {
            this.currentSolutionStep++;
            this.updateSolutionDisplay();
        }
    }
    
    updateAlgorithmStats(algorithm, result, timeElapsed) {
        const statsDiv = document.getElementById('algorithmStats');
        const heuristicName = algorithm === 'astar' ? 'Manhattan Distance' : 
                            algorithm === 'astar-misplaced' ? 'Misplaced Tiles' : 'N/A';
        
        statsDiv.innerHTML = `
            <p><span class="stat-highlight">Algorithm:</span> ${algorithm.toUpperCase()}</p>
            ${algorithm.includes('astar') ? `<p><span class="stat-highlight">Heuristic:</span> ${heuristicName}</p>` : ''}
            <p><span class="stat-highlight">Solution Length:</span> ${result.pathLength} moves</p>
            <p><span class="stat-highlight">Nodes Explored:</span> ${result.nodesExplored}</p>
            <p><span class="stat-highlight">Time Elapsed:</span> ${timeElapsed.toFixed(2)}ms</p>
            <p><span class="stat-highlight">Efficiency:</span> ${(result.pathLength / result.nodesExplored * 100).toFixed(2)}%</p>
        `;
    }
    
    resetPuzzle() {
        this.currentState = [...this.goalState];
        this.moveCount = 0;
        this.updateMoveCount();
        this.renderPuzzle();
        this.hideSolution();
        this.updateStatus('Puzzle reset to solved state.');
    }
    
    updateMoveCount() {
        document.getElementById('moveCount').textContent = this.moveCount;
    }
    
    updateStatus(message) {
        document.getElementById('gameStatus').textContent = message;
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new PuzzleGame();
});