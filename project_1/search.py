# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getInitialState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getNextStates(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (nextState,
        action, stepCost), where 'nextState' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getInitialState())
    print("Is the start a goal?", problem.isGoalState(problem.getInitialState()))
    print("Start's nextStates:", problem.getNextStates(problem.getInitialState()))
    """
    "*** YOUR CODE HERE ***"



    from util import Stack

    stack = Stack()
    start = problem.getInitialState()
    stack.push((start, []))
    visited = set()

    while not stack.isEmpty():
        current, path = stack.pop()

        if problem.isGoalState(current):
            return path

        if current not in visited:
            visited.add(current)
            # Use "_" because we don't care about that result
            for nextState, action, _ in problem.getNextStates(current):
                stack.push((nextState, path + [action]))

    return []

    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"



    from util import Queue

    queue = Queue()
    start = problem.getInitialState()
    queue.push((start, []))
    visited = set()

    while not queue.isEmpty():
        current, path = queue.pop()

        if problem.isGoalState(current):
            return path

        if current not in visited:
            visited.add(current)
            # Use "_" because we don't care about that result
            for nextState, action, _ in problem.getNextStates(current):
                queue.push((nextState, path + [action]))

    return []


    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"


    from util import PriorityQueue

    priorityQueue = PriorityQueue()
    start = problem.getInitialState()
    priorityQueue.push((start, [], 0), 0)
    visited = {}

    while not priorityQueue.isEmpty():
        current, path, cost = priorityQueue.pop()

        if (current in visited) and (visited[current] <= cost): continue

        visited[current] = cost

        if problem.isGoalState(current): return path

        for nextState, action, stepCost in problem.getNextStates(current):
            totalCost = cost + stepCost
            if (current not in visited) or (totalCost < visited.get(nextState, float('inf'))):
                priorityQueue.push((nextState, path + [action], totalCost), totalCost)

    return []


    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    start = problem.getInitialState()
    priorityQueue = PriorityQueue()
    priorityQueue.push((start, [], 0), heuristic(start, problem))
    visited = {}

    while not priorityQueue.isEmpty():
        current, path, cost = priorityQueue.pop()

        if current in visited and visited[current] <= cost:
            continue

        visited[current] = cost

        if problem.isGoalState(current):
            return path

        for nextState, action, stepCost in problem.getNextStates(current):
            newCost = cost + stepCost
            priority = newCost + heuristic(nextState, problem)
            if nextState not in visited or newCost < visited.get(nextState, float('inf')):
                priorityQueue.push((nextState, path + [action], newCost), priority)

    return []

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
