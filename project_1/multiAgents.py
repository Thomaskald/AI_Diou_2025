# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getAvailableActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateNextState(agentIndex, action):
        Returns the nextState game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        _, bestAction = self.minimax(gameState, 0, 0)
        return bestAction

    def minimax(self, state, agentIndex, depth):

        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        if agentIndex == 0:
            return self.maxValue(state, depth)
        else:
            return self.minValue(state, agentIndex, depth)

    def maxValue(self, state, depth):
        bestValue = float('-inf')
        bestAction = None

        for action in state.getAvailableActions(0):
            nextState = state.generateNextState(0, action)
            value, _ = self.minimax(nextState, 1, depth)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestValue, bestAction

    def minValue(self, state, agentIndex, depth):
        bestValue = float('inf')
        bestAction = None
        nextAgent = agentIndex + 1
        numAgents = state.getNumAgents()

        for action in state.getAvailableActions(agentIndex):
            nextState = state.generateNextState(agentIndex, action)

            if nextAgent == numAgents:
                value, _ = self.minimax(nextState, 1, depth + 1)
            else:
                value, _ = self.minimax(nextState, nextAgent, depth)

            if value < bestValue:
                bestValue = value
                bestAction = action

        return bestValue, bestAction

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        score, bestAction = self.alphabeta(gameState, agentIndex = 0, depth = 0, alpha = float('-inf'), beta = float('inf'))
        return bestAction

    def alphabeta(self, state, agentIndex, depth, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        if agentIndex == 0:
            return self.maxValue(state, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(state, agentIndex, depth, alpha, beta)

    def maxValue(self, state, agentIndex, depth, alpha, beta):
        v = float('-inf')
        bestAction = None

        for action in state.getAvailableActions(agentIndex):
            successor = state.generateNextState(agentIndex, action)
            nextAgent = agentIndex + 1
            nextDepth = depth

            if nextAgent == state.getNumAgents():
                nextAgent = 0
                nextDepth += 1

            value, _ = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)

            if value > v:
                v = value
                bestAction = action

            if v > beta:
                return v, bestAction

            alpha = max(alpha, v)

        return v, bestAction


    def minValue(self, state, agentIndex, depth, alpha, beta):
        v = float('inf')
        bestAction = None

        for action in state.getAvailableActions(agentIndex):
            successor = state.generateNextState(agentIndex, action)
            nextAgent = agentIndex + 1
            nextDepth = depth

            if nextAgent == state.getNumAgents():
                nextAgent = 0
                nextDepth += 1

            value, _ = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)

            if value < v:
                v = value
                bestAction = action

            if v < alpha:
                return v, bestAction

            bestAction = min(beta, v)

        return v, bestAction

        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
