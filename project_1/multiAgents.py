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

        bestScore, bestAction = self.minimax(gameState, 0, 0)
        return bestAction

    def minimax(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, currentDepth, agentIndex):
        v = float("-inf")
        bestAction = None
        for action in gameState.getAvailableActions(agentIndex) :
            successor = gameState.generateNextState(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.minimax(successor, successorIndex, successorDepth)[0]

            if successorValue > v:
                v = successorValue
                bestAction = action

        return v, bestAction

    def minValue(self, gameState, currentDepth, agentIndex):
        v = float("inf")
        bestAction = None
        for action in gameState.getAvailableActions(agentIndex):
            successor = gameState.generateNextState(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.minimax(successor, successorIndex, successorDepth)[0]

            if successorValue < v:
                v = successorValue
                bestAction = action

        return v, bestAction

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
        bestScore, bestAction = self.alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))
        return bestAction

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.minValue(gameState, depth, agentIndex, alpha, beta)

    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
        v = float("-inf")
        bestAction = None
        for action in gameState.getAvailableActions(agentIndex):
            successor = gameState.generateNextState(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue > v:
                v = successorValue
                bestAction = action

            if v > beta:
                return v, bestAction

            alpha = max(alpha, v)

        return v, bestAction

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        v = float("inf")
        bestAction = None
        for action in gameState.getAvailableActions(agentIndex):
            successor = gameState.generateNextState(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue < v:
                v = successorValue
                bestAction = action

            if v < alpha:
                return v, bestAction

            beta = min(beta, v)

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

        bestScore, bestAction = self.expectimax(gameState, 0, 0)
        return bestAction

    def expectimax(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.expValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth, agentIndex):
        v = float("-inf")
        bestAction = None
        for action in gameState.getAvailableActions(agentIndex):
            successor = gameState.generateNextState(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.expectimax(successor, successorIndex, successorDepth)[0]

            if successorValue > v:
                v = successorValue
                bestAction = action

        return v, bestAction

    def expValue(self, gameState, depth, agentIndex):
        v = 0
        bestAction = None

        if len(gameState.getAvailableActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState), None

        for action in gameState.getAvailableActions(agentIndex):
            successor = gameState.generateNextState(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.expectimax(successor, successorIndex, successorDepth)[0]

            v += successorValue

        return v, bestAction

        #util.raiseNotDefined()

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
