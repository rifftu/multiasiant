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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        currentFood = currentGameState.getFood()
        
        minDist = 9999999
        for food in newFood.asList():
            minDist = min(minDist, util.manhattanDistance(newPos, food))
        score += 1/minDist

        if currentFood[newPos[0]][newPos[1]]:
            score += 1

        for ghost in newGhostStates:
            ghostPos = ghost.configuration.getPosition()
            if manhattanDistance(ghostPos, newPos) < 2:
                score -= 1000

        return score

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

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self._action_recurse(gameState, self.depth, 0)[0]

    def _action_recurse(self, gameState, level, agent):

        if level == 0:
            return (None, gameState, self.evaluationFunction(gameState))

        action_list = gameState.getLegalActions(agent)
        if not action_list:
            return (None, gameState, self.evaluationFunction(gameState))

        successor_list = [gameState.generateSuccessor(agent, action) for action in action_list]

        if agent >= gameState.getNumAgents() - 1:
            scores_list = [self._action_recurse(succ_state, level - 1, 0) for succ_state in successor_list]

        else:
            scores_list = [self._action_recurse(succ_state, level, agent + 1) for succ_state in successor_list]

        if agent == 0:
            best_index = self._max_index(scores_list)

        else:
            best_index = self._min_index(scores_list)

        if agent > 0:
            return (None, None, scores_list[best_index][2])

        return (action_list[best_index], successor_list[best_index], scores_list[best_index][2])

    @staticmethod
    def _max_index(scores_list):
        curr_index = 0
        curr_max = float('-inf')
        for i in range(len(scores_list)):
            score = scores_list[i][2]
            if score > curr_max:
                curr_max = score
                curr_index = i
        return curr_index

    @staticmethod
    def _min_index(scores_list):
        curr_index = 0
        curr_min = float('inf')
        for i in range(len(scores_list)):
            score = scores_list[i][2]
            if score < curr_min:
                curr_min = score
                curr_index = i
        return curr_index

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self._action_recurse(gameState, self.depth, 0, float('-inf'), float('inf'))[0]


    def _action_recurse(self, gameState, level, agent, alpha, beta):

        if level == 0:
            return (None, gameState, self.evaluationFunction(gameState))

        action_list = gameState.getLegalActions(agent)
        if not action_list:
            return (None, gameState, self.evaluationFunction(gameState))

        #successor_list = [gameState.generateSuccessor(agent, action) for action in action_list]

        successor_list = []

        scores_list = []

        if agent >= gameState.getNumAgents() - 1:
            next_level = level - 1
            next_agent = 0
        else:
            next_level = level
            next_agent = agent + 1


        if agent == 0:


            v = float('-inf')

            for i in range(len(action_list)):

                action = action_list[i]

                succ_state = gameState.generateSuccessor(agent, action)

                successor_list.append(succ_state)

                succ = self._action_recurse(succ_state, next_level, next_agent, alpha, beta)

                scores_list.append(succ)

                if succ[2] > v:
                    v = succ[2]
                    best_index = i
                if v > beta:
                    return (action_list[i], successor_list[i], v)
                alpha = max(alpha, v)

        else:
            v = float('inf')

            for i in range(len(action_list)):

                action = action_list[i]

                succ_state = gameState.generateSuccessor(agent, action)

                successor_list.append(succ_state)

                succ = self._action_recurse(succ_state, next_level, next_agent, alpha, beta)

                scores_list.append(succ)

                if succ[2] < v:
                    v = succ[2]
                    best_index = i
                if v < alpha:
                    return (action_list[i], successor_list[i], v)
                beta = min(beta, v)


        return (action_list[best_index], successor_list[best_index], scores_list[best_index][2])


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
        return self._action_recurse(gameState, self.depth, 0)[0]
    def _action_recurse(self, gameState, level, agent):

        if level == 0:
            return (None, gameState, self.evaluationFunction(gameState))

        action_list = gameState.getLegalActions(agent)
        if not action_list:
            return (None, gameState, self.evaluationFunction(gameState))

        successor_list = [gameState.generateSuccessor(agent, action) for action in action_list]

        if agent >= gameState.getNumAgents() - 1:
            scores_list = [self._action_recurse(succ_state, level - 1, 0) for succ_state in successor_list]

        else:
            scores_list = [self._action_recurse(succ_state, level, agent + 1) for succ_state in successor_list]

        if agent == 0:
            best_index = self._max_index(scores_list)
            return (action_list[best_index], successor_list[best_index], scores_list[best_index][2])

        else:


            return (None, None, self._avg_score(scores_list))


    @staticmethod
    def _max_index(scores_list):
        curr_index = 0
        curr_max = float('-inf')
        for i in range(len(scores_list)):
            score = scores_list[i][2]
            if score > curr_max:
                curr_max = score
                curr_index = i
        return curr_index

    @staticmethod
    def _avg_score(scores_list):
        total = 0
        length = len(scores_list)
        for i in range(length):
            total += scores_list[i][2]
        return total/length

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
