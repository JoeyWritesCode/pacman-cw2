# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
from re import sub

from numpy import number

import math

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        self.score = state.getScore()
        self.state = state

    def getLegalActions(self):
        return self.state.getLegalPacmanActions()

class QValueTable:
    def __init__(self):
        self.data = {}
        self.frequencies = {}

    def __getitem__(self, key):
        return self.data.get(key, 0)

    def __setitem__(self, key, value):
        self.data[key] = value

    def getCount(self, key):
        return self.frequencies.get(key, 0)
    def updateCount(self, key):
        self.frequencies[key] = self.frequencies.get(key, 0) + 1

    def __str__(self) -> str:
        printable = ""
        for key in self.data:
            printable += "%s:\n" % str(key)
            printable += "Q value : %s\n" % str(self.data[key])
            printable += "Count : %s\n" % str(self.frequencies[key])
            printable += "---------------\n"
        return printable

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # A frequency table of state-action pairs
        self.stateActionVisits = {}
        # A table of Q values associated to state-action pairs
        self.qValueTable = QValueTable()
        # The last state visited, and the last action. Used for the update
        self.lastState = None
        self.lastAction = None
        self.lastReward = 0.0
        # Our optimisitic reward (is this how you do it?)
        self.optimisticReward = 100.0

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getEpsilon(self):
        return self.epsilon

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        # come back to this...
        return startState.getScore() - endState.getScore()
        util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        # If the state-action pair has not been seen before, initialise with 0
        return self.qValueTable[(state, action)]
        util.raiseNotDefined()
        

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        subsequentQs = [0] # a default value of 0
        for action in state.getLegalActions():
            subseqent_q = self.getQValue(state, action)
            subsequentQs.append(subseqent_q)
        return max(subsequentQs)


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        initialQValue = self.getQValue(state, action)
        self.qValueTable[state, action] = initialQValue + self.getAlpha() * (reward + self.getGamma() * self.maxQValue(nextState) - initialQValue)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.qValueTable.getCount((state, action))

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.qValueTable.updateCount((state, action))

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        if (counts < self.maxAttempts):
            return self.optimisticReward
        else:
            return utility

    def doTheRightThing(self, state):
        legal = state.getLegalPacmanActions()
        # in the first half of trianing, the agent is forced not to stop
        # or turn back while not being chased by the ghost
        if self.getEpisodesSoFar()*1.0/self.getNumTraining()<0.5:
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)
            if self.lastAction != None:
                last_action = self.lastAction
                distance0 = state.getPacmanPosition()[0]- state.getGhostPosition(1)[0]
                distance1 = state.getPacmanPosition()[1]- state.getGhostPosition(1)[1]
                if math.sqrt(distance0**2 + distance1**2) > 2:
                    if (Directions.REVERSE[last_action] in legal) and len(legal)>1:
                        legal.remove(Directions.REVERSE[last_action])
        tmp = util.Counter()
        for action in legal:
          tmp[action] = self.getQValue(state, action)
        return tmp.argMax()


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        """ print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore()) """

        stateFeatures = GameStateFeatures(state)
        

        # Now pick what action to take.
        # The current code shows how to do that but just makes the choice randomly.
        if self.lastState == None:
            self.lastState = state
        
        reward = state.getScore() - self.lastState.getScore()
        print("Reward : ", reward)

        if self.lastState != None:
            self.learn(self.lastState, self.lastAction, reward, state)
        else:
            reward = 0.0

        # assign s <- s'
        self.lastState = state

        if util.flipCoin(self.getEpsilon()):
            #legal.remove(actionMax)
            self.lastAction = random.choice(legal)
        else:
            """ # assign a <- argmax_a' f(Q[s', a'], N[s', a'])
            explorationValues = util.Counter()
            for action in legal:
                # assign u <- Q[s', a']
                utility = self.getQValue(state, action)
                # assign n <- N[s', a']
                numberOfVisits = self.getCount(state, action)
                # f(u, n). Fun!
                explorationValues[action] = self.explorationFn(utility, numberOfVisits)
            actionMax = explorationValues.argMax()
            self.lastAction = actionMax """
            self.lastAction = self.doTheRightThing(state)

        self.updateCount(state, self.lastAction)

        # assign r <- r'
        self.lastReward = reward

        return self.lastAction

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # update Q-values
        """ reward = state.getScore()-self.lastReward
        self.lastState = self.lastState[-1]
        last_action = self.lastAction[-1]
        self.updateQ(last_state, last_action, reward, state) """

        reward = self.computeReward(self.lastState, state)            
        self.updateCount(self.lastState, self.lastAction)
        self.learn(self.lastState, self.lastAction, reward, state)

        # reset attributes
        self.lastReward = 0.0
        self.lastState = None
        self.lastAction = None

        # decrease epsilon during the trianing
        ep = 1 - self.getEpisodesSoFar()*1.0/self.getNumTraining()
        self.setEpsilon(ep*0.1)

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            #print(self.qValueTable)
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
