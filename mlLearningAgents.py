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
        self.isTerminal = False

    def getLegalActions(self):
        return self.state.getLegalPacmanActions()

    def setTerminal(self):
        self.isTerminal = True

    def updateScore(self, new_score):
        self.score = new_score



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

    def getBestAction(self, state, legal_actions):
        """
        Assesses the QValueTable for a given state.
        Determines which trajectory holds the largest expected utility.
        
        Args:
            state: the current state
            legal_actions: availble actions to the state

        Returns:
            the action that would yield the highest q value
        """
        values_for_actions = {}
        for action in legal_actions:
            values_for_actions[self[(state, action)]] = action
        q_max = max(values_for_actions.keys())
        return values_for_actions[q_max]



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
        # A table of Q values associated to state-action pairs
        self.qValueTable = QValueTable()
        # The last state visited, and the last action. Used for the update
        self.state = None
        self.action = None
        self.score = 0
        # Our optimisitic reward (is this how you do it?)
        self.explorationBonus = 2.0
        self.beta = 10.0

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
        return endState.getScore() - startState.getScore()

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
            
        subsequentQs = [] # a default value of 0
        for action in state.getLegalActions():
            subseqent_q = self.getQValue(state, action)
            subsequentQs.append(subseqent_q)
        if len(subsequentQs) == 0:
            return 0
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
        initalQValue = self.getQValue(state, action)
        qMax = self.maxQValue(nextState)
        self.qValueTable[(state,action)] = initalQValue + self.getAlpha() * (reward + self.getGamma() * qMax - initalQValue)
     

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
            return utility + (self.beta / math.sqrt(counts))
        else:
            return 0

    def epislonGreedy(self,
                      state: GameState) -> Directions:
        """
        Use an epsilon-greedy policy to select an action

        Args:
            state: the current state

        Returns:
            The action to take
        """
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)        

        if random.uniform(0, 1) < self.getEpsilon():
            return random.choice(legal)
        else:
            return self.qValueTable.getBestAction(state, legal)
            

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

        # logging to help you understand the inputs, feel free to remove
        """ print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore()) """

        #stateFeatures = GameStateFeatures(state)

        # Now pick what action to take.
        # The current code shows how to do that but just makes the choice randomly.

        # Epsilon greedy policy
        action = self.epislonGreedy(state)
        print(action)

        if self.state != None and self.action != None:
            # Update reward
            self.score = self.computeReward(self.state, state)
            # Ammend with exploration function
            self.score += self.explorationFn(self.score, self.getCount(self.state, self.action))
            # update Q-value
            self.learn(self.state, self.action, self.score, state)

        # update counts
        self.qValueTable.updateCount((state, action))

        self.state = state
        self.action = action
        return action

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
        self.state = self.state[-1]
        last_action = self.lastAction[-1]
        self.updateQ(state, last_action, reward, state) """
        
        stateFeatures = GameStateFeatures(state)
        stateFeatures.setTerminal()

        # update Q-values
        self.score = self.computeReward(self.state, state)
        self.learn(self.state, self.action, self.score, state)

        # reset attributes
        self.score = 0
        self.state = None
        self.action = None

        # decrease epsilon during the trianing
        ep = 1 - self.getEpisodesSoFar()*1.0/self.getNumTraining()
        self.setEpsilon(ep*0.1)


        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() % 100 == 0:
            print("Completed %s runs of training" % self.getEpisodesSoFar())

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
