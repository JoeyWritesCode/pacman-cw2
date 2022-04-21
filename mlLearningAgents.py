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

from sqlalchemy import true

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

    def getGoodActions(self):
        legal = self.state.getLegalActions()
        
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        return legal

        



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
                 maxAttempts: int = 5,
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
        self.beta = 5.0
        self.wins = 0
        self.loses = 0

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
        #print("Reward : ", endState.getScore() - startState.getScore())
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
        if (nextState.isTerminal()):
            qMax = 0
        else:
            qMax = self.maxQValue(nextState)
        self.qValueTable[(state,action)] = initalQValue + self.getAlpha() * \
        (reward + self.getGamma() * qMax - initalQValue)
        #print(self.qValueTable[(state, action)])
     

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

    def argMax(self, dict):
        """
        Returns the key with the highest value.
        """
        if len(list(dict.keys())) == 0: return None
        all_items = list(dict.items())
        values = [x[1] for x in all_items]
        maxIndex = values.index(max(values))
        return all_items[maxIndex][0]

    def epislonGreedy(self,
                      state: GameState) -> Directions:
        """
        Use an epsilon-greedy policy to select an action

        Args:
            state: the current state

        Returns:
            The action to take
        """   

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(state.getLegalPacmanActions())
        else:
            tmp = {}
            for action in state.getLegalPacmanActions():
                tmp[action] = self.getQValue(state, action)   
            
            distance0 = state.getPacmanPosition()[0]- state.getGhostPosition(1)[0]
            distance1 = state.getPacmanPosition()[1]- state.getGhostPosition(1)[1]
            if math.sqrt(distance0**2 + distance1**2) > 2 and self.action != None:
                if len(tmp)>1:
                    tmp.pop(Directions.REVERSE[self.action], None)         
                    
            return self.argMax(tmp)
            

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

        if self.state != None and self.action != None:
            # Update reward
            self.score = self.computeReward(self.state, state)
            # Ammend with exploration function
            #self.score += self.explorationFn(self.score, self.getCount(self.state, self.action))
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
        
        if (state.isLose()):
            self.loses += 1
        else:
            self.wins += 1
        """ stateFeatures = GameStateFeatures(state)
        stateFeatures.setTerminal() """

        # update Q-values
        self.score = self.computeReward(self.state, state)
        self.learn(self.state, self.action, self.score, state)

        self.score = 0
        self.state = None
        self.action = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))
            success_rate = self.wins / (self.wins + self.loses)
            print("Success rate : %f" % success_rate)
            self.setAlpha(0)
            self.setEpsilon(0)
        else:
            new_epsilon = self.getEpsilon() * (1 - self.getEpisodesSoFar() / self.getNumTraining())
            self.setEpsilon(new_epsilon)

class OtherAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Q-values
        self.q_value = util.Counter()
        # current score
        self.score = 0
        # last state
        self.lastState = []
        # last action
        self.lastAction = []



    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # functions for calculation
    # get Q(s,a)
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
        return self.q_value[(state, action)]
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
        if (nextState.isTerminal):
            qmax = 0
        else:
            qmax = self.maxQValue(nextState.state)
        q = self.getQValue(state.state,action)
        self.q_value[(state.state,action)] = q + self.alpha*(reward + self.gamma*qmax - q)
        #print(self.qValueTable[(state, action)])

    def updateQ(self, state, action, reward, qmax):
        q = self.getQValue(state,action)
        self.q_value[(state,action)] = q + self.alpha*(reward + self.gamma*qmax - q)

    # return the action maximises Q of state
    def doTheRightThing(self, state):
        legal = state.getLegalPacmanActions()
        # in the first half of trianing, the agent is forced not to stop
        # or turn back while not being chased by the ghost
        if self.getEpisodesSoFar()*1.0/self.getNumTraining()<0.5:
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)
            if len(self.lastAction) > 0:
                last_action = self.lastAction[-1]
                distance0 = state.getPacmanPosition()[0]- state.getGhostPosition(1)[0]
                distance1 = state.getPacmanPosition()[1]- state.getGhostPosition(1)[1]
                if math.sqrt(distance0**2 + distance1**2) > 2:
                    if (Directions.REVERSE[last_action] in legal) and len(legal)>1:
                        legal.remove(Directions.REVERSE[last_action])
        tmp = util.Counter()
        for action in legal:
          tmp[action] = self.getQValue(state, action)
        return tmp.argMax()

    def argMax(self, dict):
        """
        Returns the key with the highest value.
        """
        if len(list(dict.keys())) == 0: return None
        all_items = list(dict.items())
        values = [x[1] for x in all_items]
        maxIndex = values.index(max(values))
        return all_items[maxIndex][0]

    def epislonGreedy(self,
                      state: GameState) -> Directions:
        """
        Use an epsilon-greedy policy to select an action

        Args:
            state: the current state

        Returns:
            The action to take
        """   

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(state.getLegalPacmanActions())
        else:
            tmp = {}
            for action in state.getLegalPacmanActions():
                tmp[action] = self.getQValue(state, action)   
            
            distance0 = state.getPacmanPosition()[0]- state.getGhostPosition(1)[0]
            distance1 = state.getPacmanPosition()[1]- state.getGhostPosition(1)[1]
            if math.sqrt(distance0**2 + distance1**2) > 2 and len(self.lastAction) > 0:
                if len(tmp)>1:
                    tmp.pop(Directions.REVERSE[self.lastAction[-1]], None)         
                    
            return self.argMax(tmp)

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        # the legal action of this state
        features = GameStateFeatures(state)
        
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # update Q-value
        reward = state.getScore()-self.score
        if len(self.lastState) > 0:
            last_features = self.lastState[-1]
            last_action = self.lastAction[-1]
            self.learn(last_features, last_action, reward, features)
            """ max_q = self.maxQValue(state)
            self.updateQ(last_state, last_action, reward, max_q) """

        # e-greedy
        if util.flipCoin(self.epsilon):
            action =  random.choice(legal)
        else:
            action =  self.epislonGreedy(state)

        # update attributes
        self.score = state.getScore()
        self.lastState.append(features)
        self.lastAction.append(action)

        return action

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        features = GameStateFeatures(state)
        features.isTerminal = True
        # update Q-values
        reward = state.getScore()-self.score
        last_state = self.lastState[-1]
        last_action = self.lastAction[-1]

        self.learn(last_state, last_action, reward, features)
        #self.updateQ(last_state, last_action, reward, 0)

        # reset attributes
        self.score = 0
        self.lastState = []
        self.lastAction = []
        
        #stateFeatures.isTerminal = False

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

class UntouchedAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Q-values
        self.q_value = util.Counter()
        # current score
        self.score = 0
        # last state
        self.lastState = []
        # last action
        self.lastAction = []



    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # functions for calculation
    # get Q(s,a)
    def getQValue(self, state, action):
        return self.q_value[(state,action)]

    # return the maximum Q of state
    def getMaxQ(self, state):
        q_list = []
        for a in state.getLegalPacmanActions():
            q = self.getQValue(state,a)
            q_list.append(q)
        if len(q_list) ==0:
            return 0
        return max(q_list)

    # update Q value
    def updateQ(self, state, action, reward, qmax):
        q = self.getQValue(state,action)
        self.q_value[(state,action)] = q + self.alpha*(reward + self.gamma*qmax - q)

    # return the action maximises Q of state
    def doTheRightThing(self, state):
        legal = state.getLegalPacmanActions()
        # in the first half of trianing, the agent is forced not to stop
        # or turn back while not being chased by the ghost
        if self.getEpisodesSoFar()*1.0/self.getNumTraining()<0.5:
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)
            if len(self.lastAction) > 0:
                last_action = self.lastAction[-1]
                distance0 = state.getPacmanPosition()[0]- state.getGhostPosition(1)[0]
                distance1 = state.getPacmanPosition()[1]- state.getGhostPosition(1)[1]
                if math.sqrt(distance0**2 + distance1**2) > 2:
                    if (Directions.REVERSE[last_action] in legal) and len(legal)>1:
                        legal.remove(Directions.REVERSE[last_action])
        tmp = util.Counter()
        for action in legal:
          tmp[action] = self.getQValue(state, action)
        return tmp.argMax()

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        # the legal action of this state
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # update Q-value
        reward = state.getScore()-self.score
        if len(self.lastState) > 0:
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            max_q = self.getMaxQ(state)
            self.updateQ(last_state, last_action, reward, max_q)

        # e-greedy
        if util.flipCoin(self.epsilon):
            action =  random.choice(legal)
        else:
            action =  self.doTheRightThing(state)

        # update attributes
        self.score = state.getScore()
        self.lastState.append(state)
        self.lastAction.append(action)

        return action

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # update Q-values
        reward = state.getScore()-self.score
        last_state = self.lastState[-1]
        last_action = self.lastAction[-1]
        self.updateQ(last_state, last_action, reward, 0)

        # reset attributes
        self.score = 0
        self.lastState = []
        self.lastAction = []

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