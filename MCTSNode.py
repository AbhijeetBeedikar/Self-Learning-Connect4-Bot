from kaggle_environments import make
import math
import numpy as np
import copy
import pandas as pd


MCtree = {}
c = math.sqrt(2)

# model that the MCTS algorithm uses. Is this required?
model = make("connectx", debug=True)
model.reset()

def validActions(state): #kaggle env.state
  board = state[0]['observation']["board"]
  actions = []
  for i in range(7):
    if board[i] == 0:
      actions.append(True)
    else:
      actions.append(False)
  return actions
class MCTSNode:
  def __init__(self, state, action=None, parent=None):
    global model
    self.state = state
    self.action = action # action = None only for root node
    self.parent = parent # parent = None only for root node
      # (s',a) for all a must be added to this list. If some (s',a) is missing then this is added as a leaf node and evaluated
    self.N = 0 # Number of times this node has been visited by the tree policy
    self.Q = 0 # We dont know much about the newly created node so we can keep bias 0 if we set this value as 0

    self.isRoot = True if self.parent == None else False
    tempModel = copy.deepcopy(model)
    tempModel.reset()
    tempModel.state = self.state
    if not self.isRoot:
      step = [None,None]
      player = int((self.state[0]["status"] != "ACTIVE")) # if player == red, then self.state[0]["status"] != "ACTIVE" --> False --> 0 --> index of the red player
      step[player] = self.action
      self.nextState = tempModel.step(step)
    else:
      self.nextState = self.state
    self.children = list(np.full(7,True)*validActions(self.nextState)) # add only MCTS nodes to this
    self.isTerminal = tempModel.done


  def addChild(self,child):
    self.children[child.action] = child

  @staticmethod
  def randRolloutPolicyReturn(tempModelForRollout): # this will never be called when the model inputted is at the terminal state
    '''
    returns the reward from a sample trajectory simulated based off of random action selection rollout policy
    '''

    if tempModelForRollout.done:
      return tempModelForRollout.state[0]["reward"] # we want the reward with respect to red only (because we negate this reward if a state is yellow)
    else:
      actionList = [int(i)*[0,1,2,3,4,5,6][index] for index,i in enumerate(validActions(tempModelForRollout.state)) if i]

      randAction = np.random.choice(actionList)

      p = int((tempModelForRollout.state[0]["status"] != "ACTIVE"))

      step = [None,None]
      step[p] = int(randAction)
      #print(validActions(tempModelForRollout.state))

      #FIX: Check whether any invalid moves were made or not

      tempModelForRollout.step(step)
      #tempModelForRollout.render(mode="ipython")

      return MCTSNode.randRolloutPolicyReturn(tempModelForRollout)

  def treeTraverse(self):
    '''Traverses along the Monte Carlo Tree. If called on the root, it performs an update along a trajectory based on
    the tree policy. Uses a rollout algorithm to update the value of a newly added leaf node and backs up this value along
    the tree trajectory. The reward backed up is different (opposite of each other) for the two colours.

    Must be called after each complete run (rollout + backup) of the MCTS algorithm'''
    global model
    global c
    action = None

    # Recursive Cases: Terminal, Leaf, root, intermediate fully expanded, intermediate semi-expanded

    # root case is handles by all the code blocks

    # Terminal Case
    if self.isTerminal: # True is not in self.children in the draw case (also covers win case)
      self.N +=1
      return self.Q

    # intermediate semi-expanded Case --> create a leaf node (covers leaf case)
    elif True in self.children: # When node is not fully expanded so a leaf node is created
      action = list(self.children).index(True)

      # set the model's state to the child's state for the constructor to work correctly when defining the child
      tempModel = copy.deepcopy(model)
      tempModel.reset()
      tempModel.state = self.nextState

      #add child (leaf node) to the list of children
      child = MCTSNode(self.nextState,action,self)

      self.addChild(child)

      # calculate the expected return using rollout algorithm
      rolloutReturn = 0
      for i in range(20):
        rolloutReturn += MCTSNode.randRolloutPolicyReturn(copy.deepcopy(tempModel)) #

      rolloutReturn /= 20

      # correctly assign the value of the expected return based on colour of the state of the new child
      # color of child.state represents whose turn it is when taking child.action
              # if red wins in the trajectory and this (state,action) node represents red taking an action,
              # then the cumulative reward for this node should be incremented by 1
      if child.state[0]["status"] == "ACTIVE":
        child.Q = rolloutReturn
      else:
        child.Q = -1*rolloutReturn


      self.N +=1

      return child.Q # return the value of the child calculated by the rollout policy (used in backup process)

    # intermediate expanded Case --> move to the next node using tree policy
    else: #UCT tree policy
      try:
        maxAction = []
        for i in self.children:
          if i == False:
            maxAct = -1*np.inf #action must never be taken
          elif i.N == 0:
            maxAct = np.inf # expansion
          else:
            maxAct = i.Q + c*math.sqrt(math.log(self.N)/i.N)
          maxAction.append(maxAct)
        action = np.argmax(maxAction)
        if len(maxAction) != 7:
          raise Exception(f"The number of actions to be evaluated by argmax is not correct: Expected 7 Got {len(maxAction)}")
      except:
        print(self.children)
        raise

      child = self.children[action]
      reward = child.treeTraverse()

      if self.state[0]["status"] == "ACTIVE": # if the state is red.
        self.Q += reward
      else: # if the state is not red i.e yellow
        self.Q -= reward
      self.N +=1
      return reward