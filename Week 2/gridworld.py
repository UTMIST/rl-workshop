import numpy as np


ACTIONS = {
  0: [-1, 0], # north
  1: [0, 1], # east
  2: [1, 0], # south
  3: [0, -1], # west
}


class GridWorld:
  """
  Simple n x n gridworld environment from the slides.

  Actions: north (0), east (1), south (2), west (3)
  Taking an action that would cause the agent to fall off will keep the agent in the same state.

  Reward is -1 for each step until the terminal states are reached.
  """
  def __init__(self, n=4):
    self.n = n
    self.action_space = np.arange(4)
    self.terminal_states = [0, (n ** 2) - 1]
    
    grid = -1 * np.ones((4, 4))
    grid[0, 0] = 0
    grid[0, 1:n] = -10
    grid[n-2, 0:n-1] = -10
    self.reward_grid = grid
    
  def state(self, loc):
    """
    Convert the coordinate into a state number.
    A 3 x 3 grid would have the following states in the following coordinates:
      | 0 | 1 | 2 |
      | 3 | 4 | 5 |
      | 6 | 7 | 8 |
    """
    return int(loc[0] * self.n + loc[1])
    
  def loc(self, state):
    """
    Convert the state number into a coordinate.
    Inverse of the state function.
    """
    return (state // self.n, state % self.n)
  
  def _action_prob(self, a):
    """
    Returns a dictionary of the action probabilities. 
    There is a 0.7 chance the agent will take the action and a 0.3 chance of taking a random action.
    """
    action_prob = {i: 0.1 for i in self.action_space}
    action_prob[a] = 0.7
    return action_prob

  def _is_inbounds(self, row, col):
    return row >= 0 and row < self.n and col >= 0 and col < self.n

  def transitions(self, s, a):
    """
    new state, reward, probability
    """
    assert a in self.action_space, "Invalid action"
    
    row, col = self.loc(s)
    
    if s in self.terminal_states:
      return np.array([[s, 0, 1]])
    
    action_prob = self._action_prob(a)

    transitions = {}
    for next_action in self.action_space:
      next_row, next_col = row + ACTIONS[next_action][0], col + ACTIONS[next_action][1]
      if not self._is_inbounds(next_row, next_col):
        next_row, next_col = row, col
      next_state = self.state((next_row, next_col))
      reward = self.reward_grid[next_row, next_col]
      probability = action_prob[next_action]

      if next_state in self.terminal_states:
        reward = 0

      if (next_state, reward) in transitions:
        transitions[(next_state, reward)] += probability
      else:
        transitions[(next_state, reward)] = probability
    return np.array([[*s, np.round(transitions[s], 1)] for s in transitions])

  def render(self):
    print(self.reward_grid)


if __name__=="__main__":
    env = GridWorld()
    print(env.transitions(1, 0))
    env.render()
