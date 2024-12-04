import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from typing import Tuple

import a_star

ALPHA=0.8
GAMMA=0.5
EPSILON=0.1
REWARD_VALUE=30
STEP_PENALTY=-0.2
class newGridWorld:
    def __init__(self, agents, size):
        self.agents = agents
        self.targets=agents[0].targets
        for target in self.targets.targets_list:
            target.visitors = [0 for _ in agents]
        self.size = size
        self.done=False
        for agent in self.agents:
            agent.world=self
            agent.size=self.size
        self.global_reward=0
        self.reward_string=' '


    def step(self, multiple_policies=True):
        # reset number of agents at target
        target_list = self.targets.targets_list
        for target in target_list:
            target.num_agents_present=0
        #list to keep old rewards
        old_locations=[]*len(self.agents)
        directions=[]*len(self.agents)
        #all agents move at once
        for agent in self.agents:
            agent.path.append(agent.location)
            old_locations.append(agent.location)
            #moves agent and gets direction moved
            direction=agent.move_from_policy()
            directions.append(direction)
        for i, target in enumerate(target_list):
            target_location = target.location
            for j, agent in enumerate(self.agents):
                reward = agent.get_reward()
                agent.update_policy(old_locations[j][0], old_locations[j][1], directions[j], reward)
                if agent.location == target_location:
                    self.global_reward += reward
                    if reward > 0:
                        self.reward_string += str(reward) + ', '
                    # need to reset after each step
                    target.num_agents_present += 1
                    if i not in agent.trees_visited:
                        agent.policy_location[(agent.location)] =len(agent.trees_visited)
                        agent.trees_visited.append(i)
                        target.reward_value *= 0.2
                    # trying to change value after collection

                    # target.fully_measured=True
                    target.visits += 1
                    target.visitors[j] = 1 # unique visits
                    if len(agent.trees_visited)<len(agent.policies) and multiple_policies:
                        agent.policy=agent.policies[len(agent.trees_visited)]
                    else:
                        agent.start_policy()
                        agent.policies.append(agent.policy)
        self.global_reward+=STEP_PENALTY
        if all([all(t.visitors) for t in target_list]):
            self.done = True

    def viz(self):
        #might be switched
        plt.figure()
        grid = np.zeros((self.size[1], self.size[0]))
        #move_strings=np.empty((self.size[1], self.size[0]), dtype='U50')
        #only works for this many agents
        cmaps=     ['Blues', 'Purples', 'Greys', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        for i,agent in enumerate(self.agents):
            #yellow patch for start location
            x=agent.start_location[0]
            y=agent.start_location[1]
            plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='yellow'))
            plt.text(x-0.3,y, 'a'+str(i))
            path=agent.path
            val=0
            for entry in path:
                val+=1
                grid[entry[1],entry[0]]=val

        for target in self.targets.targets_list:
            x=target.location[0]
            y=target.location[1]
            if target.fully_measured:
                color='green'
            else:
                color='red'
            plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color))
            plt.text(x-0.3, y, 'o'+str(sum(target.visitors)))
        plt.imshow(grid, cmap=cmaps[i], interpolation='nearest')
        plt.show(block = False)

    def path_viz(self):

        plt.figure()
        colors = ['b', 'm', 'r', 'y', 'g', 'l']
        div = 0.1 # how far apart each line is

        for i, agent in enumerate(self.agents):
            color = colors[i%len(colors)]
            path = np.array(agent.path)
            ofst =i * div
            plt.plot(path[0,0], path[0,1], color+'o', markersize=10)
            plt.plot(path[:,0]+ofst, path[:,1]+ofst, colors[i%len(colors)]+"--")
        
        for target in self.targets.targets_list:
            x=target.location[0]
            y=target.location[1]
            if target.fully_measured:
                color='green'
            else:
                color='red'
            plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color))
            plt.text(x-0.3, y, 'o'+str(sum(target.visitors)))

        plt.ylim(0,self.size[1])
        plt.xlim(0,self.size[0])
        plt.xticks(np.arange(-1, self.size[0], step=1))
        plt.yticks(np.arange(-1, self.size[1], step=1))
        plt.grid()
        plt.show()

class targetsObj:
    def __init__(self, targets):
        targets_list=[]
        for target in targets:
            t=tree(target)
            targets_list.append(t)
        self.targets_list=targets_list

class tree:
    def __init__(self, location):
        self.location=(location[0], location[1])
        self.current_measurement=0
        self.true_measurement=random.random()
        self.fully_measured=False
        self.num_agents_present=0
        self.reward_value=REWARD_VALUE
        self.visits=0
        self.visitors = []


class newAgent:
    def __init__(self, start_location, targets):
        self.policy = []
        self.location = start_location
        self.world = []
        self.size = []
        self.targets=targets
        self.start_location = start_location
        self.path=[]
        self.trees_visited=[]
        self.policies=[]
        self.policy_location={}



    def move(self, instruction):
        x = self.location[0]
        y = self.location[1]
        #-y is up in visualizer
        if instruction == 0:
            y -= 1
        if instruction == 1:
            y += 1
        if instruction == 2:
            x -= 1
        if instruction == 3:
            x += 1
        #check if move is allowed
        allowed=False
        if x>=0 and x<self.size[0] and y>=0 and y<self.size[1]:
            allowed=True
        if (x,y) in self.path:
            visited=True
        else:
            visited=False
        return allowed, x, y, visited

    def start_policy(self):
        policy = np.empty((self.size[1], self.size[0]), dtype=object)
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                #up down left right
                policy[y][x] = np.ones(4)
                if x == 0:
                    policy[y][x][2] = -10
                elif x == self.size[0] - 1:
                    policy[y][x][3] = -10
                if y == 0:
                    policy[y][x][0] = -10
                elif y == self.size[1] -1:
                    policy[y][x][1] = -10
        self.policy = policy


    def move_from_policy(self, avoid_old_locations=False):
        x = self.location[0]
        y = self.location[1]
        val = random.random()
        if val>EPSILON:
            direction = random.choice(np.where(self.policy[y,x]==np.max(self.policy[y, x]))[0])
            allowed, x, y, visited=self.move(direction)
        else:
            direction = random.choice(range(4))
            allowed, x, y, visited = self.move(direction)
        #track allowed directions to choose from if all nearby tiles have been visited
        if avoid_old_locations:
            allowed_moves=[]
            bad_move_count=0
            #try other directions if not allowed
            if not allowed or visited:
                direction=0
                bad_move_count+=1
                allowed, x, y, visited = self.move(direction)
                if allowed:
                    allowed_moves.append(direction)
            #try all possible directions
            while (not allowed or visited) and direction<3:
                bad_move_count+=1
                direction+=1
                allowed, x, y, visited = self.move(direction)
                if allowed:
                    allowed_moves.append(direction)
            #if all allowed directions have been visited
            if bad_move_count==4:
                direction=random.choice(allowed_moves)
                allowed, x, y, visited = self.move(direction)
        if allowed:
            self.location = (x, y)
            # a little funky, only want to return allow direction
            return direction

    def get_reward(self):
        #UPDATE FUNCTION BELOW TOO
        x = self.location[0]
        y = self.location[1]
        reward=0
        for i, target in enumerate(self.targets.targets_list):
            # if target.location == (x,y):
            #     print('coupling ', target.num_agents_present>=COUPLING)
            #     print('visited: ', i not in self.trees_visited)
            if target.location == (x,y) and i not in self.trees_visited:
                reward+=target.reward_value
            else:
                reward+=STEP_PENALTY

        return reward

    #check reward at next location without actually changing x,y
    #used for propogation
    def get_reward_location(self,x,y):
        reward=0
        for target in self.targets.targets_list :
            if target.location == (x,y):
                reward+=target.reward_value
            else:
                reward+=0
        return reward

    def find_next_position(self, x, y, direction):
        # directions are: up, down, left, right, but -y is up
        if direction == 0:
            y -= 1
        elif direction == 1:
            y += 1
        elif direction == 2:
            x -= 1
        elif direction == 3:
            x += 1
        return x,y


    def update_policy(self,x,y,direction, reward):
        x_next, y_next = self.find_next_position(x,y,direction)
        if reward > 0:
            self.policy[y,x][direction] += reward
            self.propogate_reward_along_path(reward)
        else:
            # reward is empty
            self.policy[y][x][direction] += ALPHA * (GAMMA*max(self.policy[y_next,x_next]) - self.policy[y,x][direction])

    def propogate_reward_along_path(self, true_reward):
        # uses A* to move along best path, starting at current location
        path = a_star.a_star_search(np.zeros(self.size), self.location, self.start_location)
        if path is None:
            return
        x_next, y_next = path[0]
        for x,y in path[1:]:
            if x_next == x:
                direction = 0 if y_next < y else 1
            elif y_next == y:
                direction = 2 if x_next < x else 3
            else:
                print(f"Path moved strangely to {x_next,y_next} from {x,y}!")
                exit(1)
            if true_reward > 0:
                self.policy[y,x][direction] = true_reward
                true_reward = 0
            else:
                reward = self.policy[y,x][direction] + ALPHA * (GAMMA*max(self.policy[y_next,x_next]) - self.policy[y,x][direction])
                self.policy[y][x][direction]=reward
                assert self.policy[y][x][direction] == reward
            x_next, y_next = x,y
            

    def propagate_reward(self):
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for direction in range(4):
                    new_x, new_y = self.find_next_position(x,y, direction)
                    #check if possible move
                    if new_x >= 0 and new_x < self.size[0] and new_y >= 0 and new_y < self.size[1]:
                        reward=self.policy[y,x][direction]+ALPHA*(GAMMA*max(self.policy[new_y,new_x])-self.policy[y,x][direction])
                        reward=round(reward,4)
                        #print(f"Qsa: {self.policy[y,x][direction]} reward at next step: {self.get_reward_location(new_x,new_y)} best possible reward: {max(self.policy[new_y, new_x])} final reward: {reward}")
                    #bad reward if trying to run off map
                    else:
                        reward=-10

                    self.policy[y][x][direction]=reward


def policy_translation(agent, starting_location_focused=False):
    agent_policy=agent.policy.copy()
    start_location=agent.start_location
    targets=agent.targets.targets_list.copy()
    target_locations=[]
    for tree in targets:
        target_locations.append(tree.location)
    translated_policy = np.zeros((len(agent_policy), agent_policy[0].size), dtype='<U5')
    for y in range(len(agent_policy)):
        for x in range(agent_policy[0].size):
            directions=np.where(agent_policy[y,x]==max(agent_policy[y,x]))[0]

            for direction in directions:
                if direction==2:
                    arrow="\u2190"
                elif direction==0:
                    arrow="\u2191"
                elif direction==3:
                    arrow="\u2192"
                elif direction==1:
                    arrow="\u2193"
                translated_policy[y,x]+=arrow

            if (x,y) in target_locations:
                translated_policy[y,x] += '*'
            elif (x,y)==start_location and starting_location_focused:
                translated_policy[y, x] += '^'
            else:
                translated_policy[y, x] +=' '
    print(translated_policy)


def plot_policy(agent, policy_index):
    plt.figure()
    policy=agent.policies[policy_index]
    max_policy = np.zeros_like(policy, dtype=np.float32)
    for x in range(agent.size[0]):
            for y in range(agent.size[1]):
                max_policy[y,x] = max(policy[y,x])
                directions=np.where(policy[y,x]==max(policy[y,x]))[0]
                delta_x = 0
                delta_y = 0
                for direction in directions:
                    if direction==0:
                        x_offset = 0
                        y_offset = -0.1
                        x_width = 0
                        y_width = -0.35
                    elif direction==1:
                        x_offset = 0
                        y_offset = 0.1
                        x_width = 0
                        y_width = 0.35
                    elif direction==2:
                        x_offset = -0.1
                        y_offset = 0
                        x_width = -0.35
                        y_width = 0
                    elif direction==3:
                        x_offset = 0.1
                        y_offset = 0
                        x_width = 0.35
                        y_width = 0
                    plt.annotate('',xy=(x+x_offset+x_width,y+y_offset+y_width),xytext=(x+x_offset,y+y_offset),arrowprops={'arrowstyle':'->'})

    for target in agent.targets.targets_list:
        plt.plot(target.location[0], target.location[1], 'ro', markersize=8)
    plt.plot(agent.start_location[0], agent.start_location[1], 'mo', markersize=8)
    plt.imshow(max_policy, cmap='Purples', interpolation='nearest')
    plt.title('Agent Policy')
    plt.show()


def generate_random_point(grid_size: Tuple[int])-> Tuple[int]:
    """
    Selects a random position within the grid of size grid_size
    """
    assert len(grid_size) == 2
    return random.choice(range(grid_size[0])), random.choice(range(grid_size[1]))


def is_valid_tree(pt, grid, exclusion_zone):
    w,h  = grid
    if pt[0] < 0 or pt[1] < 0:
        return False
    if pt[0] >= w or pt[1] >= h:
        return False
    if exclusion_zone:
        if pt[0] < exclusion_zone[0] and pt[1] < exclusion_zone[1]:
            return False
    return True


def generate_trees(grid_size, num_trees, exclusion_zone = None) -> Tuple[int, int]:


    # distribute trees evenly along the space, adding randomness and redrawing for the exclusion zone
    trees_placed = 0
    tree_positions = []
    w,h = grid_size
    squared_distribution = int(np.sqrt(num_trees))
    x_positions = [w//squared_distribution*i for i in range(squared_distribution)]
    y_positions = [h//squared_distribution*i for i in range(squared_distribution)]
    for x_pos in x_positions:
        for y_pos in y_positions:
            position = (x_pos + random.randint(-2,2), y_pos + random.randint(-2,2))
            if is_valid_tree(position, grid_size, exclusion_zone):
                tree_positions.append(position)
                trees_placed += 1

    while trees_placed < num_trees:
        # take a guess at a location
        position = (random.choice(range(grid_size[0])), random.choice(range(grid_size[1])))
        if is_valid_tree(position, grid_size, exclusion_zone) and position not in tree_positions:
            tree_positions.append(position)
            trees_placed += 1

    return tree_positions
    

def testing(eps):
    # keeping this here so we can use it to determine random points
    grid_size = (15,15)

    rover_start_area = (grid_size[0]//4, grid_size[1]//4)
    num_targets = 5
    # two targets could be at the same location....eh, as grid size expands, this won't be likely
    targets=targetsObj(generate_trees(grid_size, num_targets, rover_start_area))

    num_agents = 3
    agents = [newAgent(generate_random_point(rover_start_area), targets) for _ in range(num_agents)]
    gWorld = newGridWorld(agents, grid_size)
    for agent in gWorld.agents:
        agent.start_policy()
        agent.policies.append(agent.policy)
    iterations=80
    epochs=1000
    epsilon = eps
    global_rewards=[]
    reward_strings=[]
    for epoch in range(epochs):
        gWorld.global_reward=0
        gWorld.reward_string=' '
        #reset path traveled and it tree is measured every epoch
        for agent in gWorld.agents:
            agent.path=[]
            # reset to a new random position
            agent.start_location = generate_random_point(rover_start_area)
            agent.location = agent.start_location
            agent.policy=agent.policies[0]
            agent.trees_visited=[]
        for target in targets.targets_list:
            target.fully_measured=False
            target.visits=0
            target.visitors=[0 for _ in gWorld.agents]
            target.reward_value=REWARD_VALUE
        gWorld.done=False
        for iteration in range(iterations):
            if not gWorld.done:
                gWorld.step()

            if gWorld.done:
                print(f"moves to solve: {iteration}")
                reward_strings.append(gWorld.reward_string)
                break
        epsilon *= .99
        #policy_translation(agents[0], True)
        # if epoch%10==0:
            # gWorld.viz()
        #green: visited target, red: unvisited target, blue: path increasing darkness withtime, yellow: start location w/agent number
        global_rewards.append(gWorld.global_reward)
    #print(reward_strings)
    plt.plot(global_rewards)
    plt.title('Global Rewards vs Epochs Using A* Propagation and One Policy')
    plt.xlabel('Epochs')
    plt.ylabel('Global Reward')
    gWorld.path_viz()
    # plot_policy(agents[0],0)
    # plot_policy(agents[0],1)

    policy_translation(agents[0], True)


if __name__ == "__main__":
    testing(EPSILON)
    #need to explore then propogate rewards with q learning