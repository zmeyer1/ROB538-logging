import numpy as np
import random
import sys
import matplotlib.pyplot as plt

ALPHA=0.95
GAMMA=0.2
EPSILON=0.2
COUPLING=1
class newGridWorld:
    def __init__(self,agents, size=(10, 5)):
        self.agents = agents
        self.targets=agents[0].targets
        self.size = size
        for agent in self.agents:
            agent.world=self
            agent.size=self.size

    def step(self):
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

        for target in target_list:
            target_location=target.location
            for agent in self.agents:
                if agent.location==target_location:
                    #need to reset after each step
                    target.num_agents_present+=1
                    if target.num_agents_present==COUPLING:
                        target.reward_value=0
                        target.fully_measured=True
        for i, agent in enumerate(self.agents):
            reward = agent.get_reward()
            agent.update_policy(old_locations[i][0],old_locations[i][1], directions[i], reward)

    def viz(self):

        #might be switched
        grid = np.zeros((self.size[1], self.size[0]))
        #move_strings=np.empty((self.size[1], self.size[0]), dtype='U50')
        for agent in self.agents:

            path=agent.path
            #makes grid have numbers
            #breaks for long paths
            val=0
            for entry in path:
                val+=1
                grid[entry[1],entry[0]]=val
        #         start_string=move_strings[entry[1],entry[0]]
        #         new_string=start_string+','+str(val)
        #         move_strings[entry[1],entry[0]]=new_string
        #
        # for y in range(self.size[1]):
        #     for x in range(self.size[0]):
        #         plt.text(x, y, move_strings[y,x])
        for target in self.targets.targets_list:
            x=target.location[0]
            y=target.location[1]
            #if target.fully_measured:
                #color=
            plt.gca().add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='red'))
        plt.imshow(grid, cmap='Blues', interpolation='nearest')
        plt.show()



#might be overkill, want to make sure it's a shared object
class targetsObj_old:
    def __init__(self, targets):
        self.targets_list=targets

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
        self.reward_value=30


class newAgent:
    def __init__(self, start_location, targets):
        self.policy = []
        self.location = start_location
        self.world = []
        self.size = []
        self.targets=targets
        self.start_location = start_location
        self.path=[]

    def move(self, instruction):
        x = self.location[0]
        y = self.location[1]
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
        return allowed, x, y
    def start_policy(self):
        policy = np.empty((self.size[1], self.size[0]), dtype=object)
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                #up down left right
                policy[y][x] = np.array([1.0,1.0,1.0,1.0])
        self.policy = policy

    def move_from_policy(self):
        x = self.location[0]
        y = self.location[1]
        #call policy y,x
        val = random.random()
        if val<EPSILON:
            direction = random.choice(np.where(self.policy[y,x]==np.max(self.policy[y, x]))[0])
            allowed, x, y=self.move(direction)
        else:
            direction = random.choice(range(4))
            allowed, x, y = self.move(direction)
        #try other directions if not allowed
        if not allowed:
            direction=0
            allowed, x, y = self.move(direction)
        while not allowed:
            direction+=1
            allowed, x, y = self.move(direction)
        if allowed:
            self.location = (x, y)
            # a little funky, only want to return allow direction
            return direction


    def get_reward(self):
        #UPDATE FUNCTION BELOW TOO
        x = self.location[0]
        y = self.location[1]
        reward=0
        for target in self.targets.targets_list:
            if target.location == (x,y) and target.num_agents_present>=COUPLING:
                reward+=target.reward_value
            else:
                reward+=0
        return reward

    #check reward at next location without actually changing x,y
    #used for propogation
    def get_reward_location(self,x,y):
        reward=0
        for target in self.targets.targets_list :
            if target.location == (x,y) and target.num_agents_present>=COUPLING:
                reward+=target.reward_value
            else:
                reward+=0
        return reward

    def update_policy(self,x,y,direction, reward):
        self.policy[y,x][direction] = reward

    def propagate_reward(self):
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for direction in range(4):
                    new_y=y
                    new_x=x
                    if direction == 0:
                        new_y -= 1
                    if direction == 1:
                        new_y += 1
                    if direction == 2:
                        new_x -= 1
                    if direction == 3:
                        new_x += 1
                    #check if possible move
                    if new_x >= 0 and new_x < self.size[0] and new_y >= 0 and new_y < self.size[1]:
                        reward=self.policy[y,x][direction]+ALPHA*(self.get_reward_location(new_x,new_y)+GAMMA*max(self.policy[new_y,new_x])-self.policy[y,x][direction])
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

#twoAgents()
#one_agent()
#system_level_reward()
def testing():
    targets=[(7,3), (1,1)]
    targets=targetsObj(targets)
    agent1=newAgent((3,2),targets)
    gWorld = newGridWorld([agent1])
    for agent in gWorld.agents:
        agent.start_policy()
    iterations=100
    epochs=4
    for epoch in range(epochs):
        #reset path traveled every epoch
        for agent in gWorld.agents:
            agent.path=[]
        for iteration in range(iterations):
            gWorld.step()
        #policy_translation(agent1, True)
        gWorld.viz()
        for agent in gWorld.agents:
            agent.propagate_reward()
    #print(agent1.policy)
    policy_translation(agent1, True)

testing()
#need to explore then propogate rewards with q learning


