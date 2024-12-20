import numpy as np
import random
import sys
import matplotlib.pyplot as plt

ALPHA=0.95
GAMMA=0.4
EPSILON=0.2
COUPLING=1
REWARD_VALUE=100
class newGridWorld:
    def __init__(self,agents, size=(20, 20)):
        self.agents = agents
        self.targets=agents[0].targets
        self.size = size
        self.done=False
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
        for i, agent in enumerate(self.agents):
            reward = agent.get_reward()
            agent.update_policy(old_locations[i][0],old_locations[i][1], directions[i], reward)
        for i, target in enumerate(target_list):
            target_location = target.location
            for agent in self.agents:
                if agent.location == target_location:
                    # need to reset after each step
                    target.num_agents_present += 1
                    if i not in agent.trees_visited:
                        agent.trees_visited.append(i)
                        #new tree tracks if the location s new for the agent
                        new_tree=True
                    else:
                        new_tree=False
                    #go to next policy or start new one after visiting
                    if (len(agent.policies)<=len(agent.trees_visited)) and new_tree:
                        print('new policy created')
                        blank_policy=agent.start_policy()
                        agent.policies.append(blank_policy)
                    if len(agent.policies) <=len(target_list):
                        print('going to policy: ', len(agent.trees_visited))
                        agent.current_policy=len(agent.trees_visited)
                        agent.policy=agent.policies[len(agent.trees_visited)]
                    if target.num_agents_present >= COUPLING:
                        # trying to change value after collection
                        target.reward_value *= 0.8
                        #target.fully_measured=True
                        target.visits += 1
                        if i not in target.visitors:
                            target.visitors.append(i)
        for i, target in enumerate(target_list):
            if len(target.visitors)>=len(self.agents):
                target.fully_measured=True
        #if all are measured it's done
        if all([t.fully_measured for t in target_list]):
            self.done = True



    def viz(self):
        #might be switched
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
            plt.text(x,y, str(i))
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
        plt.imshow(grid, cmap=cmaps[i], interpolation='nearest')
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
        self.current_policy=0

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
                policy[y][x] = np.array([0.0,0.0,0.0,0.0])
        #self.policy = policy
        return policy

    def move_from_policy(self, avoid_visited=False):
        x = self.location[0]
        y = self.location[1]
        #call policy y,x
        val = random.random()
        if val>EPSILON:
            direction = random.choice(np.where(self.policy[y,x]==np.max(self.policy[y, x]))[0])
            allowed, x, y, visited=self.move(direction)
        else:
            direction = random.choice(range(4))
            allowed, x, y, visited = self.move(direction)
        if avoid_visited:
            #track allowed directions to choose from if all nearby tiles have been visited
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
            if target.location == (x,y) and i not in self.trees_visited:  #and target.num_agents_present>=COUPLING
                #not using coupling+ would have to change the order of establishing num agents present and getting reward
                reward+=target.reward_value
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
        #only propogates on most recently policy (maybe good?)
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
                        #reward=self.policy[y,x][direction]+ALPHA*(GAMMA*max(self.policy[new_y,new_x])-self.policy[y,x][direction])
                        reward = max(self.policy[new_y,new_x])*.9
                        reward=round(reward,2)
                        #print(f"Qsa: {self.policy[y,x][direction]}  best possible reward: {max(self.policy[new_y, new_x])} final reward: {reward}")
                    #bad reward if trying to run off map
                    else:
                        reward=0
                    self.policy[y][x][direction]=reward

        #above goes from top left to bottom right, need opposite as well
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for direction in range(4):
                    new_y=self.size[1]-1-y
                    new_x=self.size[0]-1-x
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
                        #reward=self.policy[y,x][direction]+ALPHA*(GAMMA*max(self.policy[new_y,new_x])-self.policy[y,x][direction])
                        reward = max(self.policy[new_y, new_x]) * .9
                        reward=round(reward,2)
                        #print(f"Qsa: {self.policy[y,x][direction]} reward at next step: {self.get_reward_location(new_x,new_y)} best possible reward: {max(self.policy[new_y, new_x])} final reward: {reward}")
                    #bad reward if trying to run off map
                    else:
                        reward=0
                    self.policy[self.size[1]-1-y][self.size[0]-1-x][direction]=reward

def policy_translation(policy, agent, starting_location_focused=False):
    agent_policy=policy.copy()
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
def testing(eps):
    targets=[(7,7), (15,15)] #,, (8,8)
    targets=targetsObj(targets)
    agent1=newAgent((3,2),targets)
    #agent2 = newAgent((4, 4), targets)
    #agent3 = newAgent((5, 4), targets)
    gWorld = newGridWorld([agent1]) #, agent2, agent3
    for agent in gWorld.agents:
        #start with a blank policy
        agent.policy=agent.start_policy()
        agent.policies.append(agent.policy)
    iterations=300
    epochs=100
    for epoch in range(epochs):
        #reset path traveled and it tree is measured every epoch
        epsilon = eps
        for agent in gWorld.agents:
            agent.path=[]
            agent.location=agent.start_location
            agent.trees_visited=[]
            agent.policy=agent.policies[0]
            agent.current_policy=0
            #for t in agent.targets.targets_list:
                #print(t.fully_measured)
        for target in targets.targets_list:
            target.fully_measured=False
            target.visitors=[]
            target.reward_value=REWARD_VALUE
        gWorld.done=False
        for iteration in range(iterations):
            if not gWorld.done:
                gWorld.step()
                epsilon*=.99
            if gWorld.done:
                print(f"moves to solve: {iteration}")
                break
        #policy_translation(agent1, True)
        if epoch%10==0:
            gWorld.viz()
        #green: visited target, red: unvisited target, blue: path increasing darkness withtime, yellow: start location w/agent number
        for agent in gWorld.agents:
            if gWorld.done:
                # print('current policy: ',agent.current_policy)
                # print(agent.policy)
                agent.propagate_reward()
                # print('after prop: ')
                # print(agent.policy)

    print(len(agent1.policies))
    for policy in agent1.policies:
        policy_translation(policy,agent1, True)


testing(EPSILON)
#need to explore then propogate rewards with q learning
#policy collapses with too many iterations
#something with back prop rule, just goes to 0
#only good for 10 epochs
#bad paths caused by deviation with e greedy, then no retrace goes randomly exploring
#make propogation sweep around
