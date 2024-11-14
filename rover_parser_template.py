import yaml, sys, os
import rover_domain_python
import numpy as np
from typing import List, Tuple

NUM_ITERATIONS=500
EPOCHS=5

# https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self  # fight me, I'm not using cmd line arguments


def choose_actions(
        sim: rover_domain_python.RoverDomainVel,
        agents: List[rover_domain_python.RoverAgent]
    ) -> Tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((len(agents), 2))
    directions = np.zeros(len(agents), dtype=np.int8)

    for i,agent in enumerate(agents):
        actions[i,:], directions[i] = agent.get_action(sim.rover_pos[i], sim.rover_vel[i])

    return actions, directions


def update_policies(
        sim: rover_domain_python.RoverDomainVel,
        agents: List[rover_domain_python.RoverAgent],
        rewards: np.ndarray, # vector of floats of shape (num_agents, )
        directions: np.ndarray,
    ) -> None:

    for i,agent in enumerate(agents):
        position = sim.rover_pos[i][0:2]
        agent.update_policy(rewards[i], position, directions[i])


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        with open(filename, 'r') as file:
            args = AttrDict(yaml.safe_load(file))
    except IndexError:
        print(f"Usage: python3 {sys.argv[0]} [yaml_file]")
        print("Using default yaml")
        filename = os.path.join(os.getcwd(), 'ROB538-logging/rover_domain.yaml')
        with open(filename, 'r') as file:
            args = AttrDict(yaml.safe_load(file))


    sim = rover_domain_python.RoverDomainVel(args)
    agents = [rover_domain_python.RoverAgent(args) for _ in range(args.num_agents)]

    for k in range(EPOCHS):
        sim.reset()
        for j in range(NUM_ITERATIONS):
            
            # select actions
            actions, directions = choose_actions(sim, agents)

            # step forward the simulation
            sim.step(actions)

            # TODO; be smarter here
            rewards=(sim.get_global_reward()-1)*np.ones(args.num_agents)

            # update the policies
            update_policies(sim, agents, rewards, directions)

    sim.viz()
    print(f"Local Rewards: {sim.get_local_reward()}")
