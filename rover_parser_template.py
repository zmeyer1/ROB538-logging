import yaml, sys
import rover_domain_python
import numpy as np

NUM_ITERATIONS=5
EPOCHS=3

# https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self  # fight me, I'm not using cmd line arguments


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        with open(filename, 'r') as file:
            args = AttrDict(yaml.safe_load(file))
    except IndexError:
        print(f"Usage: python3 {sys.argv[0]} [yaml_file]")

    sim = rover_domain_python.RoverDomainVel(args)
    agents = []
    for agent in range(args.num_agents):
        agents.append(rover_domain_python.RoverAgent(args))

    for k in range(EPOCHS):
        sim.reset()
        actions = []
        directions = []
        positions = []
        for i, agent in enumerate(agents):
            action, direction = agent.get_action(sim.rover_pos[i], sim.rover_vel[i])
            actions.append(action)
            directions.append(direction)
            position = sim.rover_pos[i][0:2]
            positions.append(position)
        for j in range(NUM_ITERATIONS):
            sim.step(np.array(actions))
            reward=sim.get_global_reward()-1
            for i, agent in enumerate(agents):
                new_position=sim.rover_pos[i][0:2]

                if int(new_position[0])!=int(positions[i][0]) or int(new_position[1])!=int(positions[i][1]):
                    agent.update_policy(reward,[int(p) for p in new_position], directions[i])
                    action, direction = agent.get_action(sim.rover_pos[i], sim.rover_vel[i])
                    actions[i]=action
                    directions[i]=direction
                    print('new square')
                else:
                    actions[i][1]= 0
                print(new_position, actions)
                positions[i]=new_position

    sim.viz()
    print(f"Local Rewards: {sim.get_local_reward()}")
