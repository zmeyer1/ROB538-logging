import yaml, sys
import rover_domain_python

# https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self # fight me, I'm not using cmd line arguments

if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        with open(filename, 'r') as file:
            args = AttrDict(yaml.safe_load(file))
    except IndexError:
        print(f"Usage: python3 {sys.argv[0]} [yaml_file]")
        
    sim = rover_domain_python.RoverDomainVel(args)
    sim.reset()
    print(f"Local Rewards: {sim.get_local_reward()}")