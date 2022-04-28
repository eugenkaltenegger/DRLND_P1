from unityagents import UnityEnvironment


class Environment:
    def __init__(self, world, graphics=True):
        environment_file_name = None
        if world == "Banana":
            environment_file_name = "./env/Banana_Linux/Banana.x86_64"
        if world == "VisualBanana":
            environment_file_name = "./env/VisualBanana_Linux/Banana.x86_64"

        self.env = UnityEnvironment(file_name=environment_file_name, no_graphics=not graphics)
        self.brain_names = self.env.brain_names
        self.default_brain_name = self.brain_names[0]

    def state_size(self):
        return self.env.brains[self.default_brain_name].vector_observation_space_size

    def action_size(self):
        return self.env.brains[self.default_brain_name].vector_action_space_size

    def get(self):
        return self.env

    def reset(self, brain_name=None):
        if brain_name is None:
            brain_name = self.default_brain_name
        env_info = self.env.reset(train_mode=True)[brain_name]
        return env_info.vector_observations[0]

    def action(self, action, brain_name=None):
        if brain_name is None:
            brain_name = self.default_brain_name
        info = self.env.step(action)[brain_name]
        return {'next_state': info.vector_observations[0],
                'reward': info.rewards[0],
                'done': info.local_done[0]}
