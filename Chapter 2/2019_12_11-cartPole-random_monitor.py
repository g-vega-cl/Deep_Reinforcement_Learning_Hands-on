import gym


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, "recording") #Monitor lets you take a look at your agent inside the environment
        #Note "recording" is the folder where the monitor data will be saved to, it should not excist
        #Note 2, you need to have video environments xvfb, ffmpeg  , opengl, etc
        #conda install -c conda-forge ffmpeg
        #python 2019_12_11-cartPole-random_monitor.py
    total_reward = 0.0 
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
    env.close()
    env.env.close()