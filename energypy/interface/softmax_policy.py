import os
import energypy
from energypy.interface.policies import collapse_episode_data
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp


class SoftmaxPolicy():
    def __init__(self, env_id, weights_dir=None):
        self.env = energypy.make_env(env_id)

        discrete_actions = self.env.action_space.discretize(20)

        self.net = tf.keras.Sequential([
            Dense(32, input_shape=self.env.observation_space.shape),
            Dense(16),
            Dense(len(discrete_actions), activation='softmax')
        ])

        self.trainable_variables = self.net.trainable_variables

    def load(self, filepath):
        print('loading model from {}'.format(filepath))
        self.net.load_weights(filepath)

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        print('saving model to {}'.format(filepath))
        self.net.save_weights(os.path.join(filepath, 'weights.h5'))

    def __call__(self, obs):
        probs = self.net(obs)
        return tf.random.categorical(probs, obs.shape[0])

    def log_prob(self, obs):
        return tf.log(self.net(obs))

    def get_loss(self, data):
        data = collapse_episode_data(data)
        #  check all same policy :)
        data = self.log_prob(data)
        return - tf.reduce_mean(data['log_prob'] * data['returns'])


if __name__ == '__main__':
    env_id = 'cartpole-v0'
    pol = SoftmaxPolicy(env_id)
    env = energypy.make_env(env_id)
    obs = env.reset()

    act = pol(obs)

