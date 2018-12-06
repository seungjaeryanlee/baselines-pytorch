import gym
import torch

from .torch_wrappers import TorchWrapper


def test_TorchWrapper_reset_type():
    """
    Test if environment wrapped in TorchWrapper returns a `torch.Tensor` when
    `reset()` is called.
    """
    env = gym.make('CartPole-v0')
    env = TorchWrapper(env)

    state = env.reset()
    print('State type of wrapped env : ', type(state))
    assert type(state) == torch.Tensor


def test_TorchWrapper_reset_dtype():
    """
    Test if environment wrapped in TorchWrapper returns a `torch.Tensor` state
    with `torch.float32` datatype when `reset()` is called.
    """
    env = gym.make('CartPole-v0')
    env = TorchWrapper(env)

    state = env.reset()
    print('State dtype of wrapped env : ', state.dtype)
    assert state.dtype == torch.float32


def test_TorchWrapper_reset_shape():
    """
    Test if environment wrapped in TorchWrapper returns a `torch.Tensor` of the
    correct shape.
    """
    env = gym.make('CartPole-v0')
    wrapped_env = TorchWrapper(env)

    state = env.reset()
    wrapped_state = wrapped_env.reset()
    print('State shape of unwrapped env : ', state.shape)
    print('State shape of wrapped env   : ', wrapped_state.shape)

    assert wrapped_state.shape == torch.Size([1, state.shape[0]])


def test_TorchWrapper_step_type():
    """
    Test if environment wrapped in TorchWrapper returns `next_state, reward,
    done` with `torch.Tensor` types when `step()` is called.
    """
    env = gym.make('CartPole-v0')
    env = TorchWrapper(env)
    env.reset()

    next_state, reward, done, info = env.step(env.action_space.sample())
    print('S\'.shape  : ', type(next_state))
    print('R.shape   : ', type(reward))
    print('D.shape   : ', type(done))
    assert type(next_state) == torch.Tensor
    assert type(reward) == torch.Tensor
    assert type(done) == torch.Tensor


def test_TorchWrapper_step_dtype():
    """
    Test if environment wrapped in TorchWrapper returns `next_state, reward,
    done` with correct dtypes when `step()` is called.
    """
    env = gym.make('CartPole-v0')
    env = TorchWrapper(env)
    env.reset()

    next_state, reward, done, info = env.step(env.action_space.sample())
    print('S\'.dtype  : ', next_state.dtype)
    print('R.dtype   : ', reward.dtype)
    print('D.dtype   : ', done.dtype)
    assert next_state.dtype == torch.float32
    assert reward.dtype == torch.float32
    assert done.dtype == torch.float32


def test_TorchWrapper_step_shape():
    """
    Test if environment wrapped in TorchWrapper returns `next_state, reward,
    done` with correct dtypes when `step()` is called.
    """
    env = gym.make('CartPole-v0')
    wrapped_env = TorchWrapper(env)
    env.reset()
    wrapped_env.reset()

    next_state, reward, done, info = env.step(env.action_space.sample())
    w_next_state, w_reward, w_done, info = wrapped_env.step(
        wrapped_env.action_space.sample())
    print('S\'.shape  : ', w_next_state.shape)
    print('R.shape   : ', w_reward.shape)
    print('D.shape   : ', w_done.shape)

    assert w_next_state.shape == torch.Size([1, next_state.shape[0]])
    assert w_reward.shape == torch.Size([1])
    assert w_done.shape == torch.Size([1])
