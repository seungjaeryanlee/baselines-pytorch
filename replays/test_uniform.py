import gym
import numpy as np
import torch

from .uniform import UniformReplayBuffer


def test_uniform_batch_type():
    """
    Test if `replay_buffer.sample()` returns five `torch.Tensor` objects.
    """
    BATCH_SIZE = 1
    STATE_LEN = 4
    replay_buffer = UniformReplayBuffer(BATCH_SIZE)

    for i in range(BATCH_SIZE):
        state = torch.FloatTensor([[0, 0, 0, 0]])
        action = 0
        reward = torch.FloatTensor([1])
        next_state = torch.FloatTensor([[0, 0, 0, 0]])
        done = torch.FloatTensor([0])
        replay_buffer.push(state, action, reward, next_state, done)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(1)

    print('S  : ', state_batch)
    print('A  : ', action_batch)
    print('R  : ', reward_batch)
    print('S\' : ', next_state_batch)
    print('D  : ' , done_batch)
    assert type(state_batch) == torch.Tensor
    assert type(action_batch) == torch.Tensor
    assert type(reward_batch) == torch.Tensor
    assert type(next_state_batch) == torch.Tensor
    assert type(done_batch) == torch.Tensor

def test_uniform_batch_shape():
    """
    Test if `replay_buffer.sample()` returns five `torch.Tensor` objects with correct shapes.
    """
    BATCH_SIZE = 2
    STATE_LEN = 4
    replay_buffer = UniformReplayBuffer(BATCH_SIZE)

    for i in range(BATCH_SIZE):
        state = torch.FloatTensor([[0, 0, 0, 0]])
        action = 0
        reward = torch.FloatTensor([1])
        next_state = torch.FloatTensor([[0, 0, 0, 0]])
        done = torch.FloatTensor([0])
        replay_buffer.push(state, action, reward, next_state, done)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(BATCH_SIZE)

    print('S  : ', state_batch)
    print('A  : ', action_batch)
    print('R  : ', reward_batch)
    print('S\' : ', next_state_batch)
    print('D  : ' , done_batch)
    assert state_batch.shape == torch.Size([BATCH_SIZE, STATE_LEN])
    assert action_batch.shape == torch.Size([BATCH_SIZE])
    assert reward_batch.shape == torch.Size([BATCH_SIZE])
    assert next_state_batch.shape == torch.Size([BATCH_SIZE, STATE_LEN])
    assert done_batch.shape == torch.Size([BATCH_SIZE])
