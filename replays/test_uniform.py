import torch

from .uniform import UniformReplayBuffer


def test_uniform_batch_type():
    """
    Test if `replay_buffer.sample()` returns five `torch.Tensor` objects.
    """
    BATCH_SIZE = 1
    replay_buffer = UniformReplayBuffer(BATCH_SIZE)

    for i in range(BATCH_SIZE):
        state = torch.FloatTensor([[0, 0, 0, 0]])
        action = 0
        reward = torch.FloatTensor([1])
        next_state = torch.FloatTensor([[0, 0, 0, 0]])
        done = torch.FloatTensor([0])
        replay_buffer.push(state, action, reward, next_state, done)

    state_b, action_b, reward_b, next_state_b, done_b = replay_buffer.sample(1)

    print('S  : ', state_b)
    print('A  : ', action_b)
    print('R  : ', reward_b)
    print('S\' : ', next_state_b)
    print('D  : ', done_b)
    assert type(state_b) == torch.Tensor
    assert type(action_b) == torch.Tensor
    assert type(reward_b) == torch.Tensor
    assert type(next_state_b) == torch.Tensor
    assert type(done_b) == torch.Tensor


def test_uniform_batch_shape():
    """
    Test if `replay_buffer.sample()` returns five `torch.Tensor` objects with
    correct shapes.
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

    state_b, action_b, reward_b, next_state_b, done_b = replay_buffer.sample(
        BATCH_SIZE)

    print('S  : ', state_b)
    print('A  : ', action_b)
    print('R  : ', reward_b)
    print('S\' : ', next_state_b)
    print('D  : ', done_b)
    assert state_b.shape == torch.Size([BATCH_SIZE, STATE_LEN])
    assert action_b.shape == torch.Size([BATCH_SIZE])
    assert reward_b.shape == torch.Size([BATCH_SIZE])
    assert next_state_b.shape == torch.Size([BATCH_SIZE, STATE_LEN])
    assert done_b.shape == torch.Size([BATCH_SIZE])
