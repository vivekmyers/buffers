import jax
import jax.numpy as jnp
import optax
import haiku as hk
import functools
import collections

Batch = collections.namedtuple(
    "Batch",
    ["state", "action", "next_state", "future_state", "done", "idx", "future_idx"],
)

@jax.tree_util.register_pytree_node_class
class ContrastiveBuffer:
    def __init__(self, obs_dim, size, gamma, batch_size):
        super().__init__()
        self.obs_dim = obs_dim
        self.size = size
        self.gamma = gamma
        self.state_buffer = jnp.zeros((size, obs_dim))
        self.action_buffer = jnp.zeros((size,), dtype=jnp.int32)
        self.next_pos = 0
        self.buflen = 0
        self.batch_size = batch_size

    def tree_flatten(self):
        return (self.gamma, self.state_buffer, self.action_buffer, self.next_pos, self.buflen), (self.batch_size, self.obs_dim, self.size)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        gamma, state_buffer, action_buffer, next_pos, buflen = children
        batch_size, obs_dim, size = aux_data
        new = cls.__new__(cls)
        new.obs_dim = obs_dim
        new.size = size
        new.gamma = gamma
        new.state_buffer = state_buffer
        new.action_buffer = action_buffer
        new.next_pos = next_pos
        new.buflen = buflen
        new.batch_size = batch_size
        return new

    def add(self, obs, action):
        self.state_buffer = self.state_buffer.at[self.next_pos].set(obs)
        self.action_buffer = self.action_buffer.at[self.next_pos].set(action)
        self.next_pos += 1
        self.buflen = min(self.buflen + 1, self.size)
        if self.next_pos == self.size:
            self.next_pos = 0

    def extend(self, obs, action):
        if self.next_pos + len(obs) > self.size:
            self.extend(obs[: self.size - self.next_pos], action[: self.size - self.next_pos])
            self.extend(obs[self.size - self.next_pos :], action[self.size - self.next_pos :])
        else:
            self.state_buffer = self.state_buffer.at[self.next_pos:self.next_pos + len(obs)].set(obs)
            self.action_buffer = self.action_buffer.at[self.next_pos:self.next_pos + len(action)].set(action)
            self.next_pos += len(obs)
            self.buflen = min(self.buflen + len(obs), self.size)
            if self.next_pos >= self.size:
                self.next_pos = 0

    @jax.jit
    def _sample(self, key, params):
        state_buf, action_buf, next_pos, buflen = params
        key, subkey = jax.random.split(key)

        idx = jax.random.randint(key, (), 0, buflen)
        delta = jax.random.geometric(subkey, 1 - self.gamma)
        future_idx = (idx + delta) % buflen

        state = state_buf[idx]
        action = action_buf[idx]

        future_state = state_buf[future_idx]

        done = idx + 1 == next_pos
        next_state = state * done + state_buf[idx + 1] * (1 - done)

        return dict(
            state=state,
            action=action,
            next_state=next_state,
            future_state=future_state,
            done=done,
            idx=idx,
            future_idx=future_idx,
        )

    def sample(self, key):
        keys = jax.random.split(key, self.batch_size)
        samples = jax.vmap(self._sample, in_axes=(0, None))(
            keys, (self.state_buffer, self.action_buffer, self.next_pos, self.buflen)
        )
        return Batch(**samples)


