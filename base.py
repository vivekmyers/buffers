import abc
import jax

class Buffer(abc.ABC):
    @abc.abstractmethod
    def add(self, **kwargs):
        pass

    @abc.abstractmethod
    def extend(self, **kwargs):
        pass

    @abc.abstractmethod
    def sample(self, key: jax.random.PRNGKey):
        pass


