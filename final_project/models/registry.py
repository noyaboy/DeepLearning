# Simple model registry.

_models = {}


class ModelRegistry:
    @classmethod
    def register(cls, name):
        def decorator(model_cls):
            _models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def build(cls, name, config):
        if name not in _models:
            raise ValueError(f"Unknown model: {name}")
        return _models[name](config)

    @classmethod
    def list_available(cls):
        return list(_models.keys())
