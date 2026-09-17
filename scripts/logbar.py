class LogBar:
    @classmethod
    def shared(cls): return cls()
    def __init__(self, *args, **kwargs): pass
    def pb(self, iterable, **kwargs): return iterable
    def spinner(self, *args, **kwargs): return self
    def attach(self, *args, **kwargs): return self
    def set(self, **kwargs): return self
    def title(self, *args, **kwargs): return self
    def subtitle(self, *args, **kwargs): return self
    def draw(self, *args, **kwargs): return self
    def refresh(self): return self
    def next(self, *args, **kwargs): return self
    def close(self): return None
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
