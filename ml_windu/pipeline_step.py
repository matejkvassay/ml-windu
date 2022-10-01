class PipelineStep:
    def __init__(self, number, name, input_spec, output_spec, docs, func, func_kwargs, cache):
        self.number = number
        self.name = name
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.docs = docs
        self.func = func
        self.func_kwargs = func_kwargs
        self.cache = cache
        self.cache_key = f'{number}_{name}'

    def execute(self, *data_in, store_to_cache=False):
        results = self.func(*data_in, **self.func_kwargs)
        if store_to_cache is True:
            self.cache.write(self.cache_key, results)
        return results
