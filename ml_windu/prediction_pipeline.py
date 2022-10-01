class PredictionPipeline:

    def __init__(self,
                 model,
                 data_schema,
                 featurize_fn,
                 featurize_fn_kwargs,
                 transformer,
                 transform_fn,
                 transform_fn_kwargs,
                 target_encoder,
                 target_decode_fn,
                 predict_fn,
                 predict_fn_kwargs):
        self.model = model
        self.data_schema = data_schema
        self.featurize_fn = featurize_fn
        self.featurize_fn_kwargs = featurize_fn_kwargs
        self.transformer = transformer
        self.transform_fn = transform_fn
        self.transform_fn_kwargs = transform_fn_kwargs
        self.target_encoder = target_encoder
        self.target_encoder_decode_fn = target_decode_fn
        self.predict_fn = predict_fn
        self.predict_fn_kwargs = predict_fn_kwargs
