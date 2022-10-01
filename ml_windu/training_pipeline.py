import logging
from sklearn.metrics import f1_score
from ml_windu.default_udf import split, fit_transformer, transform, inverse_transform, fit_model, predict
from .prediction_pipeline import PredictionPipeline
from .cache_store import InMemoryCacheStore

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self,
                 model,
                 data_load_fn=None,
                 data_load_fn_kwargs={},
                 preprocess_fn=None,
                 preprocess_fn_kwargs={},
                 data_schema='infer',
                 featurize_fn=None,
                 featurize_fn_kwargs={},
                 split_fn=split,
                 split_fn_kwargs={},
                 transformer=None,
                 transform_fn=transform,
                 transform_fn_kwargs={},
                 target_encoder=None,
                 target_encoder_fit_fn=fit_transformer,
                 target_encode_fn=transform,
                 target_decode_fn=inverse_transform,
                 fit_model_fn=fit_model,
                 fit_model_fn_kwargs={},
                 predict_fn=predict,
                 predict_fn_kwargs={},
                 evaluate_fn=f1_score,
                 evaluate_fn_kwargs={},
                 autocache=False,
                 cache_store=InMemoryCacheStore(),
                 ):
        self.model = model
        self.data_load_fn = data_load_fn
        self.data_load_fn_kwargs = data_load_fn_kwargs
        self.preprocess_fn = preprocess_fn
        self.preprocess_fn_kwargs = preprocess_fn_kwargs
        self.data_schema = None if data_schema == 'infer' else data_schema
        self._inferred_data_schema = None if self.data_schema is None else data_schema
        self.featurize_fn = featurize_fn
        self.featurize_fn_kwargs = featurize_fn_kwargs
        self.split_fn = split_fn
        self.split_fn_kwargs = split_fn_kwargs
        self.transformer = transformer
        self.transform_fn = transform_fn
        self.transform_fn_kwargs = transform_fn_kwargs
        self.target_encoder = target_encoder
        self.target_encoder_fit_fn = target_encoder_fit_fn
        self.target_encoder_encode_fn = target_encode_fn
        self.target_encoder_decode_fn = target_decode_fn
        self.fit_model_fn = fit_model_fn
        self.fit_model_fn_kwargs = fit_model_fn_kwargs
        self.predict_fn = predict_fn
        self.predict_fn_kwargs = predict_fn_kwargs
        self.evaluate_fn = evaluate_fn
        self.evaluate_fn_kwargs = evaluate_fn_kwargs
        self._autocache = False
        self._cache_store = cache_store
        if autocache is True:
            self.enable_autocache()

    def run(self):
        raise NotImplementedError()

    def next_step(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def revert_to_step(self, name, number=None):
        raise NotImplementedError()

    def revert_to_last_cached_step(self):
        raise NotImplementedError()

    def cache_last_step(self):
        raise NotImplementedError()

    def enable_autocache(self):
        logger.warning('Enabling autocache can result in increased storage usage / memory consumption.')
        self._autocache = True

    def disable_autocache(self):
        self._autocache = False

    def load_data(self, data=None):
        if data is None:
            return self.data_load_fn(**self.data_load_fn_kwargs)

    def _infer_data_schema(self, df):
        raise NotImplementedError

    def to_prediction_pipeline(self):
        return PredictionPipeline(model=self.model,
                                  data_schema=self._inferred_data_schema,
                                  featurize_fn=self.featurize_fn,
                                  featurize_fn_kwargs=self.featurize_fn_kwargs,
                                  transformer=self.transformer,
                                  transform_fn=self.transform_fn,
                                  transform_fn_kwargs=self.transform_fn_kwargs,
                                  target_encoder=self.target_encoder,
                                  target_decode_fn=self.target_decode_fn,
                                  predict_fn=self.predict_fn,
                                  predict_fn_kwargs=self.predict_fn_kwargs
                                  )
