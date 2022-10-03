# Machine Learning Windu

Top secret alien technology recovered from classified UFO impact site.

# Dev notes

### Metastore

Model artifacts (M = mandatory, O = optional):

- featurization input schema - M
- column order - M
- example of input batch -M
- fitted label encoder - O
- fitted transformers - O
- fitted model - M

Additionaly log:

- evaluation metric
- stats about data/training

### Steps

Training pipeline steps (M = mandatory, O = optional):

- load data - M
- preprocess - O
- featurize - M
- split - M
- fit transformers - O
- transform - O
- fit label encoder - O
- encode - O
- fit model - M
- predict - M
- evaluate - M

Steps shared with prediction pipeline

- featurize
- transform
- predict

### Desired feature ideas:

- declarative training / prediction pipeline definition via YAML config
- enable individual step execution in Jupyter or complete run 
- ability to implement multiple versions of steps and try them out using config
- ability to cache to metastore and run pipeline only from some point
- enable experiment flow with model selection - one train/dev/test split (possibly special eval pipeline???)
- support for binary, multiclass, multilabel and regression tasks
- 2 types of pipeline - production training / model selection pipeline (this could be left for Jupyter experiments)
- support for threshold selection
- make training optional to have only partial pipeline e.g. to prepare features & transformers???
- 1 command transformation from training to preprocess pipeline
- support configuration of training pipeline in code or by using YAML files
- storage and loading of artifacts and prediction pipeline (training as well possibly??)
- enable modification of featurization function (or other functions) without re-training of stored models for interactive mode
- enable inference of data schema during training
- prediction pipeline enables validation of data schema
- make everything modular, support different flavors for all trainable artifacts to be stored easily
- support multiple storage/load targets