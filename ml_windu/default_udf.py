from sklearn.model_selection import train_test_split


def split(df, test_size, random_state=None, stratify=None):
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    return df_train, df_test


def fit_transformer(transformer, data):
    transformer.fit(data)
    return transformer


def transform(transformer, data):
    return transformer.transform(data)


def inverse_transform(df, transformer):
    return transformer.inverse_transform(df)


def fit_model(model, features, targets):
    model.fit(features, targets)
    return model


def predict(df, model):
    return model.predict(df)
