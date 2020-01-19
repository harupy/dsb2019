from .base import BaseModel
import lightgbm as lgb


class LgbModel(BaseModel):

    def fit(self, X_trn, y_trn, X_val, y_val, config):
        trn_set = lgb.Dataset(X_trn, y_trn)
        val_set = lgb.Dataset(X_val, y_val)
        eval_result = {}
        model = lgb.train(config.params, trn_set,
                          valid_sets=[trn_set, val_set],
                          valid_names=['train', 'valid'],
                          callbacks=[lgb.record_evaluation(eval_result)],
                          **config.fit)

        return model, eval_result

    def predict(self, model, X):
        return model.predict(X)

    def feature_importance(self, model, importance_type):
        features = model.feature_name()
        importance = model.feature_importance(importance_type=importance_type)
        return features, importance
