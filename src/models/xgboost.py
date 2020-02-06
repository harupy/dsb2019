from .base import BaseModel
import xgboost as xgb


class XgbModel(BaseModel):

    def fit(self, X_trn, y_trn, X_val, y_val, config):
        # Ref.: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train
        dtrn = xgb.DMatrix(X_trn, y_trn)
        dval = xgb.DMatrix(X_val, y_val)
        eval_result = {}
        model = xgb.train(config.params, dtrn,
                          evals=[(dtrn, 'train'), (dval, 'valid')],
                          callbacks=[xgb.callback.record_evaluation(eval_result)],
                          **config.fit)
        return model, eval_result

    def predict(self, model, X):
        return model.predict(xgb.DMatrix(X))

    def feature_importance(self, model, importance_type):
        """
        Note that feature importance must be aligned to feature names.
        """
        scores = model.get_score(importance_type=importance_type)
        return [scores.get(fn, 0) for fn in model.feature_names]

    def feature_name(self):
        return self.models[0].feature_names
