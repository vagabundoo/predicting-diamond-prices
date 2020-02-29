import h2o
h2o.init()
from h2o.estimators import deeplearning
from h2o.estimators import H2ORandomForestEstimator
from h2o.automl import H2OAutoML
import pandas as pd

aml = H2OAutoML(max_runtime_secs = 10, project_name = "diamonds", keep_cross_validation_models=True)
aml.train(y = 'price', training_frame = dum_train)