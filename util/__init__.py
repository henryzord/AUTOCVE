from AUTOCVE.AUTOCVE import get_unweighted_area_under_roc
import numpy as np


def unweighted_area_under_roc(score_handler, some_arg, y_true):
    score = get_unweighted_area_under_roc(
        y_true=np.array(y_true, copy=False, order='C', dtype=np.int64),
        y_score=np.array(score_handler.y_scores, copy=False, order='C', dtype=np.float64)
    )
    return score
