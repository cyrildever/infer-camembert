# Private functions

import numpy as np

from infercamembert.labels import Labels


def get_preds_from_logits(logits, threshold):
    """
    Build the predictions based on the passed logits
    """
    ret = np.zeros(logits.shape)

    ret[:, Labels.ALL_INDICES] = (logits[:, Labels.ALL_INDICES] >= threshold).astype(
        int
    )

    return ret
