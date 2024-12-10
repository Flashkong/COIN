# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import pprint
import sys
from collections.abc import Mapping
from detectron2.utils.logger import setup_logger
import pandas as pd
from detectron2.utils import comm

def print_csv_format(cfg, results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, Mapping) or not len(results), results
    logger = setup_logger(cfg.OUTPUT_DIR, name = __name__, distributed_rank = comm.get_rank())
    for task, res in results.items():
        if isinstance(res, Mapping):
            important_res = [(k, v) for k, v in res.items()]
            df = pd.DataFrame(important_res, columns=["Metric", "Value"])
            df["Task"] = task
            df = df[["Task"] + list(df.columns[:-1])]
            logger.info("\n"+df.to_markdown(index=False))
            per_class_res = ["{}:{} ".format(k.split("-")[1],round(v,4)) for k, v in res.items() if "-" in k]
            logger.info("".join(per_class_res))
        else:
            logger.info(f"copypaste: {task}={res}")


def verify_results(cfg, results):
    """
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}

    Returns:
        bool: whether the verification succeeds or not
    """
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True

    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task].get(metric, None)
        if actual is None:
            ok = False
            continue
        if not np.isfinite(actual):
            ok = False
            continue
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False

    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))

        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r
