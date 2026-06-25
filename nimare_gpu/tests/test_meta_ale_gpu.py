import os
import numpy as np
from nimare_gpu.ale import DeviceALE
from nimare.correct import FWECorrector
from nimare.dataset import Dataset
from nimare.extract import download_nidm_pain
from nimare.utils import get_resource_path
from nimare.meta.cbma import ALE

import pytest


# load the example dataset
dset_dir = download_nidm_pain()
dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file, target="mni152_2mm", mask=None)


def test_cpu_gpu_identity():
    """
    Test identity of ALE results on GPU vs CPU
    """
    results = {}
    corrected_results = {}
    estimators = {}
    estimators["cpu"] = ALE()
    estimators["gpu"] = DeviceALE(inv_step_size=100_000)
    fwe_correctors = {}
    fwe_correctors["cpu"] = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=100)
    fwe_correctors["gpu"] = FWECorrector(method="montecarlo", voxel_thresh=0.001, n_iters=100, batch_size=100)
    for hardware in estimators.keys():
        np.random.seed(0)
        results[hardware] = estimators[hardware].fit(dset)
        corrected_results[hardware] = fwe_correctors[hardware].transform(results[hardware])
    assert np.allclose(results['cpu'].estimator.null_distributions_['histogram_bins'], results['gpu'].estimator.null_distributions_['histogram_bins'])
    assert np.allclose(results['cpu'].maps['stat'], results['gpu'].maps['stat'])
    assert np.allclose(results['cpu'].maps['z'], results['gpu'].maps['z'])
    assert np.allclose(results['cpu'].maps['p'], results['gpu'].maps['p'])
    assert np.allclose(
        corrected_results['cpu'].maps['z_desc-mass_level-cluster_corr-FWE_method-montecarlo'], 
        corrected_results['gpu'].maps['z_desc-mass_level-cluster_corr-FWE_method-montecarlo']
    )
    assert np.allclose(
        corrected_results['cpu'].maps['z_desc-size_level-cluster_corr-FWE_method-montecarlo'], 
        corrected_results['gpu'].maps['z_desc-size_level-cluster_corr-FWE_method-montecarlo']
    )
    assert np.allclose(
        corrected_results['cpu'].maps['z_level-voxel_corr-FWE_method-montecarlo'], 
        corrected_results['gpu'].maps['z_level-voxel_corr-FWE_method-montecarlo']
    )