import numpy
import torch
from scipy import stats

def find_closest(s, all_xyz_contours, contor_probs_all_cov):
    """
    Find closest contour, with a given contour coverage probability, of all passed contours to a given point *s*.
    Returns coverage probability of closest contour to s.
    """
    pass

def get_real_coverage_value(true_pos, xy_contours_for_coverage, actual_expected_coverage):
    """
    Calculate real coverage based on contours.
    """
    pass

def calculate_approximate_coverage(base_evals, dim, expected_coverage_probs):
    """
    Used by main class to calculate coverage for various scenarios.

    Returns: True coverage probs
             Twice logprobs
             chi2 CDF of true delta llh
    """

    gauss_log_eval_at_0=-(dim/2.0)*numpy.log(2*numpy.pi)
    actual_twice_logprob=2*(gauss_log_eval_at_0-base_evals)
  
    expected_twice_logprob=stats.chi2.ppf(expected_coverage_probs, df=dim)

    actual_coverage_probs=[]
   
    for ind,true_cov in enumerate(expected_coverage_probs):

        actual_coverage_probs.append(float(sum(actual_twice_logprob<expected_twice_logprob[ind]))/float(len(actual_twice_logprob)))

    return numpy.array(actual_coverage_probs), actual_twice_logprob, stats.chi2.cdf(actual_twice_logprob, df=dim) 
