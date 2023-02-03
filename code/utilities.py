import numpyro
import jax.numpy as jnp
from jax.scipy.special import erf
import scipy
import numpy as np

logit_std = 2.5
tmp_max = 100.
tmp_min = 2.

def truncatedNormal(samples,mu,sigma,lowCutoff,highCutoff):

    """
    Jax-enabled truncated normal distribution
    
    Parameters
    ----------
    samples : `jax.numpy.array` or float
        Locations at which to evaluate probability density
    mu : float
        Mean of truncated normal
    sigma : float
        Standard deviation of truncated normal
    lowCutoff : float
        Lower truncation bound
    highCutoff : float
        Upper truncation bound

    Returns
    -------
    ps : jax.numpy.array or float
        Probability density at the locations of `samples`
    """

    a = (lowCutoff-mu)/jnp.sqrt(2*sigma**2)
    b = (highCutoff-mu)/jnp.sqrt(2*sigma**2)
    norm = jnp.sqrt(sigma**2*np.pi/2)*(-erf(a) + erf(b))
    ps = jnp.exp(-(samples-mu)**2/(2.*sigma**2))/norm
    return ps

def massModel(m1,alpha,mu_m1,sig_m1,f_peak,mMax,mMin,dmMax,dmMin):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    p_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Define power-law and peak
    p_m1_pl = (1.+alpha)*m1**(alpha)/(tmp_max**(1.+alpha) - tmp_min**(1.+alpha))
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter

def get_value_from_logit(logit_x,x_min,x_max):

    """
    Function to map a variable `logit_x`, defined on `(-inf,+inf)`, to a quantity `x`
    defined on the interval `(x_min,x_max)`. Here, the logit transform is given by

    $
    \begin{equation}
    \mathrm{logit}x = \log\left( \frac{x - x_\mathrm{min}}{ x_\mathrm{max} - x} \right)
    \end{equation}
    $

    Parameters
    ----------
    logit_x : float
        Quantity to inverse-logit transform
    x_min : float
        Lower bound of `x`
    x_max : float
        Upper bound of `x`

    Returns
    -------
    x : float
       The inverse logit transform of `logit_x`
    dlogit_dx : float
       The Jacobian between `logit_x` and `x`; divide by this quantity to convert a uniform prior on `logit_x` to a uniform prior on `x`
    """

    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return x,dlogit_dx

def build_ar1(total,new_element):

    """
    Helper function to iteratively construct an AR process, given a previous value and a new parameter/innovation pair.
    Used together with `jax.lax.scan`

    Parameters
    ----------
    total : float
        Processes' value at the previous iteration
    new_element : tuple
        Tuple `(c,w)` containing new parameter/innovation; see Eq. 4 of the associated paper

    Returns
    -------
    total : float
        AR process value at new point
    """

    c,w = new_element
    total = c*total+w
    return total,total

def truncated_gaussian(xs,mu,sigma,low_cutoff,high_cutoff):

    """
    Function defining the probability density due to a truncated Gaussian

    Parameters
    ----------
    xs : np.array
        Array of values at which to evaluate probability density
    mu : float
        Mean parameter of truncated normal
    sigma : float
        Standard deviation parameter of truncated normal
    low_cutoff : float
        Lower cutoff
    high_cutoff : float
        Upper cutoff

    Returns
    -------
    ys : np.array
        Corresponding probability densities
    """
    
    # Normalization
    a = (low_cutoff-mu)/np.sqrt(2*sigma**2)
    b = (high_cutoff-mu)/np.sqrt(2*sigma**2)
    norm = np.sqrt(sigma**2*np.pi/2)*(-scipy.special.erf(a) + scipy.special.erf(b))

    ys = np.exp(-(xs-mu)**2/(2.*sigma**2))/norm
    ys[xs<low_cutoff] = 0
    ys[xs>high_cutoff] = 0

    return ys

def calculate_gaussian_2D(chiEff, chiP, mu_eff, sigma2_eff, mu_p, sigma2_p, cov, chi_min=-1):

    """
    Function to evaluate our bivariate gaussian probability distribution on chiEff and chiP
    See e.g. http://mathworld.wolfram.com/BivariateNormalDistribution.html

    Parameters
    ----------
    chiEff : float
        Array of chi-effective values at which to evaluate probability distribution
    chiP : float      
        Array of chi-p values
    mu_eff : float     
        Mean of the BBH chi-effective distribution
    sigma2_eff : float
        Variance of the BBH chi-effective distribution
    mu_p : float
        Mean of the BBH chi-p distribution
    sigma2_p : float
        Variance of the BBH chi-p distribution
    cov : float
        Degree of covariance (off-diagonal elements of the covariance matrix are cov*sigma_eff*sigma_p)

    Returns
    -------
    y : `np.array`         
        Array of probability densities
    """

    dchi_p = 0.01
    dchi_eff = (1.-chi_min)/200

    chiEff_grid = np.arange(chi_min,1.+dchi_eff,dchi_eff)
    chiP_grid = np.arange(0.,1.+dchi_p,dchi_p)
    CHI_EFF,CHI_P = np.meshgrid(chiEff_grid,chiP_grid)


    # We need to truncate this distribution over the range chiEff=(-1,1) and chiP=(0,1)
    # Compute the correct normalization constant numerically, integrating over our precomputed grid from above
    norm_grid = np.exp(-0.5/(1.-cov**2.)*(
                    np.square(CHI_EFF-mu_eff)/sigma2_eff
                    + np.square(CHI_P-mu_p)/sigma2_p
                    - 2.*cov*(CHI_EFF-mu_eff)*(CHI_P-mu_p)/np.sqrt(sigma2_eff*sigma2_p)
                    ))
    norm = np.sum(norm_grid)*dchi_eff*dchi_p
    if norm<=1e-12:
        return np.zeros(chiEff.shape)

    # Now evaluate the gaussian at (chiEff,chiP)
    y = (1./norm)*np.exp(-0.5/(1.-cov**2.)*(
                            np.square(chiEff-mu_eff)/sigma2_eff
                            + np.square(chiP-mu_p)/sigma2_p
                            - (2.*cov)*(chiEff-mu_eff)*(chiP-mu_p)/np.sqrt(sigma2_eff*sigma2_p)
                            ))

    y[chiEff<chi_min] = 0.

    return y
