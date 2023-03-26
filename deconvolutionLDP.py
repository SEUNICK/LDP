import numpy as np

def get_est_noise_psd(pure_p, pure_q, domain_size, m, n_type="frequency"):
    """
    :param pure_p: p* of pure LDP protocol
    :param pure_q: q* of pure LDP protocol
    :param domain_size: domain size
    :param m: the number of users
    :param n_type: the type of noise(frequency or count)
    :return est_psd_n: the estimate power spectrum density of noise
    """
    if n_type == "frequency":
        expect_p_i = domain_size * pure_q * (1 - pure_q) / m + (pure_p - pure_q) * (1 - pure_p - pure_q) / m
    elif n_type == "count":
        expect_p_i = domain_size * pure_q * (1 - pure_q) * m + (pure_p - pure_q) * (1 - pure_p - pure_q) * m
    else:
        raise Exception("wrong n_type, n_type should be frequency or count")
    est_psd_n = np.full(domain_size, expect_p_i)
    return est_psd_n


def direct_wiener(h: np.ndarray, g: np.ndarray, est_psd_n: np.ndarray, f0: np.ndarray = None):
    """
    :param h: transfer Vector h
    :param g: total gathered suggest outputs
    :param est_psd_n: the estimate psd of noise
    :param f0: the estimated initial value of vector f, default is none
    :return est_f: the estimate of f after DW filter
    """
    fft_h = np.fft.fft(h)
    fft_g = np.fft.fft(g)

    if f0 is None:
        # When f0 is none, use fft_g / fft_h as the value of est_fft_f0, which is equivalent to using the est_f of
        # original aggregation step as the initial value
        est_fft_f0 = fft_g / fft_h
    else:
        est_fft_f0 = np.fft.fft(f0)

    est_psd_f = np.conj(est_fft_f0) * est_fft_f0
    wiener_filter = np.conj(fft_h) * est_psd_f / (fft_h * np.conj(fft_h) * est_psd_f + est_psd_n)
    est_fft_f = fft_g * wiener_filter
    est_f = np.real(np.fft.ifft(est_fft_f))

    return est_f


def improved_iterative_wiener(h: np.ndarray, g: np.ndarray, est_psd_n: np.ndarray, iter_time, f0: np.ndarray = None):
    """
    :param h: transfer Vector h
    :param g: total gathered suggest outputs
    :param est_psd_n: the estimate psd of noise
    :param iter_time: iteration times
    :param f0: the estimated initial value of vector f, default is none
    :return est_f: the estimate f after IIW filter
    """
    fft_g = np.fft.fft(g)
    fft_h = np.fft.fft(h)

    if f0 is None:
        est_fft_f0 = fft_g / fft_h
    else:
        est_fft_f0 = np.fft.fft(f0)

    est_psd_f0 = est_fft_f0 * np.conj(est_fft_f0)
    psd_h = fft_h * np.conj(fft_h)
    w_filter = np.conj(fft_h) * est_psd_f0 / (psd_h * est_psd_f0 + est_psd_n)

    est_fft_f = w_filter * fft_g
    est_f = np.real(np.fft.ifft(est_fft_f))

    est_psd_f1 = est_psd_f0
    for i in range(iter_time - 1):
        est_psd_f0 = est_psd_f1
        est_psd_f1_correction = est_psd_n * est_psd_f0 / (psd_h * est_psd_f0 + est_psd_n)
        est_psd_f1 = est_fft_f * np.conj(est_fft_f) + est_psd_f1_correction
        w_filter = np.conj(fft_h) * est_psd_f1 / (psd_h * est_psd_f1 + est_psd_n)
        est_fft_f = w_filter * fft_g
        est_f = np.real(np.fft.ifft(est_fft_f))
    return est_f


def average_multiple_random_permuted(h: np.ndarray, g: np.ndarray, est_psd_n: np.ndarray, map_time):
    """
    :param h: transfer Vector h
    :param g: total gathered suggest outputs
    :param est_psd_n: the estimate psd of noise
    :param map_time: random permuted times
    :return est_f: the estimate f after AMRP
    """
    rng = np.random.default_rng()
    domain_size = np.size(h)
    est_f_total = direct_wiener(h, g, est_psd_n)
    est_f_reverse = np.zeros(domain_size)

    for i in range(map_time - 1):
        map_index = rng.permutation(domain_size)
        random_permuted_g = g[map_index]
        est_random_permuted_f = direct_wiener(h, random_permuted_g, est_psd_n)
        for i in range(domain_size):
            est_f_reverse[map_index[i]] = est_random_permuted_f[i]
        est_f_total += est_f_reverse

    est_f = est_f_total / map_time
    return est_f