import json
import numpy as np

QUAD_TASK_ASSIGN_N_PARAM = 10


def gen_time_dist(
    n_destinations: int,
    package_weight_dist: np.ndarray,
    time_sigma: int = 1000,
    T: int = 5000,
) -> np.ndarray:
    """Generate a distribution of time taken to travel to each destination

    Args:
        n_destinations (int): number of destinations
    """
    assert len(package_weight_dist) == n_destinations
    # For each destination, generate a gaussian at a random time
    dist_over_time = []
    for d in range(n_destinations):
        volume = package_weight_dist[d]
        t = np.random.randint(0, T)
        # Ensure the distribution is symmetric around `t`
        gaussian = np.exp(-((np.arange(1, T + 1) - t)**2) /
                          (2 * time_sigma**2))

        # Normalize the Gaussian to match the volume
        normalized_gaussian = gaussian / np.sum(gaussian) * volume
        dist_over_time.append(normalized_gaussian)
    dist_over_time = np.asarray(dist_over_time).T
    # dist_over_time /= np.sum(dist_over_time, axis=1, keepdims=True)
    return dist_over_time


def gen_dist(n_destinations: int, proportions: np.ndarray) -> np.ndarray:
    """Generate a distribution of packages going to each destination according
    the proportions s.t.
    high_prob, mid_prob, low_prob = proportions
    1. the first `high_prob` of the destinations have a prob of `low_prob`
    2. the second `mid_prob` of the destinations have a prob of `mid_prob`
    3. the last `low_prob` of the destinations have a prob of `high_prob`

    Reference: https://github.com/HarukiMoriarty/CAL-MAPF/blob/c3dd0359a88aa2f019718348d99c53dbcd9764ce/calmapf/src/graph.cpp#L5-L41

    Args:
        n_destinations (int): number of destinations
        proportions (np.ndarray): proportions of packages going to each destination
    """
    assert len(proportions) == 3
    high_prob, mid_prob, low_prob = proportions
    high_dest, mid_dest, low_dest = low_prob, mid_prob, high_prob

    def calculate_sum(start, end):
        sum = 0.0
        for i in range(start, end + 1):
            sum += 1.0 / (i + 1)
        return sum

    probs = [0] * n_destinations

    # The first `high_dest`% of destinations
    sum_high_dest = calculate_sum(0, int(n_destinations * high_dest) - 1)

    for i in range(int(n_destinations * high_dest)):
        probs[i] = high_prob / sum_high_dest * (1.0 / (i + 1))

    # For item 10, it has the same probability as item 9
    probs[int(n_destinations *
              high_dest)] = probs[int(n_destinations * high_dest) - 1]

    # The next `mid_dest`% of destinations
    sum_mid_dest = calculate_sum(
        int(n_destinations * high_dest),
        int(n_destinations * (high_dest + mid_dest)) - 1)
    for i in range(
            int(n_destinations * high_dest) + 1,
            int(n_destinations * (high_dest + mid_dest))):
        probs[i] = mid_prob / sum_mid_dest * (1.0 / (i + 1))

    # For item 30, it has the same probability as item 29
    probs[int(n_destinations * high_dest +
              mid_dest)] = probs[int(n_destinations * (high_dest + mid_dest)) -
                                 1]

    # The last `low_dest`% of destinations
    sum_low_dest = calculate_sum(int(n_destinations * (high_dest + mid_dest)),
                                 n_destinations - 1)
    for i in range(
            int(n_destinations * (high_dest + mid_dest)) + 1, n_destinations):
        probs[i] = low_prob / sum_low_dest * (1.0 / (i + 1))

    probs = np.asarray(probs, dtype=float)
    probs /= np.sum(probs)
    return probs


def gen_721_dist(n_destinations):
    """Given number of destinations, generate a fixed 721 package distribution

    Args:
        n_destinations (int): number of destinations
    """
    return gen_dist(n_destinations, np.array([0.7, 0.2, 0.1]))
    # def calculate_sum(start, end):
    #     sum = 0.0
    #     for i in range(start, end + 1):
    #         sum += 1.0 / (i + 1)
    #     return sum

    # probs = [0] * n_destinations

    # # The first 10% of destinations
    # sum_first_10 = calculate_sum(0, int(n_destinations * 0.1) - 1)

    # for i in range(int(n_destinations * 0.1)):
    #     probs[i] = 0.7 / sum_first_10 * (1.0 / (i + 1))

    # # For item 10, it has the same probability as item 9
    # probs[int(n_destinations * 0.1)] = probs[int(n_destinations * 0.1) - 1]

    # # The next 20% of destinations
    # sum_next_20 = calculate_sum(int(n_destinations * 0.1),
    #                             int(n_destinations * 0.3) - 1)
    # for i in range(int(n_destinations * 0.1) + 1, int(n_destinations * 0.3)):
    #     probs[i] = 0.2 / sum_next_20 * (1.0 / (i + 1))

    # # For item 30, it has the same probability as item 29
    # probs[int(n_destinations * 0.3)] = probs[int(n_destinations * 0.3) - 1]

    # # The last 70% of destinations
    # sum_last_70 = calculate_sum(int(n_destinations * 0.3), n_destinations - 1)
    # for i in range(int(n_destinations * 0.3) + 1, n_destinations):
    #     probs[i] = 0.1 / sum_last_70 * (1.0 / (i + 1))

    # probs = np.asarray(probs, dtype=float)
    # probs /= np.sum(probs)
    # return probs


def gen_532_dist(n_destinations):
    """Given number of destinations, generate a fixed 721 package distribution

    Args:
        n_destinations (int): number of destinations
    """
    return gen_dist(n_destinations, np.array([0.5, 0.3, 0.2]))


def get_package_dist(
    n_destinations,
    package_dist_type,
    package_path,
    T=5000,
    time_sigma=1000,
):
    """Generate the package distribution, i.e. the probability of sampling the
    package going to each of the destinations

    NOTE: This function is for testing purpose. The actual code that generate
    the distribution for the simulation is in WPPL/src/SortationSystem.cpp

    Args:
        n_destinations (int): number of destinations
        package_dist_type (str): type of package distribution

    Returns:
        package_dist (np.ndarray)
    """
    if package_dist_type == "uniform":
        return np.ones(n_destinations, dtype=float) / n_destinations
    elif package_dist_type == "721":
        return gen_721_dist(n_destinations)
    elif package_dist_type == "532":
        return gen_532_dist(n_destinations)
    elif package_dist_type == "kaggle_data":
        with open(package_path, "r") as f:
            package_dist = json.load(f)
        assert len(package_dist) == n_destinations
        return np.array(package_dist, dtype=float)
    elif package_dist_type == "721_time":
        package_dist = gen_721_dist(n_destinations)
        time_dist = gen_time_dist(n_destinations, package_dist, time_sigma, T)
        return time_dist
    else:
        raise NotImplementedError()


def get_packages(
    package_mode,
    package_dist_type,
    package_path,
    n_destinations,
):
    # Read in the packages
    if package_mode == "dist":
        # NOTE: `package_dist_weight` is also the volumes of the
        # packages corresponding to each destination
        package_dist_weight = get_package_dist(
            n_destinations,
            package_dist_type,
            package_path,
        )
        package_dist_weight_json = json.dumps(package_dist_weight.tolist())
        return package_dist_weight, package_dist_weight_json
    elif package_mode == "explicit":
        with open(package_path, "r") as f:
            # self.package_dist_weight = json.load(f)
            # TODO: read in list of packages explicitly and infer the
            # package distribution
            raise NotImplementedError()


# if __name__ == "__main__":
#     prob = gen_721_dist(100)
#     import matplotlib.pyplot as plt
#     plt.scatter(np.arange(100), sorted(prob, reverse=True))
#     plt.savefig("721.png")
