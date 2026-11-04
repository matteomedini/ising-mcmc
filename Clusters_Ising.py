import numpy as np

# directions for the 4 connected neighbors on the lattice (down, right, up, left)
NEIGHBORS = [(0, 1), (1, 0), (-1, 0), (0, -1)]


def find_clusters(sample):
    N = sample.shape[0]
    mask = np.ones(N**2, dtype=bool)

    done = set()
    todo = set()

    # We walk the sites in order to pick the next seed instead of scanning the
    # whole mask every time with np.random.choice. The final list of cluster
    # sizes does not depend on the order we pick the seeds, but this keeps the
    # routine fast even when there are a lot of small clusters (high T, large N).
    seed = 0

    cluster_sizes = []
    while mask.sum() > 0:

        # move the seed pointer to the next site that has not been visited yet
        while not mask[seed]:
            seed += 1
        s0 = seed
        i0 = s0 // N; j0 = s0 % N

        sign = sample[i0, j0]  # find the sign of the spins in the cluster
        todo.add(s0)
        count = 1

        # grow the cluster: keep popping sites from todo, check their neighbors
        # and add the ones with the same sign that we have not seen before
        while len(todo) > 0:
            s = todo.pop()
            i = s // N; j = s % N
            for di, dj in NEIGHBORS:
                ii = (i + di) % N; jj = (j + dj) % N
                ss = N * ii + jj
                if sample[ii, jj] != sign:
                    continue
                if (ss not in done) and (ss not in todo) and mask[ss]:
                    todo.add(ss)
                    count += 1
            done.add(s)
            mask[s] = False

        cluster_sizes.append(count)

    return cluster_sizes
