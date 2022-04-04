import numpy as np


def get_grid_coordinates(coords):
    return np.floor(coords).astype("int")


def modified_poisson_disk_sampling(
    N=100, r0=10, r_sigma=1, k=50, radius_type="default"
):
    """
    Implementation of the Poisson Disk Sampling algorithm, but modified so that for each
    point added the radius between points is sampled from a Gaussian distribution

    author: Leif Denby 2021

    :param N: grid size in number of pixels (assumed 2D square domain)
    :param r0: length-scale for distance between points
    :param r_sigma: std div for distance between points
    :param k: Number of iterations to find a new particle in an annulus between radius r and 2r from a sample particle.
    :param radiusType: Method to determine the distance to newly spawned particles. 'default' follows the algorithm of
                       Bridson (2007) and generates particles uniformly in the annulus between radius r and 2r.
                       'normDist' instead creates new particles at distances drawn from a normal distribution centered
                       around 1.5r with a dispersion of 0.2r.
    :return: nParticle: Number of particles in the sampling.
             particleCoordinates: 2d array containing the coordinates of the created particles.
             radii: radii for the sampled particles

    based on
    https://gitlab.com/abittner/poissondisksampling/-/blob/master/poissonDiskSampling/bridsonVariableRadius.py
    """

    def _gen_radius():
        return np.random.normal(r0, r_sigma)

    # Set-up background grid
    grid_height = grid_width = N
    grid = np.zeros((grid_height, grid_width))

    # Pick initial (active) point
    coords = (np.random.random() * grid_height, np.random.random() * grid_width)
    idx = get_grid_coordinates(coords)
    n_particles = 1
    grid[idx[0], idx[1]] = n_particles

    # Initialise active queue
    # Appending to list is much quicker than to numpy array, if you do it very often
    queue = [coords]
    # List containing the exact positions of the final particles
    particle_coordinates = [coords]
    radii = [_gen_radius()]
    active_radii = [radii[0]]

    # Continue iteration while there is still points in active list
    while queue:

        # Pick random element in active queue
        idx = int(np.random.randint(len(queue)))
        r_active = active_radii[idx]
        active_coords = queue[idx]

        success = False
        for _ in range(k):
            if radius_type == "default":
                # Pick radius for new sample particle ranging between 1 and 2 times the local radius
                new_radius = r_active * (np.random.random() + 1)
            elif radius_type == "normDist":
                # Pick radius for new sample particle from a normal
                # distribution around 1.5 times the local radius
                new_radius = r_active * np.random.normal(1.5, 0.2)

            # Pick the angle to the sample particle and determine its coordinates
            angle = 2 * np.pi * np.random.random()
            new_coords = np.zeros(2)
            new_coords[0] = active_coords[0] + new_radius * np.sin(angle)
            new_coords[1] = active_coords[1] + new_radius * np.cos(angle)

            # Prevent that the new particle is outside of the grid
            if not (
                0 <= new_coords[1] <= grid_width and 0 <= new_coords[0] <= grid_height
            ):
                continue

            # Check that particle is not too close to other particle
            new_grid_coords = get_grid_coordinates((new_coords[1], new_coords[0]))

            radius_there = (
                _gen_radius()
            )  # np.ceil(radius[newGridCoords[1], newGridCoords[0]])
            grid_range_x = (
                np.max([new_grid_coords[0] - radius_there, 0]).astype("int"),
                np.min([new_grid_coords[0] + radius_there + 1, grid_width]).astype(
                    "int"
                ),
            )
            grid_range_y = (
                np.max([new_grid_coords[1] - radius_there, 0]).astype("int"),
                np.min([new_grid_coords[1] + radius_there + 1, grid_height]).astype(
                    "int"
                ),
            )

            search_grid = grid[
                slice(grid_range_y[0], grid_range_y[1]),
                slice(grid_range_x[0], grid_range_x[1]),
            ]
            conflicts = np.where(search_grid > 0)

            if len(conflicts[0]) == 0 and len(conflicts[1]) == 0:
                # No conflicts detected. Create a new particle at this position!
                queue.append(new_coords)
                active_radii.append(radius_there)
                radii.append(radius_there)
                particle_coordinates.append(new_coords)
                n_particles += 1
                grid[new_grid_coords[1], new_grid_coords[0]] = n_particles
                success = True

            else:
                # There is a conflict. Do NOT create a new particle at this position!
                continue

        if not success:
            # No new particle could be associated to the currently active particle.
            # Remove current particle from the active queue!
            del queue[idx]
            del active_radii[idx]

    return (n_particles, np.array(particle_coordinates), np.array(radii))


def create_circular_mask(h, w):
    center = (int(w / 2), int(h / 2))
    radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask
