import numpy as np
from skimage.measure import shannon_entropy


def generate_seeds(lab_image: np.ndarray,
                   block_size=(100, 100),
                   entropy_tr=8,
                   print_status=True,
                   mode_hx=10.0,
                   mode_hc=10.0,
                   mode_tolerance=1e-3):

    """
    Generate adaptively placed list of seeds to use for SNIC superpixel algorithm.

    Based on the publication:

    "Adaptive Centroid Placement Based SNIC for Superpixel Segmentation"

    https://ieeexplore.ieee.org/document/9185361

    Cite This:

    @INPROCEEDINGS{9185361,
    author={Bandara Senanayaka, Janith and Thilanka Morawaliyadda, Dilshan and Tharuka Senarath, Shehan and Indika Godaliyadda, Roshan and Parakrama Ekanayake, Mervyn},
    booktitle={2020 Moratuwa Engineering Research Conference (MERCon)},
    title={Adaptive Centroid Placement Based SNIC for Superpixel Segmentation},
    year={2020},
    volume={},
    number={},
    pages={242-247},
    doi={10.1109/MERCon50084.2020.9185361}}

    :param lab_image: CIELAB color image (lab_image should be numpy.ndarray, not a list as for snic)
    :param block_size: size of the blocks which are used to run mean-shift for mode seeking
    :param entropy_tr: threshold value of the entropy
    :param print_status: print the current progress as a percentage of number of block processed out of total
    :param mode_hx: bandwidth value for spatial features in mean-shift (default: 10.0)
    :param mode_hc: bandwidth value for color features in mean-shift (default: 10.0)
    :param mode_tolerance: tolerance to check convergence of mean-shift (default: 1e-3)
    :return: list of seeds to use for SNIC superpixel segmentation algorithm
    """

    # prepend locations to each pixel
    feature_map = __prepend_coordinates(lab_image)
    # create array of image blocks
    block_array = __create_blocks(feature_map, block_size)

    seeds = []
    num_blocks_processed = 0
    num_blocks = len(block_array)
    for block in block_array:

        # calculate the entropy of the block image
        entropy = __calculate_entropy(block)

        if entropy > entropy_tr:
            # seek modes
            modes = __seek_modes(block, int(round(entropy)), hx=mode_hx, hc=mode_hc, tolerance=mode_tolerance)
            seeds.extend(modes)

        # status update
        num_blocks_processed += 1
        if print_status:
            print("Processed %05.2f%%" % (num_blocks_processed * 100 / num_blocks))

    seeds = np.array(seeds).reshape((-1, 2))

    # status update
    print("\nSeeds generation completed!\n")

    return seeds.tolist()


def mark_seed_points(rgb_image: np.ndarray, seeds: np.ndarray, fill_color=None, border_color=None, outer_radius=6, inner_radius=3):
    """
    Mark where the seeds are placed on the image.

    :param rgb_image: RGB image
    :param seeds: array of seeds
    :param fill_color: color to use for the inner circle of the seed marker (default: red [255, 0, 0])
    :param border_color: color to use for the outer border of the seed marker (default: white [255, 255, 255])
    :param outer_radius: radius of the outer border (default: 6)
    :param inner_radius: radius of the inner fill (default: 3)
    :return: RGB image with seeds marked on top the original image
    """
    if fill_color is None:
        fill_color = [255, 0, 0]
    if border_color is None:
        border_color = [255, 255, 255]
    
    image = rgb_image.copy()
    img_height, img_width = rgb_image.shape[:2]

    for x, y in seeds:
        # Calculate bounds with proper clamping to image dimensions
        y_start = max(0, y - outer_radius)
        y_end = min(img_height, y + outer_radius)
        x_start = max(0, x - outer_radius)
        x_end = min(img_width, x + outer_radius)
        
        # Draw the outer border
        image[y_start:y_end, x_start:x_end, :] = border_color
        
        # Calculate inner bounds
        y_start_inner = max(0, y - inner_radius)
        y_end_inner = min(img_height, y + inner_radius)
        x_start_inner = max(0, x - inner_radius)
        x_end_inner = min(img_width, x + inner_radius)
        
        # Draw the inner fill
        image[y_start_inner:y_end_inner, x_start_inner:x_end_inner, :] = fill_color

    return image


def __prepend_coordinates(image: np.ndarray):
    """
    Prepend pixel location (col, row) to the image pixel color values (l,a,b).

    :param image: numpy array of an image
    :return: tensor with location data and color values of pixels (x,y,l,a,b)
    """
    # this function takes and image and add corresponding location coordinates to each pixel

    # image height and width
    img_height, img_width = image.shape[0:2]

    # create coordinates maps
    x = np.tile(np.arange(img_width), (img_height, 1))
    y = np.tile(np.arange(img_height), (img_width, 1)).transpose()

    return np.dstack((x, y, image))


def __create_blocks(image: np.ndarray, block_size: tuple):
    """
    Create an array of blocks with given dimensions from the given image.
    (more efficient than a nested for-loop)

    :param image: numpy array of a image
    :param block_size: (height, width) of the blocks to be created
    :return: numpy array of blocks with given dimensions
    """

    # image & block dimensions
    img_height, img_width, channels = image.shape
    block_height, block_width = block_size

    # crop the image so that image height and width to be integer multiples of block height and block width respectively
    img_height = block_height * (img_height // block_height)
    img_width = block_width * (img_width // block_width)
    image = image[:img_height, :img_width, :]

    # create array of blocks with given block dimensions
    block_array = image.reshape((img_height // block_height,
                                 block_height,
                                 img_width // block_width,
                                 block_width,
                                 channels))
    block_array = block_array.swapaxes(1, 2)
    block_array = block_array.reshape((-1, block_height, block_width, channels))

    return block_array


def __calculate_entropy(block: np.ndarray):
    """
    Calculate the entropy of the image block.

    :param block: image block with five dimensional data (x,y,l,a,b)
    :return: entropy value
    """

    entropy = sum((shannon_entropy(block[:, :, 2]),
                   shannon_entropy(block[:, :, 3]),
                   shannon_entropy(block[:, :, 4]))) / 3

    return entropy


def __seek_modes(block: np.ndarray, n: int, hx: float = 10.0, hc: float = 10.0, tolerance: float = 1e-3):
    """
    Seek the locations of the modes of the given image block using mean-shift.

    :param block: image block with five dimensional data (x,y,l,a,b)
    :param n: number of random samples to initialize mean-shift
    :param hx: bandwidth value for spatial features (default: 10.0)
    :param hc: bandwidth value for color features (default: 10.0)
    :param tolerance: tolerance to check convergence of mean-shift (default: 1e-3)
    :return: array of integer coordinates of the unique modes in the image block
    """
    if n <= 0:
        return np.array([]).reshape(0, 2).astype('int')
    
    if block.size == 0:
        return np.array([]).reshape(0, 2).astype('int')

    # create feature_set
    feature_set = block.reshape(-1, block.shape[-1])
    
    # Check if we have enough data points for sampling
    n_samples = len(feature_set)
    if n > n_samples:
        n = n_samples  # Use all available points if n is too large

    # select 'n' random data points to initialize mean shift
    indexes = np.random.choice(np.arange(n_samples), n, replace=False)
    centroids = feature_set[indexes]

    # diagonal bandwidth matrix
    sqrt_of_bandwidth_diagonal = np.array([hx, hx, hc, hc, hc])
    inv_sqrt_of_bandwidth_diagonal = 1 / sqrt_of_bandwidth_diagonal

    # mean-shift
    mode_locations = []
    for centroid in centroids:
        # Make sure we're working with float arrays to avoid numerical issues
        centroid = centroid.astype(np.float64)
        
        iteration = 0
        max_iterations = 1000  # Prevent infinite loops
        
        while iteration < max_iterations:
            norms = np.linalg.norm(inv_sqrt_of_bandwidth_diagonal * (feature_set - centroid), axis=1)
            weights = np.exp(-(norms ** 2) / 2)
            
            # Check if all weights are effectively zero (can happen with high bandwidths)
            if np.all(weights < 1e-10):
                # If all weights are negligible, keep the current centroid
                break
                
            # Calculate the weighted average
            new_centroid = np.average(feature_set, axis=0, weights=weights)
            new_centroid = new_centroid.astype(np.float64)

            # check for convergence
            if np.linalg.norm(new_centroid - centroid) < tolerance:
                break

            centroid = new_centroid
            iteration += 1

        # append only the spatial locations to the mode list
        mode_locations.append(np.floor(centroid[:2]))

    if len(mode_locations) == 0:
        return np.array([]).reshape(0, 2).astype('int')

    # retain only the unique mode locations
    unique_mode_locations = np.unique(np.array(mode_locations), axis=0)

    return unique_mode_locations.astype('int')
