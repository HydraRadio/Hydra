
import numpy as np
from pyuvdata import UVData


def load_uvdata_metadata(comm, fname):
    """
    Load metadata from a UVData-compatible file and distribute 
    it to all MPI workers.

    Parameters:
        comm (MPI Communicator):
            Optional MPI communicator. The root node will load the metadata 
            and broadcast it to all other workers.
        fname (str):
            Path to the data file, which should be a UVH5 file that supports 
            partial loading.

    Returns:
        data_info (dict):
            Dictionary with several named properties of the data.
    """
    myid = 0
    if comm is not None:
        myid = comm.Get_rank()
    
    # Root worker to load metadata and distribute it
    if myid == 0:
        uvd = UVData()
        uvd.read(fname, read_data=False) # metadata only

        # Get frequency and LST arrays
        freqs = np.unique(uvd.freq_array) / 1e6 # MHz
        lsts = np.unique(uvd.lst_array)

        # Get array latitude etc.
        lat, lon, alt = uvd.telescope_location_lat_lon_alt_degrees

        # Get array baselines
        bl_ints = uvd.get_baseline_nums() # Only baselines with data
        antpairs = []
        for bl in bl_ints:
            a1, a2 = uvd.baseline_to_antnums(bl)

            # Exclude autos
            if a1 != a2:
                antpairs.append((a1, a2))

        ants1, ants2 = zip(*antpairs)

        # Get array antenna locations
        ant_ids_in_order = uvd.antenna_numbers
        ant_ids = np.unique(np.concatenate((ants1, ants2)))
        ants = {ant: list(uvd.antenna_positions[ant_ids_in_order == ant,:]) 
                for ant in ant_ids}

        # Put data in dict with named fields to avoid ambiguity
        # Use built-in Python datatypes to help MPI
        data_info = {
            'freqs':    list(freqs),
            'lsts':     list(lsts),
            'lat':      float(lat),
            'lon':      float(lon),
            'alt':      float(alt),
            'antpairs': antpairs,
            'ants1':    list(ants1),
            'ants2':    list(ants2),
            'ants':     ants
        }

        # Return data immediately if MPI not enabled
        if comm is None:
            return data_info
    else:
        # Start with empty object for all other workers 
        data_info = None

    # Broadcast data to all workers
    data_info = comm.bcast(data_info, root=0)
    return data_info


def partial_load_uvdata(fname, freq_chunk, lst_chunk, antpairs, pol='xx'):
    """
    Load data from a UVData file and unpack into the expected format. 
    Uses the partial loading feature of UVH5 files.

    Parameters:
        fname (str):
            Path to the data file, which should be a UVH5 file that supports 
            partial loading.
        freqs (array_like):
            Data frequencies that this worker should load, in MHz.
        lsts (array_like):
            Data LSTs that this worker should load, in radians.
        bls (array_like):
            Data baselines that this worker should load. These should be 
            provided as antenna pairs.
        pol (str):
            Which polarisation to retrieve from the data.
    
    Returns:
        data (array_like):
            Array of complex visibility data, with shape 
            `(Nbls, Nfreqs, Nlsts)`.
        flags (array_like):
            Array of integer flags to apply to the data. Same shape as the 
            `data` array.
    """
    # Create new object
    uvd = UVData()
    uvd.read(fname, 
             frequencies=np.array(freq_chunk)*1e6, 
             lsts=lst_chunk, 
             bls=antpairs)

    # Get data and flags
    data = np.zeros((len(antpairs), len(freq_chunk), len(lst_chunk)), 
                    dtype=np.complex128)
    flags = np.zeros((len(antpairs), len(freq_chunk), len(lst_chunk)), 
                     dtype=np.int32)
    
    # Loop over baselines and extract data
    for i, bl in enumerate(antpairs):
        ant1, ant2 = bl
        dd = uvd.get_data(ant1, ant2, pol)
        print(dd.shape, ant1, ant2)

        # squeeze='full' collapses length-1 dimensions
        data[i,:,:] = uvd.get_data(ant1, ant2, pol, squeeze='full').T
        flags[i,:,:] = uvd.get_flags(ant1, ant2, pol, squeeze='full').T
    return data, flags


def load_source_catalogue(fname, max_header_lines=20):
    """
    Load a source catalogue in an expected standard text file format. The 
    file should be formatted as follows:
    
     - Line 0: Header (starting with #) with comma-separated list of field names
     - Up to 20 optional header lines as key-value pairs separated by a comma, 
       e.g. `# ref_freq:300`
     - Data as comma-separated values.
    
    The required fields are:
     - `ra` and `dec`, equatorial coordinates in degrees
     - `flux`, the flux at the reference frequency, in Jy
     - `beta`, the spectral index of the power-law in frequency.

    Parameters:
        fname (str):
            Path to the catalogue file. This should be a comma-separated text 
            file with a header.
        max_header_lines (int):
            Maximum number of header lines to check for at the start of the 
            file. This only needs to be changed if you have more header lines 
            than the default maximum. If you have fewer header lines than 
            this, you don't need to change it.

    Returns:
        cat (dict):
            Dictionary containing arrays of values for each named field.
        meta (dict):
            Dictionary of metadata key:value pairs.
    """
    # Define required fields
    required = ['ra', 'dec', 'flux', 'beta']

    # Get the header
    with open(fname, 'r') as f:
        # Read first line and remove whitespace and leading/trailing characters
        header = f.readline()
        header = header.replace("#", "").replace("\n", "").replace(" ", "")
        fields = header.lower().split(",")

        # Get column number of each field
        field_map = {field: j for field, j in enumerate(fields)}

        # Check for metadata in subsequent lines
        metadata = {}
        for i in range(max_header_lines):
            # This will return a blank line if the end of the file is reached, 
            # so no need to test
            line = f.readline()
            if "#" and ":" in line:
                line = line.replace("#", "").replace("\n", "").replace(" ", "")
                vals = line.lower().split(":")
                key, val = vals[0], vals[1]
                metadata[key] = value

    # Check that required fields are present
    for req in required:
        if req not in field_map.keys():
            raise KeyError("Field '%s' was not found in catalogue file. "
                           "The following fields were found: %s" 
                           % (req, str(field_map.keys())))

    # Load data
    d = np.loadtxt(fname, comments='#', delimiter=',')
    assert d.shape[0] == len(field_map.keys()), \
        "Number of columns in data is different from header"

    # Re-pack catalogue into dict
    cat = {}
    for field in field_map:
        cat[field] = d[field_map[field]]

    return cat, metadata
    