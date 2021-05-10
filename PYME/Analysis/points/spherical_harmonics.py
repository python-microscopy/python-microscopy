"""
Estimate spherical harmonics from a point data set
Initial fitting/conversions ripped 100% from David Baddeley / scipy
"""
import numpy as np
from scipy.special import sph_harm
from scipy import linalg
from PYME.Analysis.points import coordinate_tools
from scipy import optimize
import logging

logger = logging.getLogger(__name__)


def r_sph_harm(m, n, azimuth, zenith):
    """
    return real valued spherical harmonics. Uses the convention that m > 0 corresponds to the cosine terms, m < zero the
    sine terms

    Parameters
    ----------
    m : int
        order of the spherical harmonic, |m| <= n
    n : int
        degree of the spherical harmonic, n >= 0

    azimuth : ndarray
        the azimuth angle in [0, 2pi]
    zenith : ndarray
        the elevation in [0, pi]

    Returns
    -------

    """
    if m > 0:
        return (1. / np.sqrt(2) * (-1) ** m) * sph_harm(m, n, azimuth, zenith).real
    elif m == 0:
        return sph_harm(m, n, azimuth, zenith).real
    else:
        return (1. / np.sqrt(2) * (-1.0) ** m) * sph_harm(m, n, azimuth, zenith).imag


def sphere_expansion(x, y, z, n_max=3):
    """
    Project coordinates onto spherical harmonics

    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    z : ndarray
        z coordinates
    n_max : int
        Maximum order to calculate to

    Returns
    -------

    modes : list of tuples
        a list of the (m, n) modes projected onto
    c : ndarray
        the mode coefficients


    """

    azimuth, zenith, r = coordinate_tools.cartesian_to_spherical(x, y, z)

    A = []
    modes = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            sp_mode = r_sph_harm(m, n, azimuth, zenith)
            A.append(sp_mode)

            modes.append((m, n))

    A = np.vstack(A)

    c = linalg.lstsq(A.T, r)[0]

    return modes, c


def sphere_expansion_clean(x, y, z, n_max=3, max_iters=2, tol_init=0.3):
    """
    Project coordinates onto spherical harmonics

    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    z : ndarray
        z coordinates
    n_max : int
        Maximum order to calculate to
    max_iters: int
        number of fit iterations
    tol_init: float
        relative outlier tolerance. Used to ignore outliers in subsequent iterations

    Returns
    -------

    modes : list of tuples
        a list of the (m, n) modes projected onto
    c : ndarray
        the mode coefficients


    """

    azimuth, zenith, r = coordinate_tools.cartesian_to_spherical(x, y, z)

    A = []
    modes = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            sp_mode = r_sph_harm(m, n, azimuth, zenith)
            A.append(sp_mode)

            modes.append((m, n))

    A = np.vstack(A).T

    tol = tol_init

    c = linalg.lstsq(A, r)[0]

    # recompute, discarding outliers
    for i in range(max_iters):
        pred = np.dot(A, c)
        error = abs(r - pred) / r
        mask = error < tol
        # print mask.sum(), len(mask)

        c, summed_residuals, rank, singular_values = linalg.lstsq(A[mask, :], r[mask])
        tol /= 2

    return modes, c, summed_residuals


AXES = np.stack([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], axis=1)


def reconstruct_shell(modes, coeffs, azimuth, zenith):
    r = 0
    for (m, n), c in zip(modes, coeffs):
        r += c * (r_sph_harm(m, n, azimuth, zenith))

    return r


def scaled_shell_from_hdf(hdf_file, table_name='harmonic_shell'):
    """

    Parameters
    ----------
    hdf_file : str or tables.file.File
        path to hdf file or opened hdf file
    table_name : str
        name of the table containing the spherical harmonic expansion information

    Returns
    -------
    shell : ScaledShell
        see nucleus.spherical_harmonics.shell_tools.ScaledShell

    """
    from PYME.IO.MetaDataHandler import HDFMDHandler
    from PYME.IO import tabular
    import tables
    try:
        opened_hdf_file = tables.open_file(hdf_file, 'r')
    except TypeError:
        opened_hdf_file = hdf_file
    
    shell_table = tabular.HDFSource(opened_hdf_file, table_name)
    shell_table.mdh = HDFMDHandler(opened_hdf_file)

    return ScaledShell.from_tabular(shell_table)

def generate_icosahedron():
    """
    Generate an icosahedron in spherical coordinates.
    
    Returns
    -------
        azimuth : np.array
            Length 12, azimuth of icosahedron vertices [0, 2*pi].
        zenith : np.array
            Length 12, zenith of icosahedron vertices [0, pi].
        faces : np.array
            20 x 3, counterclockwise triangles denoting icosahedron faces.
    """
    t = np.arctan(0.5)+np.pi/2
    zenith = np.hstack([0, np.array(5*[np.pi-t,t]).ravel(), np.pi])  # [0, pi]
    azimuth = np.hstack([0, np.arange(0, 2*np.pi, np.pi/5), 0])   # [0, 2*pi]
    idxs = np.arange(len(zenith))[1:-1]
    upper_middle_strip = np.vstack([[v0,v2,v1] for v0,v1,v2 in 
                                    zip(idxs[::2],idxs[1::2],np.roll(idxs[::2],-1))])
    lower_middle_strip = np.vstack([[v0,v2,v1] for v0,v1,v2 in 
                                    zip(np.roll(idxs[::2],-1),idxs[1::2],np.roll(idxs[1::2],-1))])
    upper_cap = np.vstack([[v0,v2,v1] for v0,v1,v2 in 
                        zip(np.zeros(5),idxs[::2],np.roll(idxs[::2],-1))])
    lower_cap = np.vstack([[v0,v2,v1] for v0,v1,v2 in 
                        zip(11*np.ones(5),np.roll(idxs[1::2],-1),idxs[1::2])])
    faces = np.vstack([upper_cap, upper_middle_strip, lower_middle_strip, lower_cap]).astype(np.int)
    
    return azimuth, zenith, faces

def icosahedron_mesh(n_subdivision=1):
    """
    Return an icosahedron subdivided n_subdivision times. This provides a
    quasi-regular sampling of the unit sphere.
    
            v0               v0
            /  \             /  \
           /    \    ==>   v3----v5
          /      \         / \  / \
         v1------v2       v1--v4--v2

    Returns
    -------
        azimuth : np.array
            Azimuth of mesh vertices [0, 2*pi].
        zenith : np.array
            Zenith of mesh vertices [0, pi].
        faces : np.array
            20 x 3, counterclockwise triangles denoting mesh faces.
    """
    
    # First, generate the icosahedron
    azimuth, zenith, faces = generate_icosahedron()
    
    if n_subdivision < 1:
        # We don't need to do anything
        return azimuth, zenith, faces
    
    # Convert pole azimuth to np.nan so we can ignore pole azimuth (irrelevant, detrimental)
    azimuth[0], azimuth[-1] = np.nan, np.nan
    
    # Make azimuth and zenith complex numbers (to deal with phase wrapping)
    azimuth, zenith = np.exp(1j*azimuth), np.exp(1j*zenith)
    
    for k in range(n_subdivision):
        # Average each edge to get new vertex at the center of each face v0,v1,v2
        az_v3 = np.nanmean(azimuth[faces[:,[0,1]]],axis=1)
        az_v4 = np.nanmean(azimuth[faces[:,[1,2]]],axis=1)
        az_v5 = np.nanmean(azimuth[faces[:,[2,0]]],axis=1)
        new_az = np.hstack([az_v3, az_v4, az_v5])
        ze_v3 = np.nanmean(zenith[faces[:,[0,1]]],axis=1)
        ze_v4 = np.nanmean(zenith[faces[:,[1,2]]],axis=1)
        ze_v5 = np.nanmean(zenith[faces[:,[2,0]]],axis=1)
        new_ze = np.hstack([ze_v3, ze_v4, ze_v5])
        
        # Create new vertices, eliminating duplicates (there should be two of each 
        # new vertex, one from each face sharing an edge)
        new_v, new_idxs = np.unique(np.vstack([new_az,new_ze]).T, axis=0, return_inverse=True)
        new_idxs += len(azimuth)
        new_idxs = new_idxs.reshape(3,-1).T  # v3 = new_idxs[:,0], v4 = new_idxs[:,1], v5 = new_idxs[:,2]
        
        # Create new faces (4 faces per old face)
        f0 = np.vstack([faces[:,0], new_idxs[:,0], new_idxs[:,2]]).T
        f1 = np.vstack([faces[:,1], new_idxs[:,1], new_idxs[:,0]]).T
        f2 = np.vstack([faces[:,2], new_idxs[:,2], new_idxs[:,1]]).T
        faces = np.vstack([f0,f1,f2,new_idxs]).astype(np.int)
        
        # Append the new vertices to azimuth
        azimuth = np.hstack([azimuth, new_v[:,0]])
        zenith = np.hstack([zenith, new_v[:,1]])
    
    # Restore poles to azimuth 0 for future conversion to real coordinates
    azimuth[np.isnan(azimuth)] = 0
    
    # Back to angles with you
    azimuth, zenith = np.angle(azimuth), np.angle(zenith)
    azimuth[azimuth<0] = azimuth[azimuth<0]+2*np.pi  # wrap to [0, 2*pi]
    zenith = np.abs(zenith) # wrap to [0, pi]
        
    return azimuth, zenith, faces


class ScaledShell(object):
    data_type = [
        ('modes', '<2i4'),
        ('coefficients', '<f4'),
    ]

    def __init__(self, sampling_fraction=1.):
        self.sampling_fraction = sampling_fraction

        self.modes = None
        self.coefficients = None

        self.x, self.y, self.z, = None, None, None
        self.x0, self.y0, self.z0 = None, None, None
        self.x_c, self.y_c, self.z_c, = None, None, None
        # note that all scalings will be centered
        self.x_cs, self.y_cs, self.z_cs, = None, None, None

        self.standard_deviations, self.principal_axes = None, None
        self.scaling_factors = None

    def to_recarray(self, keys=None):
        """

        Pretend we are a PYME.IO.tabular type

        Parameters
        ----------
        keys : None
            Ignored for this contrived function

        Returns
        -------
        numpy recarray version of self

        """
        record = np.recarray(len(self.coefficients), dtype=self.data_type)
        record['modes'] = self.modes
        record['coefficients'] = self.coefficients
        return record

    def to_hdf(self, filename, tablename='Data', keys=None, metadata=None):
        from PYME.IO import h5rFile, MetaDataHandler
        # NOTE that we ignore metadata input
        metadata = MetaDataHandler.NestedClassMDHandler()
        metadata['spherical_harmonic_shell.standard_deviations'] = self.standard_deviations.tolist()
        metadata['spherical_harmonic_shell.scaling_factors'] = self.scaling_factors.tolist()
        metadata['spherical_harmonic_shell.principal_axes'] = self.principal_axes.tolist()
        metadata['spherical_harmonic_shell.summed_residuals'] = self._summed_residuals
        metadata['spherical_harmonic_shell.n_points_used_in_fitting'] = len(self.x)
        metadata['spherical_harmonic_shell.x0'] = self.x0
        metadata['spherical_harmonic_shell.y0'] = self.y0
        metadata['spherical_harmonic_shell.z0'] = self.z0
        metadata['spherical_harmonic_shell.sampling_fraction'] = self.sampling_fraction

        with h5rFile.H5RFile(filename, 'a') as f:
            f.appendToTable(tablename, self.to_recarray(keys))
            f.updateMetadata(metadata)

    @staticmethod
    def from_tabular(shell_table):
        shell = ScaledShell()
        
        shell.standard_deviations = np.asarray(shell_table.mdh['spherical_harmonic_shell.standard_deviations'])
        shell.scaling_factors = np.asarray(shell_table.mdh['spherical_harmonic_shell.scaling_factors'])
        shell.principal_axes = np.asarray(shell_table.mdh['spherical_harmonic_shell.principal_axes'])

        shell.x0 = shell_table.mdh['spherical_harmonic_shell.x0']
        shell.y0 = shell_table.mdh['spherical_harmonic_shell.y0']
        shell.z0 = shell_table.mdh['spherical_harmonic_shell.z0']

        shell._summed_residuals = shell_table.mdh['spherical_harmonic_shell.summed_residuals']
        shell.sampling_fraction = shell_table.mdh['spherical_harmonic_shell.sampling_fraction']
        
        shell._set_coefficients(shell_table['modes'], shell_table['coefficients'])

        return shell

    def _set_coefficients(self, modes, coefficients):
        assert len(modes) == len(coefficients)
        self.modes = modes
        self.coefficients = coefficients

    def set_fitting_points(self, x, y, z):
        assert (x.shape == y.shape) and (y.shape == z.shape)
        self.x, self.y, self.z = np.copy(x), np.copy(y), np.copy(z)
        self.x0, self.y0, self.z0 = self.x.mean(), self.y.mean(), self.z.mean()

        self.x_c, self.y_c, self.z_c = self.x - self.x0, self.y - self.y0, self.z - self.z0

        self._scale_fitting_points()

    def _scale_fitting_points(self):
        self.standard_deviations, self.principal_axes = coordinate_tools.find_principle_axes(self.x_c, self.y_c, self.z_c,
                                                                                             sample_fraction=self.sampling_fraction)
        self.scaling_factors = np.max(self.standard_deviations) /(self.standard_deviations)
        self.x_cs, self.y_cs, self.z_cs, = coordinate_tools.scaled_projection(self.x_c, self.y_c, self.z_c,
                                                                              self.scaling_factors, self.principal_axes)

    def shell_coordinates(self, points):
        """Scale query points, projecting them onto the basis used in shell-
        fitting. Return in scaled spherical coordinates

        Parameters
        ----------
        points : 3-tuple
            x, y, z positions

        Returns
        -------
        azimuth, zenith, r
            scaled spherical coordinates of input points
        """
        x, y, z = points

        # scale the query points and convert them to spherical
        x_qs, y_qs, z_qs = coordinate_tools.scaled_projection(np.atleast_1d(x - self.x0), np.atleast_1d(y - self.y0),
                                                              np.atleast_1d(z - self.z0), self.scaling_factors,
                                                              self.principal_axes)
        
        azimuth_qs, zenith_qs, r_qs = coordinate_tools.cartesian_to_spherical(x_qs, y_qs, z_qs)
        return azimuth_qs, zenith_qs, r_qs

    def get_fitted_shell(self, azimuth, zenith):
        r_scaled = reconstruct_shell(self.modes, self.coefficients, azimuth, zenith)
        x_scaled, y_scaled, z_scaled = coordinate_tools.spherical_to_cartesian(azimuth, zenith, r_scaled)
        # need to scale things "down" since they were scaled "up" in the fit
        # scaling_factors = 1. / self.scaling_factors

        scaled_axes = self.principal_axes / self.scaling_factors[:, None]

        coords = x_scaled.ravel()[:, None] * scaled_axes[0, :] + y_scaled.ravel()[:, None] * scaled_axes[1,
                                                                                             :] + z_scaled.ravel()[:,
                                                                                                  None] * scaled_axes[2,
                                                                                                          :]
        x, y, z = coords.T

        return x.reshape(x_scaled.shape) + self.x0, y.reshape(y_scaled.shape) + self.y0, z.reshape(
            z_scaled.shape) + self.z0

    def fit_shell(self, max_n_mode=3, max_iterations=2, tol_init=0.3):
        modes, coefficients, summed_residuals = sphere_expansion_clean(self.x_cs, self.y_cs, self.z_cs, max_n_mode,
                                                                       max_iterations, tol_init)
        self._set_coefficients(modes, coefficients)
        self._summed_residuals = summed_residuals

    def check_inside(self, x=None, y=None, z=None):
        if x is None:
            xcs, ycs, zcs = self.x_cs, self.y_cs, self.z_cs
        else:
            xcs, ycs, zcs = coordinate_tools.scaled_projection(x - self.x0, y - self.y0, z - self.z0, self.scaling_factors,
                                                               self.principal_axes)

        azimuth, zenith, rcs = coordinate_tools.cartesian_to_spherical(xcs, ycs, zcs)
        r_cs_shell = reconstruct_shell(self.modes, self.coefficients, azimuth, zenith)
        return rcs < r_cs_shell

    def _visualize_shell(self, d_zenith=0.1, points=None):
        try:
            from mayavi import mlab
        except(ImportError):
            raise ImportError('Could not import mayavi.mlab.\
             Please make sure mayavi is installed to display fitted shell')

        if not points:
            x, y, z = self.x, self.y, self.z
        else:
            x, y, z = points
        zenith, azimuth = np.mgrid[0:(np.pi + d_zenith):d_zenith, 0:(2 * np.pi + d_zenith):d_zenith]

        xs, ys, zs = self.get_fitted_shell(azimuth, zenith)

        mlab.figure()
        mlab.mesh(xs, ys, zs)
        mlab.points3d(x, y, z, mode='point')

    def get_mesh_vertices_faces(self, d_zenith=0.1):
        """Compute vertices and faces to pass to 
        PYME.experimental._triangle_mesh.TriangleMesh.

        Note this will be non-manifold because of cuts at zenith=0, azimuth=0.

        Parameters
        ----------
        d_zenith : float, optional
            zenith step size for generating vector in plane of the shell [radians], by default 0.1
        """

        # Convert d_zenith to closest subdivision >= d_zenith (limited control due to icosahedon subdivision)
        # Mean initial d_zenith is 2*pi/5, so solve for (2*pi/10)^n_subdivision = d_zenith
        n_subdivision = int(np.ceil(np.log(d_zenith)/(np.log(np.pi)-np.log(5))))

        # Quasi-regular sample points on a unit sphere with icosahedron subdivision
        azimuth, zenith, faces = icosahedron_mesh(n_subdivision)
        # Map points onto shell radius 
        xs, ys, zs = self.get_fitted_shell(azimuth, zenith)

        # Create vertices 
        vertices = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()]).T   # Row-major ravel

        return vertices.astype(np.float32), faces.astype(np.int32)

    # def _visualize_scaled(self):
    #     from mayavi import mlab
    #     visualize_shell(self.modes, self.coefficients)#, scaling_factors=self.standard_deviations,
    #                     # scaling_axes=self.principal_axes)
    #     mlab.points3d(self.x_cs, self.y_cs, self.z_cs, mode='point')

    # def _visualize(self):
    #     from mayavi import mlab
    #     visualize_shell(self.modes, self.coefficients, scaling_factors=1./self.scaling_factors,
    #                     scaling_axes=self.principal_axes)
    #     mlab.points3d(self.x_c, self.y_c, self.z_c, mode='point')

    def distance_to_shell(self, query_points, d_angles=0.1):
        """

        Parameters
        ----------
        query_points : list-like of ndarrays
            Arrays of positions to query (in cartesian coordinates), i.e. [np.array(x), np.array(y), np.array(z)]
        d_angles : float
            Sets the step size in radians of zenith and azimuth arrays used in reconstructing the spherical harmonic shell

        Returns
        -------
        min_distance : float
            minimum distance from query points (i.e. input coordinate) to the spherical harmonic surface
        closest_points_on_surface : tuple of floats
            returns the position in cartesian coordinates of the point on the surface closest to the input 'position'

        """
        x, y, z = query_points
        x, y, z = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
        n_points = len(x)
        zenith, azimuth = np.mgrid[0:(np.pi + d_angles):d_angles, 0:(2 * np.pi + d_angles):d_angles]

        x_shell, y_shell, z_shell = self.get_fitted_shell(azimuth, zenith)
        # calculate the distance between all our points and the shell
        dist = np.sqrt(
            (x - x_shell[:, :, None]) ** 2 + (y - y_shell[:, :, None]) ** 2 + ((z - z_shell[:, :, None]) ** 2))

        # unfortunately cannot currently specify two axes for numpy.argmin, so we'll have to flatten the first two dims
        n_shell_coords = dist.shape[0] * dist.shape[1]
        dist_flat = dist.reshape((n_shell_coords, n_points))
        min_ind = np.argmin(dist_flat, axis=0)

        p_ind = range(n_points)
        return dist_flat[min_ind[p_ind], p_ind], (x_shell.reshape(n_shell_coords)[min_ind],
                                                  y_shell.reshape(n_shell_coords)[min_ind],
                                                  z_shell.reshape(n_shell_coords)[min_ind])
    
    def radial_distance_to_shell(self, points):
        """ finds distance to shell along a vector from the centre of the shell to a point.
        
        NOTE:
        -----
        
        This is not a geometric/euclidean distance, but should be a good approximation close to the shell (or far away).
        Over mid-range distances, it will give potentially very distorted distances. It is, however, very much faster and
        can safely be used in applications where strict scaling is not important - e.g. for a SDF representation of the
        surface.
        
        """
        # project points into scaled-space spherical coordinates
        azimuth_qs, zenith_qs, r_qs = self.shell_coordinates(points)

        # get scaled shell radius at those angles
        r_shell = reconstruct_shell(self.modes, self.coefficients, azimuth_qs, zenith_qs)

        # return the (scaled space) difference
        d = r_qs - r_shell
        
        #scale distance TODO - actually scale to make slightly closer to being euclidean
        return d
        

    def approximate_normal(self, x, y, z, d_azimuth=1e-6, d_zenith=1e-6, return_orthogonal_vectors=False):
        """

        Numerically approximate a vector(s) normal to the spherical harmonic shell at the query point(s).

        For input point(s), scale and convert to spherical coordinates, shift by +/- d_azimuth and d_zenith to get
        'phantom' points in the plane tangent to the spherical harmonic expansion on either side of the query point(s).
        Scale back, convert to cartesian, make vectors from the phantom points (which are by definition not parallel)
        and cross them to get a vector perpindicular to the plane.

        Returns
        -------

        Parameters
        ----------
        x : ndarray, float
            cartesian x location of point(s) on the surface to calculate the normal at
        y : ndarray, float
            cartesian y location of point(s) on the surface to calculate the normal at
        z : ndarray, float
            cartesian z location of point(s) on the surface to calculate the normal at
        d_azimuth : float
            azimuth step size for generating vector in plane of the shell [radians]
        d_zenith : float
            zenith step size for generating vector in plane of the shell [radians]

        Returns
        -------
        normal_vector : ndarray
            cartesian unit vector(s) normal to query point(s). size (len(x), 3)
        orth0 : ndarray
            cartesian unit vector(s) in the plane of the spherical harmonic shell at the query point(s), and
            perpendicular to normal_vector
        orth1 : ndarray
            cartesian unit vector(s) orthogonal to normal_vector and orth0

        """
        # scale the query points and convert them to spherical
        x_qs, y_qs, z_qs = coordinate_tools.scaled_projection(np.atleast_1d(x - self.x0), np.atleast_1d(y - self.y0),
                                                              np.atleast_1d(z - self.z0), self.scaling_factors,
                                                              self.principal_axes)
        azimuth, zenith, r = coordinate_tools.cartesian_to_spherical(x_qs, y_qs, z_qs)

        # get scaled shell radius at +/- points for azimuthal and zenith shifts
        azimuths = np.array([azimuth - d_azimuth, azimuth + d_azimuth, azimuth, azimuth])
        zeniths = np.array([zenith, zenith, zenith - d_zenith, zenith + d_zenith])
        r_scaled = reconstruct_shell(self.modes, self.coefficients, azimuths, zeniths)

        # convert shifted points to cartesian and scale back. shape = (4, #points)
        x_scaled, y_scaled, z_scaled = coordinate_tools.spherical_to_cartesian(azimuths, zeniths, r_scaled)
        # scale things "down" since they were scaled "up" in the fit
        scaled_axes = self.principal_axes / self.scaling_factors[:, None]
        coords = x_scaled.ravel()[:, None] * scaled_axes[0, :] + \
                 y_scaled.ravel()[:, None] * scaled_axes[1, :] + \
                 z_scaled.ravel()[:, None] * scaled_axes[2, :]
        x_p, y_p, z_p = coords.T
        # skip adding x0, y0, z0 back on, since we'll subtract it off in a second
        x_p, y_p, z_p = x_p.reshape(x_scaled.shape), y_p.reshape(y_scaled.shape), z_p.reshape(z_scaled.shape)

        # make two vectors in the plane centered at the query point
        v0 = np.array([x_p[1] - x_p[0], y_p[1] - y_p[0], z_p[1] - z_p[0]])
        v1 = np.array([x_p[3] - x_p[2], y_p[3] - y_p[2], z_p[3] - z_p[2]])
        if not np.any(v0) or not np.any(v1):
            raise RuntimeWarning('failed to generate two vectors in the plane - likely precision error in sph -> cart')
        # cross them to get a normal vector NOTE - direction could be negative of true normal
        normal = np.cross(v0, v1, axis=0)
        # return as unit vector(s) along each row
        normal = np.atleast_2d(normal / np.linalg.norm(normal, axis=0)).T
        # make sure normals point outwards, by dotting it with the vector to the point on the shell from the center
        points = np.stack([np.atleast_1d(x - self.x0), np.atleast_1d(y - self.y0), np.atleast_1d(z - self.z0)]).T
        outwards = np.array([np.dot(normal[ind], points[ind]) > 0 for ind in range(normal.shape[0])])
        normal[~outwards, :] *= -1
        if np.isnan(normal).any():
            raise RuntimeError('Failed to calculate normal vector')
        if return_orthogonal_vectors:
            orth0 = np.atleast_2d(v0 / np.linalg.norm(v0, axis=0)).T
            # v0 and v1 are both in a plane perpendicular to normal, but not strictly orthogonal to each other
            orth1 = np.cross(normal, orth0, axis=1)  # replace v1 with a unit vector orth. to both normal and v0
            return normal.squeeze(), orth0.squeeze(), orth1.squeeze()
        return normal.squeeze()

    def _distance_error(self, parameterized_distance, vector, starting_point):
        """

        Calculate the error in scaled space between the shell and the point reached traveling a specified distance(s)
        along the input vector from the input starting position.

        This function is to be minimized by a solver. Note that we don't actually have to calculate the distance in
        normal space, since minimizing in the scale space is equivalent.

        Parameters
        ----------
        parameterized_distance : float
            distance along vector to travel, units of nm, unscaled
        vector : ndarray
            Length three, vector in (unscaled) cartesian coordinates along which to travel
        starting_point : ndarray or list
            Length three, point in (unscaled) cartesian space from which to start traveling along 'vector'

        Returns
        -------

        """
        x, y, z = [parameterized_distance * np.atleast_2d(vector)[:, ind] + np.atleast_2d(starting_point)[:, ind] for
                   ind in range(3)]
        # scale the query points and convert them to spherical
        x_qs, y_qs, z_qs = coordinate_tools.scaled_projection(np.atleast_1d(x - self.x0), np.atleast_1d(y - self.y0),
                                                              np.atleast_1d(z - self.z0), self.scaling_factors,
                                                              self.principal_axes)
        azimuth_qs, zenith_qs, r_qs = coordinate_tools.cartesian_to_spherical(x_qs, y_qs, z_qs)

        # get scaled shell radius at those angles
        r_shell = reconstruct_shell(self.modes, self.coefficients, azimuth_qs, zenith_qs)

        # return the (scaled space) difference
        return r_qs - r_shell

    def distance_to_shell_along_vector_from_point(self, vector, starting_point, guess=None):
        """

        Calculate the distance to the shell along a given direction, from a given point.

        Parameters
        ----------
        vector : list-like
            cartesian vector indicating direction to query for proximity to shell
        starting_point : list-like
            cartesian position from which to start traveling along input vector when calculating shell proximity
        guess_distances : array, float
            initial guess for distance solver. See self._distance_error()

        Returns
        -------

        """

        if guess is None:
            guess = self._find_guess_for_distance_to_shell_along_vector_from_point(vector, starting_point)

        (res, cov_x, info_dict, mesg, res_code) = optimize.leastsq(self._distance_error, guess,
                                                                   args=(vector, starting_point),
                                                                   full_output=1)
        return res

    def _find_guess_for_distance_to_shell_along_vector_from_point(self, vector, starting_point, guess_distances=None):
        """

        Calculate the distance to the shell along a given direction, from a given point.

        Parameters
        ----------
        vector : list-like
            cartesian vector indicating direction to query for proximity to shell
        starting_point : list-like
            cartesian position from which to start traveling along input vector when calculating shell proximity
        guess_distances : array, float
            initial guess for distance solver. See self._distance_error()

        Returns
        -------

        """
        guess_distances = np.arange(0., 1000., 100.) if guess_distances is None else guess_distances
        errors = np.zeros_like(guess_distances)
        # guess = guess_distances[np.argmin(np.abs(self._distance_error(guess_distances, starting_point, vector)))]
        for ind, query in enumerate(guess_distances):
            errors[ind] = self._distance_error(query, vector, starting_point)

        return guess_distances[np.argmin(np.abs(errors))]
    
    def approximate_image_bounds(self, d_zenith=0.1, d_azimuth=0.1):
        from PYME.IO.image import ImageBounds
        
        zenith, azimuth = np.mgrid[0:(np.pi + d_zenith):d_zenith,
                                   0:(2 * np.pi + d_azimuth):d_azimuth]

        x_shell, y_shell, z_shell = self.get_fitted_shell(azimuth, zenith)

        return ImageBounds(x_shell.min(), y_shell.min(), 
                           x_shell.max(), y_shell.max(),
                           z_shell.min(), z_shell.max())

class SHShell(ScaledShell):
    '''
    Initial work on a replacement interface for ScaledShell
    
    Goals:
    
    - can be constructed directly as well as / instead of being fit
    - fitting is a single function call (rather than 3)
    - easily serialised (TODO)
    - does not own/keep a copy of data points (to facilitate serialisation)
    - when directly constructed, can used to represent synthetic nuclei etc ... whilst retaining distance
      evaluation functions
    
    '''
    def __init__(self, centre=(0,0,0), principle_axes=((1,0,0), (0,1,0), (0,0,1)), axis_scaling=(1.,1.,1.), modes=((0,0),), coefficients=(1,)):
        self.x0, self.y0, self.z0 = centre
        self.principal_axes = np.array(principle_axes)
        self.axis_scaling = np.array(axis_scaling)
        self.modes = modes
        self.coefficients = np.array(coefficients)
        
    def fit(self, points, n_max=3, n_iters=2, tol=0.3, principle_axis_sampling=1.0):
        x, y, z = points
        self.x0, self.y0, self.z0 = x.mean(), y.mean(), z.mean()
        x_c, y_c, z_c = x - self.x0, y - self.y0, z - self.z0
        
        sig, self.principal_axes = coordinate_tools.find_principle_axes(x_c, y_c, z_c, sample_fraction=principle_axis_sampling)
        self.axis_scaling = sig/np.max(sig)

        x_cs, y_cs, z_cs, = coordinate_tools.scaled_projection(x_c, y_c, z_c,self.axis_scaling, self.principal_axes)

        self.modes, self.coefficients, _ = sphere_expansion_clean(x_cs, y_cs, z_cs, n_max, n_iters, tol)
        
    @property
    def scaling_factors(self):
        import warnings
        warnings.warn(DeprecationWarning('use .axis_scaling instead'))
        return self.axis_scaling
        