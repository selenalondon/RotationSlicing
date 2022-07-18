#!/usr/bin/env python
# coding: utf-8

from textwrap import dedent
from warnings import warn

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates.transformations import (DynamicMatrixTransform,
                                                 FunctionTransform)
from astropy.coordinates.attributes import (CoordinateAttribute,
                                            QuantityAttribute)
from astropy.coordinates.matrix_utilities import (rotation_matrix,
                                                  matrix_product,
                                                  matrix_transpose)
from astropy.coordinates.baseframe import base_doc
from astropy.utils.decorators import format_doc
import numpy as np

import astropy.coordinates.angle_utilities as ang
from astropy.coordinates import SkyCoord

__all__ = ['GreatCircleICRSFrame', 'pole_from_endpoints', 'rot_gala', 'table_rot_slicer_gala']



def greatcircle_to_greatcircle(from_greatcircle_coord,
                               to_greatcircle_frame):
    """Transform between two greatcircle frames."""

    # This transform goes through the parent frames on each side.
    # from_frame -> from_frame.origin -> to_frame.origin -> to_frame
    intermediate_from = from_greatcircle_coord.transform_to(
        from_greatcircle_coord.pole)
    intermediate_to = intermediate_from.transform_to(
        to_greatcircle_frame.pole)
    return intermediate_to.transform_to(to_greatcircle_frame)


def reference_to_greatcircle(reference_frame, greatcircle_frame):
    """Convert a reference coordinate to a great circle frame."""

    # Define rotation matrices along the position angle vector, and
    # relative to the origin.
    pole = greatcircle_frame.pole.transform_to(coord.ICRS())
    ra0 = greatcircle_frame.ra0
    R_rot = rotation_matrix(0, 'z')

    if not np.isnan(ra0) and np.abs(pole.dec.value) > 1e-15:
        zaxis = pole.cartesian.xyz.value

        xaxis = np.array([np.cos(ra0), np.sin(ra0), 0.])

        if np.abs(zaxis[2]) >= 1e-15:
            xaxis[2] = -(zaxis[0]*xaxis[0] + zaxis[1]*xaxis[1]) / zaxis[2]
        else:
            xaxis[2] = 0.
        xaxis = xaxis / np.sqrt(np.sum(xaxis**2))

        yaxis = np.cross(zaxis, xaxis)
        yaxis = yaxis / np.sqrt(np.sum(yaxis**2))

        R = np.stack((xaxis, yaxis, zaxis))

    else:
        if not np.isnan(ra0) and np.abs(pole.dec.value) < 1e-15:
            warn("Ignoring input ra0 because the pole is along dec=0",
                 RuntimeWarning)

        R1 = rotation_matrix(pole.ra, 'z')
        R2 = rotation_matrix(90*u.deg - pole.dec, 'y')
        R = matrix_product(R2, R1)

    return matrix_product(R_rot, R)



def greatcircle_to_reference(greatcircle_coord, reference_frame):
    """Convert an great circle frame coordinate to the reference frame"""

    # use the forward transform, but just invert it
    R = reference_to_greatcircle(reference_frame, greatcircle_coord)
    # transpose is the inverse because R is a rotation matrix
    return matrix_transpose(R)



def greatcircle_transforms(self_transform=False):
    def set_greatcircle_transforms(cls):
        DynamicMatrixTransform(reference_to_greatcircle,
                               coord.ICRS, cls,
                               register_graph=coord.frame_transform_graph)

        DynamicMatrixTransform(greatcircle_to_reference,
                               cls, coord.ICRS,
                               register_graph=coord.frame_transform_graph)

        if self_transform:
            FunctionTransform(greatcircle_to_greatcircle,
                              cls, cls,
                              register_graph=coord.frame_transform_graph)
            
        return cls

    return set_greatcircle_transforms



_components = """
phi1 : `~astropy.units.Quantity`
    Longitude component.
phi2 : `~astropy.units.Quantity`
    Latitude component.
distance : `~astropy.units.Quantity`
    Distance.

pm_phi1_cosphi2 : `~astropy.units.Quantity`
    Proper motion in longitude.
pm_phi2 : `~astropy.units.Quantity`
    Proper motion in latitude.
radial_velocity : `~astropy.units.Quantity`
    Line-of-sight or radial velocity.
"""

_footer = """
Frame attributes
----------------
pole : `~astropy.coordinates.SkyCoord`, `~astropy.coordinates.ICRS`
    The coordinate specifying the pole of this frame.
ra0 : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
    If specified, an additional transformation will be applied to make
    this right ascension the longitude zero-point of the resulting
    coordinate frame.
rotation : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
    If specified, a final rotation about the pole (i.e. the resulting z
    axis) applied.
"""



@format_doc(dedent(base_doc), components=_components, footer=_footer)
@greatcircle_transforms(self_transform=True)
class GreatCircleICRSFrame(coord.BaseCoordinateFrame):
    """A frame rotated into great circle coordinates with the pole and longitude
    specified as frame attributes.

    ``GreatCircleICRSFrame``s always have component names for spherical
    coordinates of ``phi1``/``phi2``.
    """

    pole = CoordinateAttribute(default=None, frame=coord.ICRS)
    center = CoordinateAttribute(default=None, frame=coord.ICRS)
    ra0 = QuantityAttribute(default=np.nan*u.deg, unit=u.deg)
    
    
    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'phi1'),
            coord.RepresentationMapping('lat', 'phi2'),
            coord.RepresentationMapping('distance', 'distance')]
    }

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    _default_wrap_angle = 180*u.deg

    def __init__(self, *args, **kwargs):
        wrap = kwargs.pop('wrap_longitude', True)
        super().__init__(*args, **kwargs)
        if wrap and isinstance(self._data, (coord.UnitSphericalRepresentation,
                                            coord.SphericalRepresentation)):
            self._data.lon.wrap_angle = self._default_wrap_angle


#     TODO: remove this. This is a hack required as of astropy v3.1 in order
#     to have the longitude components wrap at the desired angle
    def represent_as(self, base, s='base', in_frame_units=False):
        r = super().represent_as(base, s=s, in_frame_units=in_frame_units)
        if hasattr(r, "lon"):
            r.lon.wrap_angle = self._default_wrap_angle
        return r

    represent_as.__doc__ = coord.BaseCoordinateFrame.represent_as.__doc__

    @classmethod
    def from_endpoints(cls, coord1, coord2, ra0=None, rotation=None):
        """Compute the great circle frame from two endpoints of an arc on the
        unit sphere.

        Parameters
        ----------
        coord1 : `~astropy.coordinates.SkyCoord`
            One endpoint of the great circle arc.
        coord2 : `~astropy.coordinates.SkyCoord`
            The other endpoint of the great circle arc.
        ra0 : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
            If specified, an additional transformation will be applied to make
            this right ascension the longitude zero-point of the resulting
            coordinate frame.
        rotation : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
            If specified, a final rotation about the pole (i.e. the resulting z
            axis) applied.
        """

        pole = pole_from_endpoints(coord1, coord2)

        kw = dict(pole=pole)
        kw['ra0'] = ra0

        return cls(**kw)


    
def pole_from_endpoints(coord1, coord2):
    """Compute the pole from a great circle that connects the two specified
    coordinates.

    This assumes a right-handed rule from coord1 to coord2: the pole is the
    north pole under that assumption.

    Parameters
    ----------
    coord1 : `~astropy.coordinates.SkyCoord`
        Coordinate of one point on a great circle.
    coord2 : `~astropy.coordinates.SkyCoord`
        Coordinate of the other point on a great circle.

    Returns
    -------
    pole : `~astropy.coordinates.SkyCoord`
        The coordinates of the pole.
    """
    cart1 = coord1.cartesian.without_differentials()
    cart2 = coord2.cartesian.without_differentials()
    if isinstance(coord1, coord.SkyCoord):
        frame1 = coord1.frame
    elif isinstance(coord1, coord.BaseCoordinateFrame):
        frame1 = coord1
    else:
        raise TypeError('Input coordinate must be a SkyCoord or coordinate frame instance.')

    c1 = cart1 / cart1.norm()

    coord2 = coord2.transform_to(frame1)
    c2 = cart2 / cart2.norm()

    pole = c1.cross(c2)
    pole = pole / pole.norm()
    return frame1.realize_frame(pole)




#######--------------------------------------------------------------##########
## These are the functions that I added in order to perform the slicing part ##
#######--------------------------------------------------------------##########

# The following will take in the data, the specific longitude and latitude zero points, the position angle (the
# the angle going counterclockwise starting at the vertical axis when looking from inside the celestial sphere), 
# and the longitude, latitude, and velocity limits of the slice you want to take.


# this function will only perform the rotations

def rot_gala(data, lon0lat0spin0, invert=False):
    lon0, lat0, spin0 = lon0lat0spin0
    lon0 = lon0*u.deg
    lat0 = lat0*u.deg
    spin0 = spin0*u.deg
    theta = 90*u.deg - spin0
    
    # finds second point that you need in order to get the pole and puts them into SkyCoord object
    distance = 10*u.deg    # arbitrary value
    lon1, lat1 = ang.offset_by(lon0, lat0, theta, distance)
    points = coord.SkyCoord(ra=[lon0, lon1], dec=[lat0, lat1])
    # finding pole and using it to find our_fr
    pole = pole_from_endpoints(points[0], points[1])
    our_fr = GreatCircleICRSFrame(pole=pole, ra0=lon0)
        
    # invert will do the opposite rotation. (ie transforming your data back to the icrs frame if you have 
    # data in your frame already)
    if invert == True:
     
        # defining your coords in our_fr already
        coords_sc = SkyCoord(phi1=data['phi1'], 
                             phi2=data['phi2'],
                             frame=our_fr, 
                             unit=u.deg
                            )
        
        # transforming coords back to ICRS frame
        transformed_coords = coords_sc.transform_to('icrs')
        
        # adding the newly transformed back coords to the original astropy table
        data['ra'] = transformed_coords.ra
        data['dec'] = transformed_coords.dec
    
    # Follows same steps as above. Only difference is that the coords are defined in ICRS frame and we are 
    # going from ra --> phi1 and dec --> phi2
    else:    
    
        # converting coords from the astropy data table into a SkyCoord object (in ICRS frame by default)
        coords_sc = SkyCoord(ra=data['ra' or 'lon'], 
                             dec=data['dec' or 'lat'], 
                             unit=u.deg
                            )
    
        # transforming the SkyCoord coordinates to the new frame
        transformed_coords = coords_sc.transform_to(our_fr)
        
        data['phi1'] = transformed_coords.phi1
        data['phi2'] = transformed_coords.phi2
        
        # phi1 = transformed_coords.phi1
        # phi2 = transformed_coords.phi2
        
    return data


def table_rot_slicer_gala(data, lon0lat0spin0, lonlim=None, latlim=None, vellim=None, invert=False):
    lon0, lat0, spin0 = lon0lat0spin0
    
    if invert == True:
        # transforming data back to ICRS
        new_data = rot_gala(data, lon0lat0spin0, invert=True)
        
        ###
        ### Do There need to be a slicing component here? there is none on the spinslice version ###
        ###
    
    else: 
        
        # transforming data
        new_data = rot_gala(data, lon0lat0spin0, invert=False)
        
        if 'z' in data.keys():
            # convert the red shift (z) to a velocity (vel)
            x = 3*10**5
            new_data['vel'] = new_data['z'][:]*x
    
        # masking the data to only get the values we want in the 'slice'
        if lonlim != None:
            new_data = new_data[(new_data['phi1'] > lonlim[0]) & (new_data['phi1'] < lonlim[1])]
        
        if latlim != None:
            new_data = new_data[(new_data['phi2'] > latlim[0]) & (new_data['phi2'] < latlim[1])]
        
        if vellim != None:
            new_data = new_data[(new_data['vel'] > vellim[0]) & (new_data['vel'] < vellim[1])]
    
    return new_data