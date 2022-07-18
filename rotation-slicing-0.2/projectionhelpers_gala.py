#!/usr/bin/env python
# coding: utf-8

# the usual imports
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import math

# imported for projection plot & graticule
#from kapteyn import maputils

#local
#from .spinslice import spin, table_spin_slice
from .spinslice_gala import *

from astropy.table import Table
import astropy.units as u

__all__ = ['lonlatspin_NP', 'pix_dimens_cdelta_crpix', 'header', 'spun_slicebound']

# ## **********
# ## Define Aitoff Plot functions 
# ## **********

# --------------
# graticule creation helper functions
# --------------

def lonlatspin_NP(lon0lat0spin0):
	""" Helper function for equator_aitoff_header().  
	Returns list [lon_NP, lat_NP, spin_NP] that corresponds to lon0lat0spin0 tuple, 
	i.e., parameters of polar spin equivalent to given equatorial spin.	 
	
	parameters
	----------
	lon0lat0spin0  : float 3-tuple, list, or array
					 (lon0, lat0, spin0)
					 longitude, latitude of new origin & right-hand spin about that origin

	"""
	try: 
		lon0,lat0,spin0 = lon0lat0spin0
	except TypeError:
			print("lon0lat0spin0 must be a 3-tuple")
			
	# define temp variables for Cos and Sin of angles
	_Clon0,_Slon0 = np.cos(np.radians(lon0)),np.sin(np.radians(lon0))
	_Clat0,_Slat0 = np.cos(np.radians(lat0)),np.sin(np.radians(lat0))
	_Cspin0,_Sspin0 = np.cos(np.radians(spin0)),np.sin(np.radians(spin0))
	
	# 1. find pole longitude and spin about pole
	#
	# 1.1 If ref pt is on equator:
	#	  find spin, and direction sin, cos for longitude
	if lat0 == 0:
		if _Sspin0 < 0:
			_plusminus = -1
			_spinNP = -np.pi/2
		else:
			_plusminus = +1
			_spinNP = +np.pi/2
			
		_sinlonNP = -_plusminus*_Clon0
		_coslonNP = +_plusminus*_Slon0

	# 1.2 if ref pt not on equator: 
	#	  find direction sin, cos for longitude and spin, 
	#	  use arctan2 to get spin with correct quadrant.   
	else:	   
		_sinlonNP = -_Cspin0*_Slat0*_Slon0 - _Clon0*_Sspin0
		_coslonNP = -_Cspin0*_Slat0*_Clon0 + _Slon0*_Sspin0
		_sinlatNP = +_Clat0*_Sspin0
		_coslatNP = -_Slat0

		# Note numpy arctan2(x1,x2) is element-wise arc tangent of "x1/x2", 
		# while Mathematica's ArcTan[x1,x2] is arc tangent of "x2/x1" (reversed).
		_spinNP = np.arctan2(_sinlatNP,_coslatNP)
	
	# 1.3 find longitude
	_lonNP = np.arctan2(_sinlonNP,_coslonNP)
	
	# 2. find pole latitude
	_latNP =  np.arcsin( _Clat0*_Cspin0 ) 
	
	    
	
	return [np.around(np.degrees(val), decimals=5) for val in [_lonNP,_latNP,_spinNP]]


def pix_dimens_cdelta_crpix(grat_dimens, max_pixelnum=100, pad=None):
	"""Helper function for equator_aitoff_header().
	Returns tuple (pixel_len, pixel_wid, cdelta, crpix1, crpix2) for given
	graticule dimension.	 
	(pixel_len, pixel_wid) are the dimensions
	of the effective pixel array, cdelta is the angular scale in  
	degrees per pixel, and (crpix1, crpix2) are the pixel positions of the center.
	See Kapteyn docs.
	
	parameters
	----------
	grat_dimens	   : float 2-tuple	(grat_len, grat_wid) 
					 grat_len is horizontal, grat_wid is vertical, in degrees 
	max_pixelnum   : integer
					 number of pixels on the longer side
	pad			   : integer
					 width in pixels padding each edge of original array
	
	"""
	
	_grat_len, _grat_wid = grat_dimens
	
	# find pixel scale cdelta in degrees per pixel
	if _grat_len >= _grat_wid:
		_cdelta = _grat_len/float(max_pixelnum)
	else:
		_cdelta = _grat_wid/float(max_pixelnum)
		
	_pixel_len = int(round(_grat_len/_cdelta))
	_pixel_wid = int(round(_grat_wid/_cdelta))
		
	# find center of slice in pixels
	_crpix1, _crpix2 = [val/2+1 for val in (_pixel_len,_pixel_wid)]
	if pad != None:
		try:
			pad = int(pad)
		except SyntaxError:
			print("pad must be None or number of pixels padding your graticule")
		# pad the image area to be larger than the slice area
		(_pixel_len, _pixel_wid) = (_pixel_len + 2*pad, _pixel_wid + 2*pad)
		(_crpix1, _crpix2) = (_crpix1 + pad, _crpix2 + pad)
	
	return (_pixel_len,_pixel_wid,_cdelta,_crpix1,_crpix2)	 

# shortcuts for coordinate types w/ Aitoff projections recognized by 
# Kapteyn's graticule generator

coord_types = {'celestial':('RA---AIT','DEC--AIT'),
			   'galactic':('GLON-AIT','GLAT-AIT'),
			   'supergalactic':('SLON-AIT','SLAT-AIT'),
			   'ecliptic':('ELON-AIT','ELAT-AIT'),
			   'helioecliptic':('HLON-AIT','HLAT-AIT')}


def header(grat_dimens, 
	lon0lat0spin0=(0.,0.,0.), coord_type='celestial', 
	max_pixelnum=200, pad=None):
	"""
	generates FITS header for Aitoff projection centered and oriented as specified 
	by the given tuple of equatorial parameters lon0lat0spin0.	To be passed to 
	Kapteyn's oblique Aitoff projection graticule generator. 
	
	parameters
	----------
	grat_dimens		:  float 2-tuple  (grat_len, grat_wid) 
					   grat_len is horizontal, grat_wid is vertical, in degrees 
	lon0lat0spin0	:  float or integer 3-tuple, list, or array (lon0, lat0, spin0)
					   lon0, lat0 is position of new lon-lat origin, in degrees
					   spin0 is RHR spin of equator about new origin, in degrees
	coord_type		:  string 
					   Key to FITS code specifying coordinate type and projection.
					   choices are 'celestial', 'galactic', 'supergalactic', 'ecliptic', 
					   'helioecliptic'; projection is Aitoff in each case.
					   Keys and codes are in dictionary coord_types.
	max_pixelnum	:  integer
					   effective number of pixels along longest edge
	pad				:  integer
					   width of pixel padding each edge of original array 
										   
	"""
	
	if grat_dimens=='allsky':
		grat_dimens = (360,180)
	if grat_dimens=='hemisphere':
		grat_dimens = (180,180)
	
	# get position and spin about new pole
	lon_pole, lat_pole, spin_pole = lonlatspin_NP(lon0lat0spin0)
	
	# get graticule pixel dimensions: len is horizontal, wid is vertical
	(_pixel_len, _pixel_wid,_cdelta,_crpix1,_crpix2) = pix_dimens_cdelta_crpix(
		grat_dimens,max_pixelnum=max_pixelnum,pad=pad)
	
	# set coordinate type
	assert coord_type in coord_types, """coord_type is {0}, 
		defined types are {1}""".format(coord_type,coord_types.keys())
	_ctype1,_ctype2 = coord_types[coord_type]
	
	_header = {'NAXIS' : 2,
		 'NAXIS1':	_pixel_len ,
		'NAXIS2':	_pixel_wid ,
		'CDELT1':  -_cdelta ,
		'CRPIX1':	_crpix1,
		'CRVAL1':	lon_pole,    
		'CTYPE1':  _ctype1,
		'CUNIT1':  'deg' ,
		'CDELT2':	_cdelta,
		'CRPIX2':	_crpix2,
		'CRVAL2':	lat_pole, 
		'CTYPE2':  _ctype2,
		'CUNIT2':  'deg' ,
		# must add 180 in LONPOLE so newlon = 0 is at center, not edge
		'LONPOLE': -spin_pole + 180,
		#IMPORTANT: reference point PV1_1, PV1_2 must be pole, do not change this; 
		'PV1_1': 0.0, 'PV1_2': 90.0 
		} 
	return _header

# -------------
# slice boundary creation functions
# --------------

# define get_minmax
# Helper function: converts a single number to a symmetric pair -/+ if two numbers not given.

def get_minmax(minmax):
	""" 
	Helper function for slice_bound().
	converts a single number to a symmetric pair -/+ if two numbers not given.
 
	parameters
	----------
	minmax	 : integer or float; or integer or float tuple or list 
	"""
	if (type(minmax) is tuple or type(minmax) is list):
		if len(minmax)==1:
			return (-minmax[0],minmax[0])
		else:
			return (minmax[0],minmax[1])
	else:
	  return (-minmax,+minmax)	
	
# define slice_bound
# Helper function: creates list of points outlining the slice, in the new coordinate system 
# (i.e., aligned with the new equator, along the new latitude and longitude lines).
# To ensure the left/right sides of the slice boundary have the same number density of points 
# as the top/bottom sides, have a default number of points for the short side, then do X times 
# that default for the longer span, where X = long/short.
def slice_bound(newlonminmax,newlatminmax, numpts=10):
	(lonmin,lonmax) = get_minmax(newlonminmax)
	(latmin,latmax) = get_minmax(newlatminmax)

	(deltalon,deltalat) = (lonmax-lonmin,latmax-latmin)
	# adjust point density so lon and lat edges of border are same density
	if deltalat < deltalon:
		numlatpts = numpts
		numlonpts = round(numpts*float(deltalon)/deltalat)
	else:
		numlonpts = numpts
		numlatpts = round(numpts*float(deltalat)/deltalon)

	latvals = np.linspace(latmin,latmax,numlatpts)
	lonvals = np.linspace(lonmin,lonmax,numlonpts)

	latmaxbound = [[lonval,latmax] for lonval in lonvals]
	latminbound = [[lonval,latmin] for lonval in lonvals] 
	lonmaxbound = [[lonmax,latval] for latval in latvals]
	lonminbound = [[lonmin,latval] for latval in latvals]
	# reverse point order in two bounds so points can form a continuous line
	boundpts = np.array(latmaxbound + lonmaxbound[::-1] + latminbound[::-1] + lonminbound)
	return boundpts.T

# define spun_slicebound & add_sliceboundary
# Finds coordinates of points outlining slice boundary in original coordinate system, 
# i.e., in RA & Dec or sgl &sgb etc. Slice is not aligned along equator in original coordinate 
# but is centered on the give longitude and latitude [e.g., (RA, Dec) = (lon0, lat0)] 
# and rotated clockwise (as seen by observer inside celestial sphere) by angle spin0 with 
# respect to line of latitude passing through that point.
# Since slice is not aligned with equator in this coordinate system, call list of these 
# coordinates "spun_slicebound" (not "slicebound").

def spun_slicebound(slice_dimens, lon0lat0spin0=(0.,0.,0.),numpts=10):
    (slice_len, slice_wid) = slice_dimens
    # create lon, lats of pts on slice boundary centered on new origin, 
    # aligned with new equator
    _boundlon,_boundlat=slice_bound(slice_len/2.,slice_wid/2.,numpts)
    # find lon, lats of those pts in original coordinate system
    
    # my input in order for my rot_gala function to get a table instead of an array
    _boundlonlat = Table()
    _boundlonlat['phi1'] = _boundlon
    _boundlonlat['phi2'] = _boundlat

   
    # changing '_boundlon,_boundlat'. The input is now '_boundlonlat' becuase my rot_gala function takes a table 
    return rot_gala(_boundlonlat, lon0lat0spin0, invert=True) # returns array ([lons, lats]) # update- it now returns astro table