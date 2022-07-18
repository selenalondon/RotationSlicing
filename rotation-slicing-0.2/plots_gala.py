import matplotlib.pyplot as plt
import numpy as np
from .projectionhelpers_gala import *
from .spinslice_gala import *
from .warn import _slice_warn
import math

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from matplotlib.projections import get_projection_class

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization.wcsaxes.frame import EllipticalFrame

import astropy.units as u

from mpl_toolkits.axisartist import angle_helper
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator, NullLocator, AutoLocator



class Annotatedimage(WCSAxes):

	#create a WCSAxes Object with a header
	def __init__(self, fig, header, rect=[0.1,0.1,0.9,0.9], wcs=None, transform=None, coord_meta=None,
				 transData=None, slices=None, frame_class=None, 
				 **kwargs):
				 
		self.fig=fig
		self.rect=rect
		self.header=header
		self.wcs=WCS(header)
		self.coord_meta=coord_meta
		self.frame_class=frame_class
		
		super().__init__(fig, rect, aspect=1, frame_class=None, **kwargs)
		h=self.header
		naxis1 = h['NAXIS1']
		naxis2 = h['NAXIS2']
		scale = min(self.bbox.width / naxis1, self.bbox.height / naxis2)
		h['NAXIS1'] = int(np.ceil(naxis1 * scale))
		h['NAXIS2'] = int(np.ceil(naxis2 * scale))
		scale1 = h['NAXIS1'] / naxis1
		scale2 = h['NAXIS2'] / naxis2
		h['CRPIX1'] = (h['CRPIX1'] - 1) * (h['NAXIS1'] - 1) / (naxis1 - 1) + 1
		h['CRPIX2'] = (h['CRPIX2'] - 1) * (h['NAXIS2'] - 1) / (naxis2 - 1) + 1
		h['CDELT1'] /= scale1
		h['CDELT2'] /= scale2
		self.reset_wcs(WCS(h))
		self.set_xlim(-0.5, h['NAXIS1'] - 0.5)
		self.set_ylim(-0.5, h['NAXIS2'] - 0.5)
		self._header = h
		self.frame_class=frame_class

class RotationSlicing:

    lonlat_names=("ra", "dec")
    
    def __init__(self, lon0lat0spin0, slice_dimens, vellim, data=None):
        self.lon0lat0spin0=lon0lat0spin0
        self.slice_dimens=slice_dimens
        self.vellim=vellim
        self.data=data

        self.lon,self.lat = self.slice_dimens
        halflon=self.lon/2 
        halflat=self.lat/2 
        self.lonlim = (-halflon, halflon)
        self.latlim = (-halflat, halflat)
        
    def graticule(self, fig, rect, grat_dimens, sliceboundary=False, tick_spacing=30, 
                  grid_main_c='black', grid_main_ls='solid', grid_main_lw=1.5, grid_main_a=0.15, 
                  grid_axes_c='black', grid_axes_ls='solid', grid_axes_lw=1.5, grid_axes_a=0.2, 
                  border_c='g', border_ls='solid', border_lw=1.5, border_a=0.7,
                  title_name='Default Name', title_c='k', title_font='Nimbus Roman', title_size=20,
                  **kwargs):
       
        if grat_dimens=='slice':
            grat_dimens = (self.lon+10, self.lat+10)
        else:
            grat_dimens=grat_dimens
            
        h = header(grat_dimens, lon0lat0spin0=self.lon0lat0spin0, coord_type='celestial')
        ax = Annotatedimage(fig=fig, header=h, rect=rect)
        
        # changing this to look more like the kapteyn graphs(ax.grid(color='black', ls='dotted',lw=1, alpha=0.7))
        # this controls the main graticule lines (the lighter ones that are everywhere).
        # can be modified using --> grid_main_c='color', grid_main_ls='style', grid_main_lw=linewidth, grid_main_a=alpha 
        ax.grid(color=grid_main_c, linestyle=grid_main_ls, linewidth=grid_main_lw, alpha= grid_main_a)
        # can modify the title using --> title_name='name', title_c='color', title_font='font', title_size=size
        ax.set_title(title_name, color=title_c, font=title_font).set_fontsize(title_size)
        fig.add_axes(ax)
        
        # sets the tick spacing of the lon, lat lines
        lon1 = ax.coords[0]
        lat1 = ax.coords[1]
        # can be modified using "tick_spacing" parameter --> tick_spacing=<int/float number of degrees between graticule lines>
        lon1.set_ticks(spacing=tick_spacing*u.deg)
        lat1.set_ticks(spacing=tick_spacing*u.deg)
        
        #elliptical frame	
        if grat_dimens== 'allsky' or grat_dimens=='hemisphere':
            #copy header
            h2 = fits.Header(h, copy=True)

            h2['CRVAL1'] = 0.0
            h2['CRVAL2'] = 90.0
            h2['LONPOLE'] = 0.0

            ax2 = Annotatedimage(fig=fig, header=h2, rect=rect)
            #make graticule. This controls the border of the graticule
            # can modify using --> border_c='color', border_ls='linestyle', border_lw=linewidth, border_a=alpha
            ax2.grid(color=border_c, linestyle=border_ls, linewidth=border_lw, alpha=border_a)

            lon2 = ax2.coords[0]
            lat2 = ax2.coords[1]
            lat2.set_ticks_visible(False)
            lat2.set_ticklabel_visible(False)
            lon2.set_ticks_visible(False)
            lon2.set_ticklabel_visible(False)

            #turn off dec
            lat2.grid(alpha=0)

            #generates a frame for allsky
            #tries to generate a frame for hemisphere
            if grat_dimens=='allsky':
                lon2.set_ticks(np.array([0,360]) * u.degree)
            else:
                lon2.set_ticks(np.array([90,270]) * u.degree)

            # Turns off background and overplots!
            ax2.set_frame_on(False)
            fig.add_axes(ax2)
            
            ax3 = Annotatedimage(fig=fig, header=h, rect=rect)
                                 
            lon3 = ax3.coords[0]
            lat3 = ax3.coords[1]

            #turn off dec
            lat3.set_ticks(np.array([0,360]) * u.degree)
            lat3.set_ticks_visible(False)
            lat3.set_ticklabel_visible(False)
            
            #only plot lines on edges
            
            # changed this from(np.array([0,180]) to what it is. It made the latitude axes dark line only be between -90 and 90
            lon3.set_ticks(np.array([0,360]) * u.degree)
            lon3.set_ticks_visible(False)
            lon3.set_ticklabel_visible(False)
            
            # this controls the the axes options where either the lon or lat is 0
            # can modify using --> grid_axes_c='color', grid_axes_ls='linestyle', grid_axes_lw=linewidth, grid_axes_a=alpha
            ax3.grid(color=grid_axes_c, linestyle=grid_axes_ls, linewidth=grid_axes_lw, alpha=grid_axes_a)
            ax3.tick_params()

            #Turns off background and overplots!
            ax3.set_frame_on(False)
            fig.add_axes(ax3)
      
    
        if sliceboundary==True:      
            lonslats = spun_slicebound(slice_dimens=self.slice_dimens, lon0lat0spin0=self.lon0lat0spin0)
            lons, lats = lonslats['ra'], lonslats['dec'] # had to add this because spun_slicebound now returns a table
            
            ax.plot(lons, lats, 'b-', transform=ax.get_transform('world'))
 
            #plot points if data supplied
        if self.data is not None:
            ax.scatter(self.data['ra'][:], self.data['dec'][:], transform=ax.get_transform('world'), **kwargs)

        return ax

    def insidelabels(self, ax, lon_lable_color='r', lon_lable_size=20, lat_lable_color='b', lat_lable_size=20):
			
			#method to add inside labels to existing annim ax
        ra = ax.coords[0]
        dec = ax.coords[1]
			#labels every 30 degrees automatically
        ra.set_ticks_visible(False)
        ra.set_ticklabel_visible(False)
			
			#only plot lines on edges
        dec.set_ticks_visible(False)
        dec.set_ticklabel_visible(False)

			#lonlabels
        ras = np.round(ra.locator(0,359)[0].value)
        len_ras = np.arange(0,len(ras),1)

			#latlabels, doesn't include the 90 degree points
        decs1 = dec.locator(-89,-1)[0].value
        len_decs1 = np.arange(0,len(decs1),1)
        
        # added in a second decs part so that the "0 lon, 0 lat" symbols do not over lap and look bad
        decs2 = dec.locator(1,89)[0].value
        len_decs2 = np.arange(0,len(decs2),1)
        
        # defining the degree symbol, had to change the value of the degree number to a string before adding on the degree sign
        a =  (u'\u00B0')
        
        # the 'ha' = 'horizontalalignment". It dictates where the label sits on the axis --> left, center, or right hand sid eof the label.
        # va does the same thing but in the vertical direction
        # can modify color and size of numbers by --> lon_lable_color='color', lon_lable_size=size, (same for lats)
        for i in len_ras:
            ax.text(ras[i], 0, str(int(ras[i])) + a, color=lon_lable_color, size=lon_lable_size, 
                    transform=ax.get_transform('world'), clip_on=True, 
                    font='Nimbus Roman', ha='center', va='center', rotation=0)
#             print(xy(ras[i] + 0.01, 0))
#             print(ras[i] - 0.01, 0)
#             print(ax.coords[0], ax.coords[1])
            
        for i in len_decs1:
            ax.text(0, decs1[i], str(int(decs1[i])) + a, color=lat_lable_color, size=lat_lable_size, 
                    transform=ax.get_transform('world'), clip_on=True, 
                    font='Nimbus Roman', ha='center', va='center')
            
        for i in len_decs2:
            ax.text(0, decs2[i], str(int(decs2[i])) + a, color=lat_lable_color, size=lat_lable_size, 
                    transform=ax.get_transform('world'), clip_on=True, 
                    font='Nimbus Roman', ha='center', va='center')
            
        # taking this out made only tick labels appear in the correct places
#         for i in len_decs:
#             ax.text(180, decs[i], decs[i], color='k', transform=ax.get_transform('world'), clip_on=True) # the constant for the longitude line before was 160. I changed it to 180 to follow the correct line



    def slicer(self, data, columnnames=('ra', 'dec', 'z'), show=True, antispin=False):

        """ 
		Takes table with columns ra, dec, and z, adds new velocity column (in km/s)
		returns data points within a slice of dimensions lonlim and latlim in 
		newlon, newlat coordinates centered around lon0 lat0 and spun clockwise by spin0.
		Also returns newlon and newlat in radians for plotting. 

		parameters
		----------
		data		: table/structured array with ra, dec, z and any other data (i.e. color, type)
		lon0lat0spin0 : central longitude and latitude of slice and clockwise spin 
		lonlim		: slice width in degrees
		latlim		: slice height in degrees
		columnnames : the names of the columns for ra, dec, and z in the datatable 
						as a 3-tuple of strings
						default is ('ra', 'dec', 'z')
		show		: whether or not to print the table, default True l
        """
		
        # raname, decname, zname = columnnames
        
        
      # going to do this in the table_rot_slicer_gala function
# 		#convert redshift to velocity and rename 
#         x = 3*10**5
#         data['vel']=data[zname][:]*x
      

		#slice it as you please
        if antispin==True:
            sliced=table_rot_slicer_gala(data, lon0lat0spin0=self.lon0lat0spin0, 
                                         lonlim=self.lonlim, latlim=self.latlim, vellim=self.vellim, invert=True)
        else:
            sliced=table_rot_slicer_gala(data, lon0lat0spin0=self.lon0lat0spin0, 
                                         lonlim=self.lonlim, latlim=self.latlim, vellim=self.vellim)

        if show==True:
            print(sliced)

        return sliced

	
    def pieframe(self, fig, rect, thetagrid=False, rgrid=False, hideoutsidedata=False, 
                 thetalabel=False, rlabel=True, radeclabels=True, labelrotation=None, **kwargs):

        """
		returns a pie-shaped scatter plot of galaxies in the slice, 
		longitude on theta axis, velocity on r axis
		if data=None, just gives an empty frame 

		parameters
		----------
		fig			: matplotlib figure instance 
		ax			: polar subplot within that figure 
		thetagrid		: whether or not to draw gridlines along theta axis, default false
		rgrid			: whether or not to draw gridlines along radial axis, default false 
		hideoutsidedata : whether or not to hide points outside of axes, default is false 
							(so points outside of vellim will still be plotted) 
		thetalabel		: whether or not theta label is displayed 
		radiallabel		: whether or not radial label is displayed
		radeclabels		: whether or not ra dec labels to show corresponding position in
							original coordinate system are displayed
        """

        fig=fig
        ax=fig.add_axes(rect=rect, projection='polar')

		#get tmin and tmax
        tmin, tmax = self.lonlim

		#get rmin and rmax
        rmin, rmax = self.vellim

        ax.set_rlim(rmin, rmax)

        ax.set_thetalim(np.deg2rad(tmin), np.deg2rad(tmax))
		
        ax.grid(False)
		
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
		
        if tmax>=20:
            ax.xaxis.set_major_locator(MultipleLocator(np.deg2rad(10)))
        if tmax>10:
            ax.xaxis.set_major_locator(MultipleLocator(np.deg2rad(5)))
        if tmax<10:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_minor_locator(NullLocator())
		
        if thetagrid==True:
            ax.xaxis.grid(True)
        if rgrid==True:
            ax.yaxis.grid(True)

        ax.tick_params(direction='in', length=6, top=True, bottom=True, left=True, right=True)
		#Let's add minor ticks ! 
        ax.yaxis.set_minor_locator(AutoMinorLocator())	 
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction='in', which='minor', length=4, top=True, bottom=True, left=True, right=True)


		#make annular if vel min is greater than zero 
        if rmin>0:
            ax.set_rorigin(-0.5)

		#set zero to center of theta axis
        ax.set_theta_offset(0.5*np.pi)

		#Labels!

		#this sets velocity label along r axis between rmin and rmax
		#first input gives horizontal position (theta)
		#second gives vertical (r)
		
        if rlabel==True:
            if tmax>10:
                ax.text(np.deg2rad(tmin-tmax/3),(rmin+rmax)/2.,'Velocity (km/s)',
                    rotation=90-tmax, ha='left', va='center')
            if tmax<10:
                ax.text(np.deg2rad(tmin-tmax/2),(rmin+rmax)/2.,'Velocity (km/s)',
                    rotation=90-tmax, ha='left', va='center')

		#label "x axis"
        if rmax>10000:
            tlabel=rmax+rmax/4
        else:
            tlabel=rmax+rmax/3
        if tmax>40:
            tlabel=rmax+rmax/2
		
        if thetalabel==True:
            ax.text(0, tlabel,'Longitude in New Coordinate System', ha='center', va='top')

		#make ra/dec in original coords labels
        tstep=10
        theta_locator=angle_helper.LocatorDMS(tmax-tmin/tstep)
        levs,n,factor=theta_locator(tmin,tmax)

		#get rid of locations not in range
        ticklocs=[]
        for t in ax.xaxis.get_majorticklocs():
            if t >= np.deg2rad(tmin) and t <=np.deg2rad(tmax):
                ticklocs.append(t)

		#set lats to zero since we sliced along equator
        newlats=np.array(np.zeros(len(ticklocs)))

		#hypothetical spin that gave us this data, going back to original coords
        # lons1, lats1=spin(np.rad2deg(ticklocs), newlats, self.lon0lat0spin0, invert=True)
        
        lonslats = Table()
        lonslats['phi1'] = np.rad2deg(ticklocs)     # had to put into table and call their keys the correct names for the rot_gala to work
        lonslats['phi2'] = newlats

        lons1lats1 = rot_gala(lonslats, self.lon0lat0spin0, invert=True)
        lons1, lats1 = lons1lats1['ra'], lons1lats1['dec']

		#make all lons positive
        lons1=np.array([lon if (lon >= 0.) else (lon+360.) for lon in lons1])
        
		#round to integer values
        lons=np.round(lons1)
        lats=np.round(lats1)

		#convert to hours or degrees
        DMSer=angle_helper.FormatterDMS()
        HMSer=angle_helper.FormatterHMS()
		#if past a certain scale, change both to hms ? 
        lon=HMSer("center",factor,lons)
        lat=DMSer("center",factor,lats)

		#the \n gives multiline tick labels
		#combine with the \n so it will be in two lines
        combined = zip(lon,lat)
        labels=[(str(i)+"\n"+str(j)) for i,j in combined]
        labels1=np.array(labels)
        
		#Now I plot them as text objects 
        rlim=rmax-rmin
        rspan=rlim/rmax
		
        if rspan<=0.5:
            r=rmax+rmax/15
        else:
            r=rmax+rmax/10

        if rmax>20000:
            r=rmax+500

        iteration7=np.arange(0,len(labels1),1)

        if radeclabels==True:
            if labelrotation is None:
                for i in iteration7:
                    ax.text(ticklocs[i], r, labels1[i], rotation=np.rad2deg(ticklocs[i]), va='bottom', ha='center', ma='center')
            else:
                for i in iteration7:
                    ax.text(ticklocs[i], r, labels1[i], rotation=labelrotation, va='bottom', ha='center', ma='center')

		#plot points if data supplied
        if self.data is not None:
            mydata=self.slicer(data=self.data, show=False)
            ax.scatter(np.deg2rad(mydata['phi1'][:]), mydata['vel'][:], **kwargs)
            
        return ax