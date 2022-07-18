def globeframe(self):
		
		"""
		method to create a frame with a globe with slice boundary drawn in and 
		a projection plot zoomed in on the slice, with corresponding pie plot to the right 
		
		returns ax (empty globe), ax_inset (empty zoomed projection plot) and pieframe (empty pieplot)
		
		parameters
		----------
		figsize		: figure size in inches, default is (10,10) 
		show		: whether or not to show the frame, default True 
		"""
		
		lonmin,lonmax=self.lonlim
		latmin,latmax=self.latlim
		lon0,lat0,spin0=self.lon0lat0spin0
		
		fig = plt.figure(dpi=100, figsize=(10,10))

		center=str(lon0)+"d"+" "+str(lat0)+"d"

		ax = plt.axes(
			[0.0, 0.0, 0.4, 0.4],
			projection='astro globe', center=center)

		ax.grid()
		
		#radius doesn't seem to actually be in degrees, how can i make it automatically a good sized selection??
		ax_inset = plt.axes(
			[0.3, 0.3, 0.2, 0.2],
			projection='astro zoom', center=center, radius=lonmax*u.degree)

		ax_inset.grid()

		ax_inset.coords['ra'].set_axislabel('RA')
		ax_inset.coords['dec'].set_axislabel('Dec')

		ax_inset.coords['ra'].set_ticklabel_position('top')
		ax_inset.coords['dec'].set_ticklabel_position('right')

		ax_inset.coords['ra'].set_ticks_position('top')
		ax_inset.coords['dec'].set_ticks_position('right')

		#bug! dec label goes to top axis when I use this method!
		ax_inset.coords['dec'].set_axislabel_position('top')

		#this works though....
		ax_inset.coords['ra'].set_axislabel_position('top')

		ax_inset.coords['ra'].set_ticklabel(rotation=45)

		ax.mark_inset_axes(ax_inset)
		ax.connect_inset_axes(ax_inset, 'upper left')
		ax.connect_inset_axes(ax_inset, 'lower right')

		#add slice boundary 
		#make a rectangle 
		lon=lonmax
		lat=latmax/2

		deltax1=lon*np.cos(np.deg2rad(spin0))

		deltax2=lat*np.sin(np.deg2rad(spin0))

		deltay1=lon*np.sin(np.deg2rad(spin0))

		deltay2=lat*np.cos(np.deg2rad(spin0))
		
		x1=lon0+deltax1+deltax2
		y1=lat0+deltay1+deltay2
		width=2*lonmax
		height=latmax

		rect1=Rectangle((x1,y1), width=width, height=height, angle=180+spin0, fill=False, ls='--', color=sliceboundarycolor, transform=ax.get_transform('fk5'))

		ax.add_patch(rect1)

		rect2=Rectangle((x1,y1), width=width, height=height, angle=180+spin0, fill=False, ls='--', color=sliceboundarycolor, transform=ax_inset.get_transform('fk5'))

		ax_inset.add_patch(rect2)
		
		#and the pie frame

		self.pieframe(fig, [0.6, 0, 0.3, 0.3])
		
			#plot points if data supplied
		if self.data is not None:
			mydata=self.slicer(data=self.data, show=False)
			ax.scatter(np.deg2rad(mydata['newlon'][:]), mydata['vel'][:], **kwargs)
		
		return ax, ax_inset, pieframe
	