import matplotlib
from matplotlib.colors import LogNorm
import pylab
import torch
import numpy

try:
    from astropy.visualization.wcsaxes import WCSAxes
    from astropy.visualization.wcsaxes.frame import EllipticalFrame
    from astropy import units as u
except:
    print("package *astropy* not found -> if you want to use plotting functionality for adaptive grids, install *astropy*!")

mhealpy_installed=False
try:
    import mhealpy
    from mhealpy import HealpixMap
    from mhealpy.plot.axes import HealpyAxes
    mhealpy_installed=True
except:
    print("package *mhealpy* not found -> if you want to use plotting functionality for adaptive grids, install *mhealpy*!")

from matplotlib.projections import register_projection
from matplotlib.transforms import Bbox

from ..contours import compute_contours, ContourGenerator
from .general import replace_axes_with_gridspec
from ...layers.spheres import sphere_base

###### plotting functions for sphere (s2), employing a flexible grid to save computing while still having smooth contours

def _transform_to_world(ax_object, coords, projection_type="zen_azi"):
    """
    Does the coordinate transformation to world coordaintes given a certain projection_type
    """
    pass

#### monkeypatching add function for more flexible ticklabels
def add(self,
        axis=None,
        world=None,
        pixel=None,
        angle=None,
        text=None,
        axis_displacement=None,
        data=None,
    ):
        """
        Add a label.

        Parameters
        ----------
        axis : str
            Axis to add label to.
        world : Quantity
            Coordinate value along this axis.
        pixel : [float, float]
            Pixel coordinates of the label. Deprecated and no longer used.
        angle : float
            Angle of the label.
        text : str
            Label text.
        axis_displacement : float
            Displacement from axis.
        data : [float, float]
            Data coordinates of the label.
        """
        pass

if(mhealpy_installed):
    class HealpyAxesAzimuthOrdering(HealpyAxes):

        def __init__(self, *args, rot = 0, **kwargs):
            """
            We creae a "custom" cylindrial axes here that is always fixed in terms of rotation
            """
            
            ## *args must contain fig and rect
            fig=args[0]
            rect=args[1:]

            if not isinstance(rect, Bbox):
                if(type(rect)==int or type(rect)==tuple):
                    ## whole figure.. assert that indices match with that
                    if(type(rect)==int):
                        assert(rect==1), rect
                        rect = Bbox.from_bounds(0,0,1,1)
                    elif(type(rect)==tuple):
                        if(len(rect)==1):
                            assert(isinstance(rect[0], Bbox))
                            rect = rect[0]
                        else:
                            assert(rect[0]==1 and rect[1]==1 and rect[2]==1), rect
                            rect = Bbox.from_bounds(0,0,1,1)
                    
                elif(type(rect)==list):
                    rect = Bbox.from_bounds(*rect)
                elif(type(rect)==matplotlib.gridspec.SubplotSpec):
                    rect=rect.get_position(fig).extents
                    rect=Bbox.from_bounds(*rect)

                else:
                    raise Exception("Unknown ax rect ", type(rect))

            ## default setting for 
            fixed_rot=numpy.array([0.0,90.0,0.0]) # rotation z set to 0 ensures azimuth starts at 0 
            
            super().__init__(fig, rect,
                             rot = fixed_rot,
                             flip="geo", # this ordering assures that azimuth spans from left to right
                             **kwargs)

        def graticule(self, 
                      dpar = 45, 
                      dmer = 60, 
                      grid = True,
                      ticks = True, 
                      show_zenith_label=True,
                      show_zenith_axis=True,
                      zenith_axislabel_minpad=2.0,
                      show_azimuth_label=True,
                      show_azimuth_axis=True,
                      tick_format = 'd', 
                      frame = True, 
                      zen_azi_mode="zen_azi",
                      text_size=12,
                      **kwargs):
            """
            Graticule overwrite.

            Args:
                dpar (float): Interval for the latitude axis (parallels)
                dmer (float): Interval for the longitude axis (meridians)
                grid (bool): Whether to show the grid
                ticks (bool): Whether to shoe the tick labels
                tick_format ('str'): Tick label formater. e.g. 'dd:mm:ss.s', 
                    'hh:mm:ss.s', 'd.d'
                frame (bool): Draw plot frame  
            """
            pass
                    
                    
            
    class MollviewAzimuth(HealpyAxesAzimuthOrdering):

        name = "mollview_azimuth"
        _wcsproj = "MOL"
        _aspect = 2

        ## multiply by 0.995 for correct labeling (due to rounding errors of WCSaxis)
        _cdelt = 2*numpy.sqrt(2)/numpy.pi*0.995 # Sqrt of pixel size
        _autoscale = True
        _center = [0,0]
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args,
                             frame_class = kwargs.pop('frame_class', EllipticalFrame),
                             **kwargs)

        def proj_plot(self, *args, **kwargs):
            """
            Calls matplotlib.plot in world coordinates, and does an internal transformation first.
            Internally the healpy ax uses dec/ra, so have to take of that here.
            """
            pass

            
    register_projection(MollviewAzimuth)

    class OrthviewAzimuth(HealpyAxes):

        name = "orthview_azimuth"
        _wcsproj = "SIN"
        _aspect  = 1
        _center = [0,0]
        # Sqrt of pixel area at point of tangency. Sign matches healpy
        _cdelt = -1/numpy.pi
        
        
        _autoscale = True

        def __init__(self, *args, zoom_center=None, zoom_diameter=None,**kwargs):

            ## *args must contain fig and rect
            fig=args[0]
            rect=args[1:]

            if not isinstance(rect, Bbox):
                if(type(rect)==int or type(rect)==tuple):
                    ## whole figure.. assert that indices match with that
                    if(type(rect)==int):
                        assert(rect==1), rect
                        rect = Bbox.from_bounds(0,0,1,1)
                    elif(type(rect)==tuple):
                        if(len(rect)==1):
                            assert(isinstance(rect[0], Bbox))
                            rect = rect[0]
                        else:
                            assert(rect[0]==1 and rect[1]==1 and rect[2]==1), rect
                            rect = Bbox.from_bounds(0,0,1,1)
                    
                elif(type(rect)==list):
                    rect = Bbox.from_bounds(*rect)
                elif(type(rect)==matplotlib.gridspec.SubplotSpec):
                    rect=rect.get_position(fig).extents
                    rect=Bbox.from_bounds(*rect)

                else:
                    raise Exception("Unknown ax rect ", type(rect))
            
            self._cdelt = 1/numpy.pi
            
            if(zoom_diameter is not None):
                # zoom_diameter is in radian .. only use if smaller than pi
                if(zoom_diameter<numpy.pi):
                    self._cdelt=(zoom_diameter/numpy.pi)*1/numpy.pi

                ## multiply by 0.995 for correct labeling (due to rounding errors of WCSaxis)
                self._cdelt=self._cdelt*0.995
                
            rot=(0.0,0.0,180.0)
            if(zoom_center is not None):
              
                assert(len(zoom_center)==2)
                # zoom_center is in zen/azimuth, but rotation entries are opposite order
                # last entry has to be 180.0 to yield correct view
                rot=(zoom_center[1]*180.0/numpy.pi, 90.0-(zoom_center[0]*180.0/numpy.pi), 180.0)

            super().__init__(fig, rect,
                             flip="geo",
                             rot=rot,
                             frame_class = kwargs.pop('frame_class', EllipticalFrame),
                             **kwargs)

        def graticule(self, 
                      dpar = 45, 
                      dmer = 60, 
                      grid = True,
                      ticks = True, 
                      show_zenith_label=True,
                      show_zenith_axis=True,
                      zenith_axislabel_minpad=4.0,
                      show_azimuth_label=True,
                      show_azimuth_axis=False,
                      tick_format = 'd', 
                      frame = True, 
                      zen_azi_mode="zen_azi",
                      text_size=12,
                      **kwargs):
            """
            Graticule overwrite.

            Args:
                dpar (float): Interval for the latitude axis (parallels)
                dmer (float): Interval for the longitude axis (meridians)
                grid (bool): Whether to show the grid
                ticks (bool): Whether to shoe the tick labels
                tick_format ('str'): Tick label formater. e.g. 'dd:mm:ss.s', 
                    'hh:mm:ss.s', 'd.d'
                frame (bool): Draw plot frame  
            """
            pass


        def proj_plot(self, *args, **kwargs):
            """
            Calls matplotlib.plot in world coordinates, and does an internal transformation first.
            Internally the healpy ax uses dec/ra, so have to take of that here.
            """
            pass

    register_projection(OrthviewAzimuth)

def get_meshed_positions_and_areas(samples, 
                                   max_entries_per_pixel=10):
    """
    Obtain the positions and areas of a meshed healpy grid. Meshing is based on samples, such that at max *max_entries_per_pixel*
    fall into a given mesh simplex.
    """
    assert(samples.shape[1]==2)
    used_samples=samples
    
    if(type(samples)==torch.Tensor):
        used_samples=samples.cpu().detach().numpy()
    assert(type(used_samples)==numpy.ndarray), type(samples)
    
    print(used_samples[:,0].min(),used_samples[:,0].max(), numpy.pi-used_samples[:,0].max())

    sample_pix = mhealpy.ang2pix(mhealpy.MAX_NSIDE, used_samples[:,0], used_samples[:,1], nest = True)
    
    moc_map=HealpixMap.moc_histogram(mhealpy.MAX_NSIDE, sample_pix, max_entries_per_pixel, nest=True)
    
    pix_ids=numpy.arange(moc_map.npix)
    ang_vals=moc_map.pix2ang(pix_ids)
    ang_vals=numpy.concatenate([ang_vals[0][:,None], ang_vals[1][:,None]], axis=1)
   
    per_pixel_nside, _ = mhealpy.uniq2nest(moc_map.pix2uniq(pix_ids))
    per_pixel_areas= mhealpy.nside2pixarea(per_pixel_nside)
    
    return ang_vals, per_pixel_areas, moc_map

def get_multiresolution_evals(pdf, 
                              conditional_input=None,
                              sub_pdf_index=0,
                              samplesize=10000, 
                              max_entries_per_pixel=5,
                              use_density_if_possible=True):
    """
    Sample a model and mesh the sky based on the samples, evaluate it, and calculate areas of mesh regions. 

    Returns:
        eval_positions (numpy.ndrray): Positions of mesh simplex centers.
        pdf_evals (numpy.ndarray): Evaluations of PDF at mesh simplex centers.
        eval_areas (numpy.ndarray): Mesh simplex areas.
        moc_map (Healpix map): Multiresolution healpix map
    """

    data_summary_repeated=None
    if(conditional_input is not None):
        data_summary_repeated=conditional_input

        if(type(conditional_input)==list):  
            if(conditional_input[0].ndim==2):
                assert(conditional_input[0].shape[0]==1), "Only a single conditional input item must be given!"
            data_summary_repeated=[ci.repeat_interleave(samplesize, dim=0) if ci.ndim==2 else ci[None,:].repeat_interleave(samplesize, dim=0) for ci in conditional_input]
        else:
            if(conditional_input.ndim==2):
                assert(conditional_input.shape[0]==1), "Only a single conditional input item must be given!"
            data_summary_repeated=conditional_input.repeat_interleave(samplesize, dim=0) if conditional_input.ndim==2 else conditional_input[None,:].repeat_interleave(samplesize, dim=0)
        
    samples,_,_,_=pdf.sample(samplesize=samplesize, conditional_input=data_summary_repeated)

    eval_positions, eval_areas, moc_map=get_meshed_positions_and_areas(samples,max_entries_per_pixel=max_entries_per_pixel)

    assert(pdf.pdf_defs_list[sub_pdf_index]=="s2"), ("Trying to get multiresolution for s2 subdimension, but subdimension %d is of type %s" % (sub_pdf_index, pdf.pdf_defs_list[sub_pdf_index]))

    if(use_density_if_possible and (sub_pdf_index==0)):
        xyz_positions=pdf.transform_target_into_returnable_params(torch.from_numpy(eval_positions).to(samples))

        if(data_summary_repeated is not None):
            
            moc_size=xyz_positions.shape[0]
            if(type(data_summary_repeated)==list):  
                assert(moc_size<=data_summary_repeated[0].shape[0])
                data_summary_repeated=[ci[:moc_size] for ci in data_summary_repeated]
            else:
                assert(moc_size<=data_summary_repeated.shape[0])
                data_summary_repeated=data_summary_repeated[:moc_size]

        log_pdf,_,_=pdf(xyz_positions, force_embedding_coordinates=True, conditional_input=data_summary_repeated)
        
        log_pdf=log_pdf.cpu().detach().numpy()
        pdf_evals=numpy.exp(log_pdf)

    else:

        
        ipix_vals=moc_map.ang2pix(samples[:,0], samples[:,1])
        sorted_pixel_vals=numpy.sort(ipix_vals)

       
        unique_indices, counts=numpy.unique(sorted_pixel_vals, return_counts=True)
       
        pdf_evals=numpy.zeros(len(eval_areas))
        pdf_evals[unique_indices]=counts/float(sum(counts))#eval_areas[unique_indices]
        pdf_evals=pdf_evals/eval_areas

        # no log_pdf in sample_based evaluation
        log_pdf=None

    return eval_positions, log_pdf, pdf_evals, eval_areas, moc_map


def plot_multiresolution_healpy(pdf,
                                fig=None, 
                                ax_to_plot=None,
                                samplesize=10000,
                                conditional_input=None, 
                                sub_pdf_index=0,
                                max_entries_per_pixel=5,
                                draw_pixels=True,
                                use_density_if_possible=True,
                                log_scale=True,
                                cbar=True,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=True,
                                contour_probs=[0.68, 0.95],
                                contour_colors=None, # None -> pick colors from color scheme
                                zoom=False,
                                zoom_contained_prob_mass=0.97,
                                projection_type="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None, # required to transform to dec/ra before plotting 
                                show_grid=False): 
    
    """
    Visualizes an S2 pdf, or a certain S2 subpart of a PDF using an adaptive healpix grid from mhealpy. Useful if the PDF
    is very small and higher nside becomes computationally expensive. Can also overlay smooth contours on this irregular grid
    using the *meander* package.

    """
    pass

def _plot_multiresolution_healpy(eval_positions,
                                pdf_evals,
                                eval_areas,
                                moc_map=None,
                                fig=None, 
                                ax_to_plot=None,
                                draw_pixels=True,
                                log_scale=True,
                                cbar=True,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=True,
                                contour_probs=[0.68, 0.95],
                                contour_colors=None, # None -> pick colors from color scheme
                                zoom=False,
                                zoom_contained_prob_mass=0.97,
                                projection_type="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None, # required to transform to dec/ra before plotting 
                                show_grid=False): 
    
    """
    Visualizes an S2 pdf, or a certain S2 subpart of a PDF using an adaptive healpix grid from mhealpy. Useful if the PDF
    is very small and higher nside becomes computationally expensive. Can also overlay smooth contours on this irregular grid
    using the *meander* package. This function differs in that it directly takes the PDF values instead of a pdf object.

    """
    pass

