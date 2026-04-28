import numpy
import numpy as np
from numpy import ma
import matplotlib
import matplotlib as mpl
from matplotlib import _api

try:
    import meander
except:
    print("Meander not installed... contours can not be calculated.")
    meander=None

try:
    import healpy
except:
    print("Healpy not installed... spherical contours can not be calculated.")
    healpy=None


def find_contour_levels(proportions, pdf_evals, areas):

    assert(len(pdf_evals.shape)==1), "pdf evals must be a 1-d array!"

    pdf_evals_with_area=pdf_evals*areas

    levels = []

    inv_sorted=numpy.argsort(pdf_evals)[::-1]
    
    sorted_pdf_with_area = pdf_evals_with_area[inv_sorted] #list(reversed(list(sorted(samples))))
    sorted_pdf=pdf_evals[inv_sorted]
   
    for proportion in proportions:
     
        larger_mask=numpy.cumsum(sorted_pdf_with_area) > proportion
        
        if(larger_mask.sum()>0):
            level_index = (numpy.cumsum(sorted_pdf_with_area) > proportion).tolist().index(True)
            
            level = (sorted_pdf[level_index] + (sorted_pdf[level_index+1] if level_index+1 < len(sorted_pdf_with_area) else 0)) / 2.0
            levels.append(level)
        else:
            ## did we alrady attach it? must have decreasing sequence
            
            levels.append(min(sorted_pdf))

    levels=numpy.array(levels)

    equal_last_levels=levels==min(sorted_pdf)

    if(equal_last_levels.sum()>1):
        # we have to make the last entries decreasing while being larger or equal than min
        last_ones=[]
        for ind in range(equal_last_levels.sum()):
            last_ones.append(min(sorted_pdf)*(1+ind*0.01))
        last_ones=last_ones[::-1]

 
        levels[equal_last_levels]=numpy.array(last_ones)

    tweak_offset=1e-10
    # Initialize the transformed array with the first element
    transformed_levels = [levels[0]]
    
    # Transform to a new array that has strictly decreasing elemeents
    for i in range(1, len(levels)):
        # If the current element is the same as the previous one
        if levels[i] == levels[i - 1]:
            # Calculate the number of times this element has appeared consecutively
            count = 1
            while i - count >= 0 and levels[i] == levels[i - count]:
                count += 1
            # Add the element with the offset
            transformed_levels.append(levels[i] - (count - 1) * tweak_offset)
        else:
            # If it's not a repeating element, just add it to the transformed array
            transformed_levels.append(levels[i])

    levels=numpy.array(transformed_levels)

    return levels

def compute_contours(proportions, pdf_evals, areas, sample_points=None, manifold="euclidean"):
    ''' Compute spherical contours using the meander package.

        Parameters:
        -----------
        proportions: list
            list of containment level to make contours for.
            E.g [0.68,0.9]
        samples: array
            array of values read in from healpix map
            E.g samples = hp.read_map(file)
        Returns:
        --------
        theta_list: list
            List of arrays containing theta values for desired contours
        phi_list: list
            List of arrays containing phi values for desired contours
    '''

    
    assert(meander is not None), "Spherical contour calculation requires meander!"

    
    levels=find_contour_levels(proportions, pdf_evals, areas)
    
    ##############################################
    combined_list=[]

    if(manifold=="euclidean"):
        assert(sample_points is not None)

        if(sample_points.shape[1]==1):
            for level in levels:
                contour=[]
                for i in range(len(pdf_evals) - 1):
                    if (pdf_evals[i] - level) * (pdf_evals[i + 1] - level) <= 0:
                        # Linear interpolation to find a more accurate point of crossing
                        x_contour = sample_points[i] + (level - pdf_evals[i]) * (sample_points[i + 1] - sample_points[i]) / (pdf_evals[i + 1] - pdf_evals[i])
                        contour.append(x_contour)
               
                ## create 1 "joint" 1-d contour here
                combined_list.append(numpy.array(contour)[...,None])
        elif(sample_points.shape[1]==2):
            contours_by_level = meander.planar_contours(sample_points, pdf_evals, levels)

    elif(manifold=="sphere"):

        assert(healpy is not None), "Spherical contour calculation requires healpy!"

        if(sample_points is None):
            nside = healpy.pixelfunc.get_nside(pdf_evals)
            sample_points = numpy.array(healpy.pix2ang(nside,numpy.arange(len(pdf_evals)))).T

        contours_by_level = meander.spherical_contours(sample_points, pdf_evals, levels)
    else:
        raise Exception("Unknown manifold for coverage! ", manifold)

    
    
    if(sample_points.shape[1]==2):
        theta_list = []
        phi_list=[]

        combined_list=[]

        for contours in contours_by_level:
            
            inner_list=[]

            for contour in contours:
               
                theta, phi = contour.T
                if(manifold=="sphere"):
                    phi[phi<0] += 2.0*numpy.pi
                inner_list.append(numpy.concatenate( [theta[:,None], phi[:,None]], axis=1))

            combined_list.append(inner_list)
       
    return combined_list

def find_1d_contours(proportions, xvals, pdf_evals_with_area, pdf_evals):
    """
    Find 1D contours for a given level of a 1D function.

    :param func: The 1D function.
    :param level: The level for which to find the contour points.
    :param domain: A tuple (start, end) representing the domain to search.
    :param num_points: Number of points to sample in the domain.
    :return: A list of x values (of size num_valuesX1) where func(x) is approximately equal to the level.
    """
    pass

"""
Custom contour generator for CustomSphereContourSet.
"""
class custom_contour_generator(object):
    
    def __init__(self, *args):

        self.contour_type=args[0]
        assert(self.contour_type=="euclidean" or self.contour_type=="zen_azi"), self.contour_type

        if(len(args)==4):
            self.has_precalculated_contours=True
            self.contour_probs=args[1]
            self.precalculated_contours=args[2]
            self.ax_obj=args[3]
            assert(len(self.contour_probs)==len(self.precalculated_contours))

        elif(len(args)==6):
            self.has_precalculated_contours=False
            self.joint_xy=np.concatenate([args[1][:,None],args[2][:,None]],axis=1)
            self.pdf_evals=args[3]
            self.areas=args[4]
            self.ax_obj=args[5]
        else:
            raise Exception("Require either 3 or 6 positional arguments for custom contour generator!")
    
    
    def _get_azimuth_split_contours(self, c, is_azimuthal=True):
        # split contour into isolated ones that get split by azimuth split
       
        pass
        
        
    def create_contour(self, contour_prob):
        
        pass
    
    

class ContourGenerator(matplotlib.contour.ContourSet):
    """
    A custom contour set that has similar structure to QuadContourSet in matplotlib,
    but is customized to work with variable resolution spherical data.
    """

    def _process_args(self, *args, corner_mask=None, algorithm=None, **kwargs):
        """
        Process args and kwargs.
        """
        pass

    def _contour_args(self, args, kwargs):
        pass

    def _check_xyz(self, x, y, z, kwargs):
        """
        Check that the shapes of the input arrays match; if x and y are 1D,
        convert them to 2D using meshgrid.
        """
        pass

    def _initialize_x_y(self, z):
        """
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i, j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        If extent is not None, x and y will be scaled to match,
        as in imshow.
        If origin is None and extent is not None, then extent
        will give the minimum and maximum values of x and y.
        """
        pass
