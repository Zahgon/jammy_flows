import torch
from torch import nn

from ..flow_options import check_flow_option, obtain_default_options, obtain_overall_flow_info
from ..extra_functions import list_from_str, NONLINEARITIES, recheck_sampling, find_init_pars_of_chained_blocks
from ..amortizable_mlp import AmortizableMLP
from ..helper_fns import contours, grid_functions
from ..helper_fns.coverage import calculate_approximate_coverage
from ..helper_fns.plotting.spherical import get_multiresolution_evals
import collections
import numpy
import copy
import sys
import scipy
import math
import time

try:
    import healpy
except:
    print("Cannot use healpy functionality. Install healpy, if you need to do entropy scanning!")

from scipy.special import iv, i0, i1
import scipy.linalg

from typing import Union

## used to peek into param generator which can be empty
def peek(iterable):
    pass

class pdf(nn.Module):

    def __init__(
        self,
        pdf_defs, 
        flow_defs, 
        options_overwrite=dict(),
        conditional_input_dim=None,
        amortization_mlp_dims="128",
        predict_log_normalization=False,
        join_poisson_and_pdf_description=False,
        hidden_mlp_dims_poisson="128",
        rank_of_mlp_mappings_poisson=0,
        amortization_mlp_use_custom_mode=False,
        amortization_mlp_ranks=0,
        amortization_mlp_highway_mode=0,
        amortize_everything=False,
        use_as_passthrough_instead_of_pdf=False,
        skip_mlp_initialization=False
    ):  
        """
        The main class of the project that defines a pytorch normalizing-flow PDF.
        The two main actions are evaluating the log-probability and sampling. Accessed via *jammy_flows.pdf*.

        Parameters:
            pdf_defs (str): String of characters describing the joint PDF structure: Sub-space structure is spearated by "+".
                            Example: "e2+s1+s2", describes a joint PDF over a 2-dimensional euclidean space, a 1-sphere and a 2-sphere: a joint 5-dimensional PDF.
            
            flow_defs (str): A string, that describes how each conditional subfflow defined in "pdfs_defs" is structured in terms of normalizing-flow layers.
                             Example: "gg+m+n" to describe a layer structure compatible with the other example "e2+s1+s2". Two letters mean two consecutive applications of a certain flow layer, 3 letters three etc.
                             Each layer holds their own parameters.
            
            options_overwrite (dict): Dictionary to overwrite default options of individual flow layers.

            conditional_input_dim (None/int/list(int)): Conditional input dimension if a conditional PDF. If a list is given, defines conditional input for each sub-pdf. None to define non-conditional PDF.

            amortization_mlp_dims (str/list(str)): Hidden structure of MLP for each sub-manifold. 

            predict_log_normalization (bool): Predict log-mean of Poisson distribution

            joint_poisson_and_pdf_description (bool): If predicting log-mean, predict it together with other flow parameters if this is a conditional PDF. Only makes sense when conditional_input_dim is not None.
    
            hidden_mlp_dims_poisson (str/list(str)): If the log-mean is predicted by its own MLP, defines the hidden structure of the MLP.
            
            amortization_mlp_use_custom_mode (bool): Use custom AmortizableMLP class instead of default chained Linear layers.

            rank_of_mlp_mappings_poisson (int): Max rank of custom Poisson predictor MLP matrices.
    
            amortization_mlp_highway_mode (int): Connectivity mode for custom MLPs if used.
    
            amortize_everything (bool): Indicates whether all parameters, including the MLPs, should be amortized.

            use_as_passthrough_instead_of_pdf (bool): Indicates, whether the class acts as a PDF, or only as a flow mapping function of the overall autoregressive flow.
            
            skip_mlp_initialization (bool): Indicates, whether to skip MLP inits entirely. Can be used for custom MLP initialization.
    

        """
        super().__init__()

        self.amortization_mlp_use_custom_mode=amortization_mlp_use_custom_mode
        self.predict_log_normalization=predict_log_normalization
        self.join_poisson_and_pdf_description=join_poisson_and_pdf_description
        self.amortization_mlp_highway_mode=amortization_mlp_highway_mode
        self.amortize_everything=amortize_everything

        if(self.amortize_everything):
            assert(self.predict_log_normalization==False), "Log Poisson prediction works only without full amortization in the default PDF. It can be used in the *fully_amortized_pdf*!"
        
        self.use_as_passthrough_instead_of_pdf=use_as_passthrough_instead_of_pdf
        self.skip_mlp_initialization=skip_mlp_initialization

        ## holds total number of params for amortization - only used if "amortize_everything" set to True
        self.total_number_amortizable_params=None

        if(self.amortize_everything):
            assert(self.amortization_mlp_use_custom_mode), "Amortizing all MLPs requires custom MLPs."
            self.total_number_amortizable_params=0
    
        self.read_model_definition(pdf_defs, 
                                   flow_defs, 
                                   options_overwrite, 
                                   conditional_input_dim, 
                                   amortization_mlp_dims, 
                                   amortization_mlp_ranks)
       
        self.init_flow_structure()
        
        self.hidden_mlp_dims_poisson=hidden_mlp_dims_poisson
        self.rank_of_mlp_mappings_poisson=rank_of_mlp_mappings_poisson

        self.init_encoding_structure()
       
        # add self reference for specific layers
        for subflow_index, subflow_description in enumerate(self.pdf_defs_list):

            layer_descr=self.flow_defs_list[subflow_index]

            for layer_index, layer in enumerate(self.layer_list[subflow_index]):

                ## sphere charts
                if(layer_descr[layer_index]=="c"):

                    layer.set_variables_from_parent(self)

        
        ## initialize params
        self.init_params()

    def read_model_definition(self, 
                              pdf_defs, 
                              flow_defs, 
                              options_overwrite,
                              conditional_input_dim,
                              amortization_mlp_dims,
                              amortization_mlp_ranks):
        
        # list of the pdf defs (e.g. e2 for 2-d Euclidean) for each subpdf
        # i.e. e2+s2 will yield 2 entries, one for each manifold
        pass

    def get_embedding_flags(self):
        """
        Returns embedding parameters flags (True/False) for different sub pdfs. An embedding flag
        is important for manifold sub dimensions. Set to True means the given sub-pdf flows expect the input to be in 
        embedding space, while set to False the sub-pdf flows expect their input to be in intrinsic coordinates.
        """
        pass

    def set_embedding_flags(self, usement_flag, sub_pdf_index=None):
        """
        Resets the default parameter embedding structure. This defines how tensors are to be given to the PDF without
        extra flags like *force_embedding_coordinates* or *force_intrinsic_coordinates*.

        Parameters:

            usement_Flag (bool): Defines if selected sub manifold should switch to embedding coordinates (True) or intrinsic coordinates (False).
            sub_pdf_index (int/None): The index of of the manifold for which to set the flag. If None, sets flag for all.
        """
        pass


    
    def init_flow_structure(self):

        pass

    def update_embedding_structure(self):

        pass
      


    def init_encoding_structure(self):

        pass

    def count_parameters(self, verbose=False):
        """
            Counts parameters of the model. It does not matter, if all paramters are amortized or not, will always return the same.
            
            Parameters:
                verbose (bool): Prints out number of parameters. Differentiates amortization from non-amortization params.
            
            Returns:
                int
                    Number of parameters (incl. amortization params).
        """
        pass

    def log_mean_poisson(self, conditional_input=None, amortization_parameters=None):
        """
        Calculates log-mean Poisson prediction.

        Parameters:
            conditional_input (Tensor/None): Must be tensor of appropriate dimension if conditional PDF. Otherwise None.
            amortization_parameters (Tensor/None): Used to amortize the whole PDF. Otherwise None.

        Returns:
            Tensor
                log-lambda of Poisson, size (B,1)
        """
        pass
        
    def all_layer_inverse(self, 
                          x, 
                          log_det, 
                          data_summary, 
                          amortization_parameters=None, 
                          force_embedding_coordinates=False, 
                          force_intrinsic_coordinates=False):
        """
        Performs the autoregressive (IAF) backward normalizing-flow mapping of all sub-manifold flows.

        Parameters:
            x (Tensor): Target Input.
            log_det (Tensor): Input log-Det. Soon Deprecated.
            data_summary (Tensor/None): Holds summary information for the conditional PDF. Otherwise None.
            amortization_parameters (Tensor/None): Used to amortize the whole PDF. Otherwise None.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the input x for this inverse mapping.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the input x for this inverse mapping.

        Returns: 
            Tensor
                Position in base space after inverse mapping.
            Tensor
                Log-Det factors of this inverse mapping.
        """

        ## make sure we transform to default settings (potentially mixed) if we force embedding/intrinsic coordinates
        if(force_embedding_coordinates):
            assert(x.shape[1]==self.total_target_dim_embedded), (x.shape[1], self.total_target_dim_embedded)

            x, log_det=self.transform_target_space(x, log_det, transform_from="embedding", transform_to="default")
        elif(force_intrinsic_coordinates):
            assert(x.shape[1]==self.total_target_dim_intrinsic)

            x, log_det=self.transform_target_space(x, log_det, transform_from="intrinsic", transform_to="default")
        else:
            assert(x.shape[1]==self.total_target_dim), (x.shape[1], self.total_target_dim)


        extra_conditional_input=[]
        base_targets=[]

        individual_logps=dict()

        extra_params = None

        if(amortization_parameters is not None):

            assert(amortization_parameters.shape[1]==self.total_number_amortizable_params)
            
        amort_param_counter=0

        for pdf_index, pdf_layers in enumerate(self.layer_list):

            extra_param_counter = 0
            this_pdf_type=self.pdf_defs_list[pdf_index]

            if(self.mlp_predictors[pdf_index] is not None):

                ## mlp preditors can be None for unresponsive layers like x/y
                if(data_summary is not None):

                    if(type(data_summary)==list):
                        this_data_summary=data_summary[pdf_index]
                    else:
                        this_data_summary=data_summary

                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat([this_data_summary]+extra_conditional_input, dim=1)
                    
                    if(amortization_parameters is not None):
                        num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                        extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                        amort_param_counter+=num_amortization_params

                    else:
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                   
                else:

                    ## data summary is None (non-conditional pdf) .. just encode previous dims
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat(extra_conditional_input, dim=1)
                        
                        if(amortization_parameters is not None):
                            num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                            extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                            amort_param_counter+=num_amortization_params

                        else:
                            extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                        
                    else:
                        raise Exception("FORWARD: extra conditional input is empty but required for encoding!")

                if(self.predict_log_normalization):
                    if( (pdf_index==0) and self.join_poisson_and_pdf_description):
                        extra_params=extra_params[:,:-1]


            else:
                ## we amortize everything with amortization_parameters .. including the first layer column if there is no encoder
                if(self.amortize_everything):
                    assert(amortization_parameters is not None)

                    tot_num_params=0
                    for l in self.layer_list[0]:
                        tot_num_params+=l.get_total_param_num()

                    if(tot_num_params>0):
                        extra_params=amortization_parameters[:,:tot_num_params]

                        amort_param_counter+=tot_num_params

            this_target=x[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]
            
            ## reverse mapping is required for pdf evaluation
            for l, layer in reversed(list(enumerate(pdf_layers))):

                this_extra_params = None

                if extra_params is not None:

                    if extra_param_counter == 0:
                            this_extra_params = extra_params[:, -layer.total_param_num :]
                    else:

                        this_extra_params = extra_params[
                            :,
                            -extra_param_counter
                            - layer.total_param_num : -extra_param_counter,
                        ]

                
                if(l==(len(pdf_layers)-1)):
                    # force embedding or intrinsic coordinates in the layer that defines the target dimension
                    this_target, log_det = layer.inv_flow_mapping([this_target, log_det], extra_inputs=this_extra_params)
                else:

                    this_target, log_det = layer.inv_flow_mapping([this_target, log_det], extra_inputs=this_extra_params)
                
                extra_param_counter += layer.total_param_num

            if(False):
                ## stems from debugging purposes, not used currently
                ind_base_eval=this_logp = torch.distributions.MultivariateNormal(
                    torch.zeros_like(this_target).to(x),
                    covariance_matrix=torch.eye(this_target.shape[1]).type_as(x).to(x),
                ).log_prob(this_target)

                ind_logdet=log_det
                

                individual_logps["%.2d_%s" % (pdf_index, this_pdf_type)]=ind_base_eval+ind_logdet
                individual_logps["%.2d_%s_logdet" % (pdf_index, this_pdf_type)]=ind_logdet
                individual_logps["%.2d_%s_base" % (pdf_index, this_pdf_type)]=ind_base_eval

            
            base_targets.append(this_target)

            prev_target=x[:,self.target_dim_indices[pdf_index][0]:self.target_dim_indices[pdf_index][1]]
            prev_target=pdf_layers[-1]._embedding_conditional_return(prev_target)

            extra_conditional_input.append(prev_target)

        base_pos=torch.cat(base_targets, dim=1)

        return base_pos, log_det

    def forward(self, 
                x, 
                conditional_input=None,
                amortization_parameters=None, 
                force_embedding_coordinates=False, 
                force_intrinsic_coordinates=False):
        """
        Calculates log-probability at the target *x*. Also returns some other quantities that are calculated as a consequence.

        Parameters:

            x (Tensor): Target position to calculate log-probability at. Must be of shape (B,D), where B = batch dimension.
            conditional_input (Tensor/list(Tensor)/None): Amortization input for conditional PDFs. If given, must be of shape (B,A), where A is the conditional input dimension defined in __init__. Can also be 
                              a list of tensors, one for each sub-PDF, if *conditional_input_dim* in __init__ is a list of ints.
            amortization_parameters (Tensor/None): If the PDF is fully amortized, defines all the parameters of the PDF. Must be of shape (B,T), where T is the total number of parameters of the PDF.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the input *x*.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the input *x*. 
        
        Returns:

            Tensor
                Log-probability, shape = (B,)

            Tensor
                Log-probability at base distribution, shape = (B,)

            Tensor
                Position at base distribution, shape = (B,D)

        """
        assert(self.use_as_passthrough_instead_of_pdf == False), "The module is only used as a passthrough of all layers, not as actually evaluating the pdf!"
        if(conditional_input is not None):
            if(type(conditional_input)==list):

                assert(len(self.conditional_input_dim)==len(conditional_input))
                for ci_ind in range(len(self.conditional_input_dim)):
                    assert(self.conditional_input_dim[ci_ind]==conditional_input[ci_ind].shape[1]), "Inputs of conditional input vector do not match with pre-defined input_dims!"

                for ci in conditional_input:
                    assert(x.shape[0]==ci.shape[0]), "Evaluating input x and condititional input shape must be similar!"
                    assert(x.is_cuda==ci.is_cuda), ("input tensor *x* and *conditional_input* are on different devices .. resp. cuda flags: 1) x, 2) conditional_input, 3) pdf model", x.is_cuda, ci.is_cuda, next(self.parameters()).is_cuda)
            else:
                assert(x.shape[0]==conditional_input.shape[0]), "Evaluating input x and condititional input shape must be similar!"
                assert(x.is_cuda==conditional_input.is_cuda), ("input tensor *x* and *conditional_input* are on different devices .. resp. cuda flags: 1) x, 2) conditional_input, 3) pdf model", x.is_cuda, conditional_input.is_cuda, next(self.parameters()).is_cuda)

        tot_log_det = torch.zeros(x.shape[0]).type_as(x)

        base_pos, tot_log_det=self.all_layer_inverse(x, tot_log_det, conditional_input, amortization_parameters=amortization_parameters, force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)

        ## must faster calculation based on std normal
        other=torch.distributions.Normal(
            0.0,
            1.0,
        ).log_prob(base_pos)

        log_pdf=other.sum(dim=-1)

        return log_pdf + tot_log_det, log_pdf, base_pos

    def obtain_flow_param_structure(self, 
                                    conditional_input=None, 
                                    predefined_target_input=None, 
                                    seed=None,
                                    dtype=None,
                                    device=None):
        """
        Obtain values of flow parameters for given input along with their name. For debugging and plotting purposes mostly.
        """
        pass

    def sample(self, 
               conditional_input=None, 
               samplesize=1,  
               seed=None, 
               allow_gradients=False, 
               amortization_parameters=None, 
               force_embedding_coordinates=False, 
               force_intrinsic_coordinates=False,
               failsafe_crosscheck_tolerance=None,
               dtype=None,
               device=None):
        """ 
        Samples from the (conditional) PDF. 

        Parameters:
            conditional_input (Tensor/list(Tensor)/None): Tensor of shape B x D where B is the batch size and D the input space dimension if given. Can also be a list of tensors, which must share batch dimensionality. Else None.
            samplesize (int): Samplesize.
            seed (None/int):
            allow_gradients (bool): If False, does not propagate gradients and saves memory by not building the graph. Off by default, so has to be switched on for training.
            amortization_parameters (Tensor/None): Used to amortize the whole PDF. Otherwise None.
            force_embedding_coordinates (bool): Enforces embedding coordinates for the sample.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates for the sample.
            dtype (torch dtype): Dtype and device are normally inferred by parameters or conditional input. If no parameters are part of the 
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.

        Returns:

            Tensor
                Sample in target space.
            Tensor
                Sample in base space.
            Tensor
                Log-pdf evaluation in target space
            Tensor
                Log-pdf evaluation in base space


        """
       
        assert(self.use_as_passthrough_instead_of_pdf == False), "The module is only used as a passthrough of all layers, not as actually evaluating the pdf or sampling from the pdf!"

        if(allow_gradients):

            sample, normal_base_sample, log_pdf_target, log_pdf_base=self._obtain_sample(conditional_input=conditional_input, 
                                                                                         seed=seed, 
                                                                                         samplesize=samplesize, 
                                                                                         amortization_parameters=amortization_parameters, 
                                                                                         force_embedding_coordinates=force_embedding_coordinates, 
                                                                                         force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                                                         failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                                                                                         device=device,
                                                                                         dtype=dtype)


            return sample, normal_base_sample, log_pdf_target, log_pdf_base

        else:   
            with torch.no_grad():
                sample, normal_base_sample, log_pdf_target, log_pdf_base=self._obtain_sample(conditional_input=conditional_input, 
                                                                                             seed=seed, 
                                                                                             samplesize=samplesize, 
                                                                                             amortization_parameters=amortization_parameters, 
                                                                                             force_embedding_coordinates=force_embedding_coordinates, 
                                                                                             force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                                                             failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                                                                                             device=device,
                                                                                             dtype=dtype)
           
            return sample, normal_base_sample, log_pdf_target, log_pdf_base

    def all_layer_forward(self, 
                          x,   
                          log_det,   
                          data_summary, 
                          amortization_parameters=None,
                          force_embedding_coordinates=False, 
                          force_intrinsic_coordinates=False):

        """
        Performs the autoregressive (IAF) forward normalizing-flow mapping of all sub-manifold flows.

        Parameters:
            x (Tensor): Target Input.
            log_det (Tensor): Input log-Det. Soon Deprecated.
            data_summary (Tensor/None): Holds summary information for the conditional PDF. Otherwise None.
            amortization_parameters (Tensor/None): Used to amortize the whole PDF. Otherwise None.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the output sample.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the output sample.

        Returns: 
            Tensor
                Position in the target space after sampling.
            
            Tensor
                Log-Det factors of the forward mapping.
        """

        extra_conditional_input=[]
        new_targets=[]

        if(amortization_parameters is not None):

            #assert(amortization_parameters.shape[0]==x.shape[0]), ("batch size of x must agree with batch size of amortization_parameters")
            assert(amortization_parameters.shape[1]==self.total_number_amortizable_params), (amortization_parameters.shape[1], self.total_number_amortizable_params)
        else:
            assert(self.amortize_everything==False)

        amort_param_counter=0

        for pdf_index, pdf_layers in enumerate(self.layer_list):

            this_pdf_type=self.pdf_defs_list[pdf_index]

            ## by default not extra_params for the layers
            extra_params = None

            if(self.mlp_predictors[pdf_index] is not None):

                if(data_summary is not None):
                    # conditional PDF (data_summary!=None) and MLP predictor given
                    if(type(data_summary)==list):
                        this_data_summary=data_summary[pdf_index]
                    else:   
                        this_data_summary=data_summary
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat([this_data_summary]+extra_conditional_input, dim=1)

                    if(amortization_parameters is not None):
                        num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                        extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                        amort_param_counter+=num_amortization_params

                    else:
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)

               
                else:

                
                    ## no conditional PDF (data_summary==None) but MLP predictor there
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat(extra_conditional_input, dim=1)
                        
                        if(amortization_parameters is not None):
                            num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                            extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                            amort_param_counter+=num_amortization_params

                        else:
                            extra_params=self.mlp_predictors[pdf_index](this_data_summary)

                    else:
                        raise Exception("SAMPLE: extra conditional input is empty but required for encoding!")

                if(self.predict_log_normalization):
                    if( (pdf_index==0) and self.join_poisson_and_pdf_description):
                        extra_params=extra_params[:,:-1]

            else:
                ## we amortize everything with amortization_parameters .. including the first layer if there is no encoder
                if(self.amortize_everything):
                    assert(amortization_parameters is not None)
                    
                    tot_num_params=0
                    for l in self.layer_list[pdf_index]:
                        tot_num_params+=l.get_total_param_num()

                    if(tot_num_params>0):
                        extra_params=amortization_parameters[:,amort_param_counter:amort_param_counter+tot_num_params]
                        amort_param_counter+=tot_num_params

            this_target=x[:,self.base_dim_indices[pdf_index][0]:self.base_dim_indices[pdf_index][1]]

            ## loop through all layers in each pdf and transform "this_target"
            
            extra_param_counter = 0
            for l, layer in list(enumerate(pdf_layers)):
               
                this_extra_params = None
                
                if extra_params is not None:
                    
                    this_extra_params = extra_params[:, extra_param_counter : extra_param_counter + layer.total_param_num]
              
                if(l==(len(pdf_layers)-1)):
                    this_target, log_det = layer.flow_mapping([this_target, log_det], extra_inputs=this_extra_params)
                else:
                    this_target, log_det = layer.flow_mapping([this_target, log_det], extra_inputs=this_extra_params)
                
                extra_param_counter += layer.total_param_num

            new_targets.append(this_target)

            prev_target=this_target
           
            prev_target=self.layer_list[pdf_index][-1]._embedding_conditional_return(prev_target)

            extra_conditional_input.append(prev_target)

        if (torch.isfinite(x) == 0).sum() > 0:
            raise Exception("nonfinite samples generated .. this should never happen!")
        
        x=torch.cat(new_targets, dim=1)

        ## transform to desired output space 
        if(force_embedding_coordinates):

            x, log_det=self.transform_target_space(x, log_det, transform_from="default", transform_to="embedding")
            
        elif(force_intrinsic_coordinates):
            assert(x.shape[1]==self.total_target_dim_intrinsic)
           
            x, log_det=self.transform_target_space(x, log_det, transform_from="default", transform_to="intrinsic")
      
        return x, log_det

    def _obtain_sample(self, 
                       conditional_input=None, 
                       predefined_target_input=None, 
                       samplesize=1, 
                       seed=None, 
                       amortization_parameters=None, 
                       force_embedding_coordinates=False, 
                       force_intrinsic_coordinates=False,
                       failsafe_crosscheck_tolerance=None,
                       dtype=None,
                       device=None):
        """
        Obtains a sample from the Multivariate Standard Normal, evaluates it and passes it through forward machinery. 
        When *predefined_target_input* is given, takes this as a sample.

        Parameters:

            conditional_input (Tensor/None): Input tensor when conditional PDF.
            predefined_target_input (Tensor/None): When given, evaluates the MVN there. Otherwise samples a MVN random variable before.
            samplesize (int):
            seed (None/int):
            amortization_parameters (bool): Used to amortize the whole PDF. Otherwise None.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the output sample.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the output sample.
            dtype (torch dtype): If given, uses this dtype. Otherwise uses dtype from parameters.
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.

        Returns:

            Tensor
                Sample in target space.
            Tensor
                Sample in base space.
            Tensor
                Log-pdf evaluation in target space
            Tensor
                Log-pdf evaluation in base space
        """

        used_sample_size = samplesize

        ## some crosschecks
        if(conditional_input is not None):
            if(type(conditional_input)==list):

                assert(len(self.conditional_input_dim)==len(conditional_input))
                for ci_ind in range(len(self.conditional_input_dim)):
                    assert(self.conditional_input_dim[ci_ind]==conditional_input[ci_ind].shape[1]), "Inputs of conditional input vector do not match with pre-defined input_dims!"

                for ci_ind, ci in enumerate(conditional_input[:-1]):

                    assert(ci.shape[0]==conditional_input[ci_ind].shape[0]), "Conditional input batch sizes do not agree!"
                    assert(ci.dtype==conditional_input[ci_ind].dtype), "Conditional input types do not match!"
                    assert(ci.device==conditional_input[ci_ind].device), "Conditional input devices do not match!"


        # make sure device is set if amortization is used
        if(self.amortize_everything):
            assert(amortization_parameters is not None)
            used_device=amortization_parameters.device
            data_type=amortization_parameters.dtype
            used_sample_size=amortization_parameters.shape[0]

            if(conditional_input is not None):
                ## TODO - maybe allow for more flexible shape combinations
                if(type(conditional_input)==list):
                    assert(conditional_input[0].shape[0]==amortization_parameters.shape[0])
                    assert(conditional_input[0].device==amortization_parameters.device)
                    assert(conditional_input[0].dtype==amortization_parameters.dtype), "Dtypes between conditional_input and amortization_paramters have to agree!"
                else:
                    assert(conditional_input.shape[0]==amortization_parameters.shape[0])
                    assert(conditional_input.device==amortization_parameters.device)
                    assert(conditional_input.dtype==amortization_parameters.dtype), "Dtypes between conditional_input and amortization_paramters have to agree!"

        elif(conditional_input is not None):

                if(type(conditional_input)==list):
                    used_sample_size = conditional_input[0].shape[0]
                    data_type = conditional_input[0].dtype
                    used_device = conditional_input[0].device
                else:
                    used_sample_size = conditional_input.shape[0]
                    data_type = conditional_input.dtype
                    used_device = conditional_input.device

        else:
            ## if one blindly uses next() on an empty param generator, it throws an error
            data_type, used_device=self.obtain_current_dtype_n_device()

            if(device is not None):
                used_device=device
            if(dtype is not None):
                data_type=dtype

        assert( (data_type is not None) and (used_device is not None)), "DType and/or device is None. This can only happen if layers without any parameters are used. In this case, you have to define dtype and device as keyword arguments!"

        x=None
        log_gauss_evals=0.0
        std_normal_samples=0.0
       
        if(predefined_target_input is not None):

            x=predefined_target_input

            assert(used_device==predefined_target_input.device)

            if(conditional_input is not None):

                ## make sure inputs agree

                if(type(conditional_input)==list):
                    assert(x.shape[0]==conditional_input[0].shape[0])
                    assert(x.dtype==conditional_input[0].dtype)
                    assert(x.device==conditional_input[0].device)
                else:
                    assert(x.shape[0]==conditional_input.shape[0])
                    assert(x.dtype==conditional_input.dtype)
                    assert(x.device==conditional_input.device)

            else:
                data_type=predefined_target_input.dtype
                used_sample_size=predefined_target_input.shape[0]
              
            log_gauss_evals=torch.distributions.Normal(0.0,1.0).log_prob(predefined_target_input).sum(dim=-1)

        else:

            if(seed is not None):
                numpy.random.seed(seed)

            std_normal = numpy.random.normal(size=(used_sample_size, self.total_base_dim))

            std_normal_samples = (
                torch.from_numpy(std_normal).type(data_type).to(used_device)
            )

            log_gauss_evals=torch.distributions.Normal(0.0,1.0).log_prob(std_normal_samples).sum(dim=-1)
            
            x = std_normal_samples

        log_det = torch.zeros(used_sample_size).type(data_type).to(used_device)
        
        new_targets, log_det=self.all_layer_forward(x, log_det, conditional_input, amortization_parameters=amortization_parameters, force_embedding_coordinates=force_embedding_coordinates, force_intrinsic_coordinates=force_intrinsic_coordinates)

        ## failsafe crosscheck?

        return_log_pdf=-log_det + log_gauss_evals

        if(failsafe_crosscheck_tolerance):

            assert(predefined_target_input is None), "Failsafe does not work with predefined input!"
            
            new_targets_prop, std_normal_samples_prop, return_log_pdf_prop, log_gauss_evals_prop=recheck_sampling(self, 
                      new_targets,
                      std_normal_samples,
                      return_log_pdf,
                      log_gauss_evals,
                      failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                      conditional_input=conditional_input,
                      amortization_parameters=amortization_parameters,
                      force_embedding_coordinates=force_embedding_coordinates,
                      force_intrinsic_coordinates=force_intrinsic_coordinates,
                      dtype=data_type,
                      device=used_device)

            if(new_targets_prop is not None):
                new_targets=new_targets_prop
                std_normal_samples=std_normal_samples_prop
                return_log_pdf=return_log_pdf_prop
                log_gauss_evals=log_gauss_evals_prop
            

        ## -logdet because log_det in sampling is derivative of forward function d/dx(f), but log_p requires derivative of backward function d/dx(f^-1) whcih flips the sign here
        return new_targets, std_normal_samples, return_log_pdf, log_gauss_evals

    def get_total_embedding_dim(self):
        """Returns embedding dimension of the overall PDF."""
        pass

    ### 
    def transform_target_into_returnable_params(self, target):
        """ 
        Transforms an input tensor from default to embedding parametrization.

        Parameters:
            target (Tensor): Input tensor in current default parametrization.

        Returns:
            Tensor
                Output tensor in embedding parametrization.

        """

        
        res, _=self.transform_target_space(target)

        return res

    def transform_target_space(self, 
                              target, 
                              log_det=0, 
                              transform_from="default", 
                              transform_to="embedding"):
        """
        Transform the destimation space tensor of the PDF as defined by *transform_from* and *transform_to*, which is the embedding space. (I.e. transform spherical angles into x/y/z pairs and so on when *transform_to* is embedding space.)
        
        Parameters:

            target (Tensor): Tensor to transform.
            log_det (Tensor): log_det Tensor (Soon Deprecated)
            transform_from (str): Coordinates to start with. One of "default"/"intrinsic"/"embedding".
            transform_to (str): Coordinates to end with. One of "default"/"intrinsic"/"embedding".

        Returns:

            Tensor
                 Target in new coordinates.

            
            Tensor
                Any additional log-det added to input log-det. 
        """

        new_target=target

        if(len(target.shape)==1):
            new_target=target.unsqueeze(0)

        ## transforming only makes sense if input has correct target shape
        if(transform_from=="default"):
            assert(new_target.shape[1]==self.total_target_dim)
        elif(transform_from=="intrinsic"):
            assert(new_target.shape[1]==self.total_target_dim_intrinsic)
        elif(transform_from=="embedding"):
            assert(new_target.shape[1]==self.total_target_dim_embedded) 
        else:
            raise Exception("Unknown transformation space! .. ", transform_from, "Allowed: default/intrinsic/embedding")
            

        potentially_transformed_vals=[]

        index=0
        for pdf_index, pdf_type in enumerate(self.pdf_defs_list):

            if(transform_from=="default"):
                this_dim=self.target_dims[pdf_index]
            elif(transform_from=="intrinsic"):
                this_dim=self.target_dims_intrinsic[pdf_index]
            elif(transform_from=="embedding"):
                this_dim=self.target_dims_embedded[pdf_index]
            else:
                raise Exception("Unknown transformation space! .. ", transform_from, "Allowed: default/intrinsic/embedding")
            
            this_target, log_det=self.layer_list[pdf_index][-1].transform_target_space(new_target[:,index:index+this_dim], log_det=log_det, transform_from=transform_from, transform_to=transform_to)
            
            potentially_transformed_vals.append(this_target)

            index+=this_dim

        potentially_transformed_vals=torch.cat(potentially_transformed_vals, dim=1)

        if(transform_to=="default"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim)
        elif(transform_to=="intrinsic"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim_intrinsic), (potentially_transformed_vals.shape, self.total_target_dim_intrinsic)
        elif(transform_to=="embedding"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim_embedded), (new_target.shape[1], self.total_target_dim_embedded)  
        else:
            raise Exception("Unknown transformation space! .. ", transform_to, "Allowed: default/intrinsic/embedding")
            

        if(len(target.shape)==1):
            potentially_transformed_vals=potentially_transformed_vals.squeeze(0)

        return potentially_transformed_vals, log_det

    ########

    def init_params(self, data=None, damping_factor=1000.0, mvn_min_max_sv_ratio=1e-4):
        """
        Initialize params of the normalizing flow such that the different sub flows play nice with each other and the starting distribution is a reasonable one.
        For the Gaussianization flow, data can be used to initilialize the starting distribution such that it roughly follows the data.
            
        Parameters:
            data (None/Tensor): If given a Tensor with target data, Gaussianization Flow subflows can make use of the distribution and initialize such that they follow the distribution.
            damping_factor (float): Weights in final matrices of amortization MLPs are divided by this factor (after already having been initialized) to dampen the impact of previous flow layers and conditional input in the autoregressive amortization structure.

        """

        global_amortization_init=None
        global_amortization_index=0
       
        if(self.amortize_everything):
            global_amortization_init=torch.zeros(self.total_number_amortizable_params)

        with torch.no_grad():
            ## 0) check data
            if(data is not None):
                ## initialization data has to match pdf dimenions
                assert(data.shape[1]==self.total_target_dim), "Initialization with data must match the target dimension of the PDF!"

            ## 1) Find initialization params of all layers - each index corresponds to all flows from a given sub manifold
            params_list=[]
            ## loop through all the layers and get the initializing parameters

            this_dim_index=0

            for subflow_index, subflow_description in enumerate(self.pdf_defs_list):
                
                this_dim=self.target_dims[subflow_index]

                this_layer_list=self.layer_list[subflow_index]

                if("e" in subflow_description):
                    
                    params=find_init_pars_of_chained_blocks(this_layer_list, data[:, this_dim_index:this_dim_index+this_dim] if data is not None else None, mvn_min_max_sv_ratio=mvn_min_max_sv_ratio)

                    params_list.append(params)

                else:
                    
                    this_list=[]
                    for l in this_layer_list:
                        this_list.append(l.get_desired_init_parameters())

                    params_list.append(torch.cat(this_list))

                this_dim_index+=this_dim

            ## 2) Depending on encoding structure, use the init params at appropriate places
       

            if(self.predict_log_normalization):

                if(self.join_poisson_and_pdf_description):
                    if(len(self.mlp_predictors)>1):
                        ## TODO: need to find a good way to predict the log-normaliaztion as a joint parameter when multiple mlp predictors are present
                        raise NotImplementedError


            # loop through all mlps
            for ind, mlp_predictor in enumerate(self.mlp_predictors):
                
                # these are the desired params at initialization for the MLP -> set bias of last MLP layer to these values
                # and make the weights and bias in previous layers very small
               
                these_params=params_list[ind]

                if(len(these_params)>0):

                    if(mlp_predictor is not None):
                       
                        ## the first MLP can predict log_lambda if desired
                        ## attach desired log-lambda to this bunch of params
                        if(self.predict_log_normalization):
                            if(self.join_poisson_and_pdf_description):
                                if(ind==0):
                                    log_lambda_init=0.1
                                    these_params=torch.cat([these_params, torch.Tensor([log_lambda_init]).type(mlp_predictor[-1].bias.data.dtype)])

                        ## custom low-rank MLPs - initialization is done inside the custom MLP class
                        if(type(mlp_predictor)== AmortizableMLP):
                           
                            if(self.amortize_everything):
                                desired_uvb_params=mlp_predictor.obtain_default_init_tensor(fix_final_bias=these_params, prev_damping_factor=damping_factor)
                                num_uvb_pars=mlp_predictor.num_amortization_params
                                global_amortization_init[global_amortization_index:global_amortization_index+num_uvb_pars]=desired_uvb_params
                                global_amortization_index+=num_uvb_pars
                            else:
                                mlp_predictor.initialize_uvbs(fix_final_bias=these_params, prev_damping_factor=damping_factor)

                        else:
                            # initialize all layers
                            for internal_layer in mlp_predictor:
                                
                                # test if this is a real Linear layer or a nonlinearity
                                if(hasattr(internal_layer, "weight")):

                                    # only initialize if a Linear layer
                                    nn.init.kaiming_uniform_(internal_layer.weight.data, a=numpy.sqrt(5))
                                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(internal_layer.weight.data)
                                    bound = 1 / numpy.sqrt(fan_in)

                                    nn.init.uniform_(internal_layer.bias.data, -bound, bound)
                                    
                                    internal_layer.weight.data/=damping_factor
                                    internal_layer.bias.data/=damping_factor
                                
                            # finally overwrite bias to be equivalent to desired parameters at initialization
                            
                            mlp_predictor[-1].bias.data=these_params.data.type(mlp_predictor[-1].bias.data.dtype)#torch.ones_like(these_params.data)*0.44  # .copy_(torch.randn(these_params.shape))

                    else:
                        ## threre is no MLP - initialize parameters of flows directly
                        tot_param_index=0
                  
                        for layer_ind, layer in enumerate(self.layer_list[ind]):
                            
                            this_layer_num_params=self.layer_list[ind][layer_ind].get_total_param_num()
                           
                            if(self.amortize_everything==False):
                                self.layer_list[ind][layer_ind].init_params(these_params[tot_param_index:tot_param_index+this_layer_num_params])

                            tot_param_index+=this_layer_num_params

                        if(self.amortize_everything):
                            global_amortization_init[global_amortization_index:global_amortization_index+tot_param_index]=these_params
                            global_amortization_index+=tot_param_index

                        if(ind==0 and self.amortize_everything and self.predict_log_normalization):
                            global_amortization_init[global_amortization_index+1]=0.1
                            global_amortization_index+=1
        
        return global_amortization_init

    def approximate_coverage(self, 
                target_x,
                conditional_input=None,
                amortization_parameters=None, 
                force_embedding_coordinates=False, 
                force_intrinsic_coordinates=False,
                num_percentile_points=100,
                sub_manifolds=[-1]):
        """
        Calculates approximate coverage via the base distribution with the quantity 2*(log(p(0))-log(p_(z_base))) which should be chi^2 distributed for good coverage.

        Parameters:

            target_x (Tensor): Target positions to calculate coverage with. Must be of shape (B,D), where B = batch dimension.
            conditional_input (Tensor/list(Tensor)/None): Amortization input for conditional PDFs. If given, must be of shape (B,A), where A is the conditional input dimension defined in __init__. Can also be 
                              a list of tensors, one for each sub-PDF, if *conditional_input_dim* in __init__ is a list of ints.
            amortization_parameters (Tensor/None): If the PDF is fully amortized, defines all the parameters of the PDF. Must be of shape (B,T), where T is the total number of parameters of the PDF.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the input *x*.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the input *x*. 
            num_percentile_points (int): At how many points along the chi2 do we want to compare true vs expected coverage?
            sub_manifolds (list(int)): Contains indices of sub-manifolds if coverage should be calculated for onditional PDF of the given sub-manifold. *-1* stands for the total PDF and is the default.

        Returns:

            expected_coverage_probs (Numpy array): Array of expected coverage probabilities.
            actual_coverage_probs (Numpy array): Array of actual coverage probabilities.
            actual_twice_logprob (Numpy array): Array of twice the log-probability difference at the base.
        """
        pass

    def coverage_and_or_pdf_scan(self,
                                 labels=None,
                                 conditional_input=None,
                                 amortization_parameters=None, 
                                 coverage_num_percentile_points=100,
                                 sub_manifolds=[-1],
                                 exact_coverage_calculation=False,
                                 save_pdf_scan=False,
                                 calculate_MAP=False):

        """
        Calculates coverage (approximate) and possibly exact. Performs pdf scan for exact coverage and save scan if desired.
        The pdf scan and the exact coverage must be calculated in intrinsic coordinates (e.g. theta/phi for direction instead of dx/dy/dz).

        Parameters:

            labels (Tensor): Target positions to calculate coverage with. Must be of shape (B,D), where B = batch dimension.
            conditional_input (Tensor/list(Tensor)/None): Amortization input for conditional PDFs. If given, must be of shape (B,A), where A is the conditional input dimension defined in __init__. Can also be 
                              a list of tensors, one for each sub-PDF, if *conditional_input_dim* in __init__ is a list of ints.
            amortization_parameters (Tensor/None): If the PDF is fully amortized, defines all the parameters of the PDF. Must be of shape (B,T), where T is the total number of parameters of the PDF.
            force_embedding_coordinates (bool): Enforces embedding coordinates in the input *x*.
            force_intrinsic_coordinates (bool): Enforces intrinsic coordinates in the input *x*. 
            coverage_num_percentile_points (int): At how many points along the chi2 do we want to compare observed vs expected coverage?
            sub_manifolds (list(int)): Contains indices of sub-manifolds if coverage should be calculated for onditional PDF of the given sub-manifold. *-1* stands for the total PDF and is the default.
            exact_coverage_calculation (bool): Calculate exact coverage based on pdf scan?
            save_pdf_scan (bool): Save a pdf scan?
            calculate_MAP (bool): Calculate Maximum APosterior (MAP) coordinates based on pdf scan?
        Returns:

            return_dict (dict): Dictionary of requested coverage and/or pdf scan values.
        """
        pass
   

#### Experimental functions
#### Some of these functions generalize existing functions and will replace them in future release.

    def entropy(self, 
                sub_manifolds=[-1], 
                conditional_input=None,
                force_embedding_coordinates=True, 
                force_intrinsic_coordinates=False,
                samplesize=100,
                failsafe_crosscheck_tolerance=None,
                dtype=None,
                device=None):

        """
        Calculates entropy of the PDF.
    
        Parameters:
            sub_manifolds (list(int)): Contains indices of sub-manifolds if entropy should be calculated for marginal PDF of the given sub-manifold. *-1* stands for the total PDF and is the default.
            conditional_input (Tensor/list(Tensor)/None): If passed defines the input to the PDF.
            force_embedding_coordinates (bool): Forces embedding coordinates in entropy calculation. Should always be true for correct manifold entropies.
            force_intrinsic_coordinates (bool): Forces intrinsic coordinates in entropy calculation. Should always be false for correct manifold entropies.
            samplesize (int): Samplesize to use for entropy approximation.
            failsafe_crosscheck_tolerance (float / None): If set, is used to crosscheck forward/bakckward pass compatability and resample if necessary. Has been introduced for the v flow in particular, so it should not be necessary for other flows.
            dtype (torch dtype): If given, uses this dtype. Otherwise uses dtype from parameters.
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.

        Returns:
            dict
                Dictionary containing entropy for each index defined in parameter *sub_manifolds*. If *-1* was given in *sub_manifolds*, the resulting entropy is stored under the *total* key.

        """
        pass

    def entropy_iterative(self, 
                sub_manifolds=[-1], 
                conditional_input=None,
                force_embedding_coordinates=True, 
                force_intrinsic_coordinates=False,
                samplesize=100,
                iterative_samplesize=10,
                max_iterative_batchsize=20,
                failsafe_crosscheck_tolerance=None,
                dtype=None,
                device=None,
                return_samples=False,
                verbose=False):

        """
        Calculates entropy of the PDF in an iterative manner. By iterating potentially both over target samples of later sub-pdfs, and over batch items, memory is saved
        and larger samplesizes (> a few 100 - 1000s) can be calculated. Probably only necessary for large sample sizes of later sub-pdfs, not of the first sub-pdf or the total PDF.
    
        Parameters:
            sub_manifolds (list(int)): Contains indices of sub-manifolds if entropy should be calculated for marginal PDF of the given sub-manifold. *-1* stands for the total PDF and is the default.
            conditional_input (Tensor/list(Tensor)/None): If passed defines the input to the PDF.
            force_embedding_coordinates (bool): Forces embedding coordinates in entropy calculation. Should always be true for correct manifold entropies.
            force_intrinsic_coordinates (bool): Forces intrinsic coordinates in entropy calculation. Should always be false for correct manifold entropies.
            samplesize (int): Samplesize to use for entropy approximation.
            iterative_samplesize (int): Number of target PDF samples evaluated simultaneously. Must be a divisor of samplesize.
            max_iterative_batchsize (int): The max number of batch samples evaluated simultaneously. 
            failsafe_crosscheck_tolerance (float / None): If set, is used to crosscheck forward/bakckward pass compatability and resample if necessary. Has been introduced for the v flow in particular, so it should not be necessary for other flows.
            dtype (torch dtype): If given, uses this dtype. Otherwise uses dtype from parameters.
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.
            return_samples (bool): Return the samples that are generated to calculate the entropy? Samples are returned as B*num_samples X sample_dim, so the effective batch dimension is B*num_samples.
            verbose (bool): Adds some extra prints if given.

        Returns:
            dict
                Dictionary containing entropy for each index defined in parameter *sub_manifolds*. If *-1* was given in *sub_manifolds*, the resulting entropy is stored under the *total* key.
            
            Tensor
                Only returned if *return_samples* is set to True. A tensor that contains the generated samples used to calculate the entropy.
        """
        pass

    def all_layer_inverse_individual_subdims(self, 
                                             x, 
                                             data_summary, 
                                             amortization_parameters=None, 
                                             force_embedding_coordinates=False, 
                                             force_intrinsic_coordinates=False,
                                             sub_manifolds=[-1]):


        ## set maximum iter to last sub dimension
        pass

    def all_layer_forward_individual_subdims_incl_sampling(self, 
                                       data_summary,
                                       total_samplesize,
                                       failsafe_crosscheck_tolerance=None,
                                       force_embedding_coordinates=False, 
                                       force_intrinsic_coordinates=False,
                                       amortization_parameters=None,
                                       dtype=None,
                                       device=None
                                       ):

        std_normal_samples = torch.randn(size=(total_samplesize, self.total_base_dim), dtype=dtype, device=device)
        
        ## save the easy cases in dict
        base_evals_dict=dict()

        ## get all manfiolds + total
        sub_manifolds=list(range(len(self.pdf_defs_list)))+[-1]
      
        for mf_dim in sub_manifolds:
           
            if(mf_dim==-1):
            
                this_mask=slice(0, self.total_base_dim)

            else:

                this_mask=slice(self.base_dim_indices[mf_dim][0], self.base_dim_indices[mf_dim][1])
        
           
            log_gauss_evals=torch.distributions.Normal(0.0,1.0).log_prob(std_normal_samples[:, this_mask]).sum(dim=-1)
                
            if(mf_dim==-1):
                base_evals_dict["total"]=log_gauss_evals
            else:
                base_evals_dict[mf_dim]=log_gauss_evals



        new_targets, logdet_per_manifold=self.all_layer_forward_individual_subdims(std_normal_samples, 
                                                  data_summary, 
                                                  force_embedding_coordinates=force_embedding_coordinates, 
                                                  force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                  sub_manifolds=sub_manifolds,
                                                  amortization_parameters=amortization_parameters)

 
        return_log_pdf=dict()
        for k in logdet_per_manifold:
            return_log_pdf[k]=base_evals_dict[k]-logdet_per_manifold[k]

        if(failsafe_crosscheck_tolerance):
            new_targets_prop, std_normal_samples_prop, return_log_pdf_prop, base_evals_dict_prop=recheck_sampling(self, 
                      new_targets,
                      std_normal_samples,
                      return_log_pdf,
                      base_evals_dict,
                      sub_manifolds=sub_manifolds,
                      failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                      conditional_input=data_summary,
                      amortization_parameters=amortization_parameters,
                      force_embedding_coordinates=force_embedding_coordinates,
                      force_intrinsic_coordinates=force_intrinsic_coordinates,
                      dtype=dtype,
                      device=device)

            if(new_targets_prop is not None):
                new_targets=new_targets_prop
                std_normal_samples=std_normal_samples_prop
                return_log_pdf=return_log_pdf_prop
                base_evals_dict=base_evals_dict_prop


        return new_targets, std_normal_samples, return_log_pdf, base_evals_dict


    def all_layer_forward_individual_subdims(self, 
                                       x,
                                       data_summary,
                                       force_embedding_coordinates=False, 
                                       force_intrinsic_coordinates=False,
                                       sub_manifolds=[-1],
                                       amortization_parameters=None
                                       ):

            

            total_batch_size=x.shape[0]
            if(force_embedding_coordinates):
                assert(force_intrinsic_coordinates!=force_embedding_coordinates), "Embedding and intrinsic coordinates can not be used at the same time!"

            ## set maximum iter to last sub dimension
            max_iter=0

            ## make sure the settings are self consistent
            for subdim in sub_manifolds:
                if(subdim!=-1):
                    
                    assert(subdim>=0 and subdim < len(self.layer_list))

                    if(subdim>max_iter):
                        max_iter=subdim
            
            ## only go as far as required
            if(-1 in sub_manifolds):
                max_iter=len(self.layer_list)-1

            
            extra_conditional_input=[]
            new_targets=[]
            logdet_per_manifold=dict()
            tot_log_det=0.0

            if(amortization_parameters is not None):

                raise Exception("Currently only supported without full amortization")
                #assert(amortization_parameters.shape[0]==x.shape[0]), ("batch size of x must agree with batch size of amortization_parameters")
                assert(amortization_parameters.shape[1]==self.total_number_amortizable_params)

            amort_param_counter=0

            for pdf_index, pdf_layers in enumerate(self.layer_list):

                this_pdf_type=self.pdf_defs_list[pdf_index]

                extra_params = None
                if(data_summary is not None and self.mlp_predictors[pdf_index] is not None):
                    
                    if(type(data_summary)==list):
                        this_data_summary=data_summary[pdf_index]
                    else:
                        this_data_summary=data_summary
                    if(len(extra_conditional_input)>0):
                        this_data_summary=torch.cat([this_data_summary]+extra_conditional_input, dim=1)

                    if(amortization_parameters is not None):
                        num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                        extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                        amort_param_counter+=num_amortization_params

                    else:
                        extra_params=self.mlp_predictors[pdf_index](this_data_summary)
                   

                else:

                    if(self.mlp_predictors[pdf_index] is not None):
                        if(len(extra_conditional_input)>0):
                            this_data_summary=torch.cat(extra_conditional_input, dim=1)
                            
                            if(amortization_parameters is not None):
                                num_amortization_params=self.mlp_predictors[pdf_index].num_amortization_params

                                extra_params=self.mlp_predictors[pdf_index](this_data_summary, extra_inputs=amortization_parameters[:,amort_param_counter:amort_param_counter+num_amortization_params])
                                amort_param_counter+=num_amortization_params

                            else:
                                extra_params=self.mlp_predictors[pdf_index](this_data_summary)

                        else:
                            raise Exception("SAMPLE: extra conditional input is empty but required for encoding!")

                    else:
                        ## we amortize everything with amortization_parameters .. including the first layer if there is no encoder
                        if(self.amortize_everything and pdf_index ==0):
                            assert(amortization_parameters is not None)
                            assert(amort_param_counter==0)

                            tot_num_params=0
                            for l in self.layer_list[0]:
                                tot_num_params+=l.get_total_param_num()

                            if(self.predict_log_normalization and self.join_poisson_and_pdf_description):
                                tot_num_params+=1

                            extra_params=amortization_parameters[:,:tot_num_params]

                            amort_param_counter+=tot_num_params


         
                if(self.predict_log_normalization):

                    if(pdf_index==0 and self.join_poisson_and_pdf_description):
                        extra_params=extra_params[:,:-1]

                this_target=x[:,self.base_dim_indices[pdf_index][0]:self.base_dim_indices[pdf_index][1]]

                ## loop through all layers in each pdf and transform "this_target"
                
                extra_param_counter = 0

                ## holds either the marginal
                logdet_this_manifold = 0.0

                ## default all layer foward
                for l, layer in list(enumerate(pdf_layers)):
                   
                    this_extra_params = None
                    
                    if extra_params is not None:
                        
                        this_extra_params = extra_params[:, extra_param_counter : extra_param_counter + layer.total_param_num]
                  
                    this_target, this_log_det = layer.flow_mapping([this_target, 0.0], extra_inputs=this_extra_params)
                    
                    extra_param_counter += layer.total_param_num

                    logdet_this_manifold = logdet_this_manifold+this_log_det

                # default joint logdet
                if(-1 in sub_manifolds):
                    tot_log_det=tot_log_det+logdet_this_manifold

                ## save logdet factors for this subdimension in dict
                logdet_per_manifold[pdf_index]=logdet_this_manifold

                ## next one
                new_targets.append(this_target)

                prev_target=this_target
               
                prev_target=self.layer_list[pdf_index][-1]._embedding_conditional_return(prev_target)

                extra_conditional_input.append(prev_target)

                if(max_iter==pdf_index):
                    break

            # we want to the total logdet also
            if(-1 in sub_manifolds):
                logdet_per_manifold["total"]=tot_log_det

            if (torch.isfinite(x) == 0).sum() > 0:
                raise Exception("nonfinite samples generated .. this should never happen!")

            res=torch.cat(new_targets, dim=1)

            if(res.shape[1]<x.shape[1]):
                #artificially increase shape
                res=torch.cat([res, torch.zeros( (int(x.shape[0]), int(x.shape[1]-res.shape[1])), dtype=x.dtype, device=x.device)],dim=1)

            ## transform to desired output space 
            if(force_embedding_coordinates):

                res, embedding_log_dets=self.transform_target_space_individual_subdims(res, transform_from="default", transform_to="embedding")
                
                tot_remaining_dim=0
                for pdf_index in range(max_iter+1):

                    tot_remaining_dim+=self.target_dims_embedded[pdf_index]

                ## shorten res if necessary
                res=res[:,:tot_remaining_dim]

                ## copy logdet correction over 
                for k in logdet_per_manifold.keys():
                    logdet_per_manifold[k]=logdet_per_manifold[k]+embedding_log_dets[k]

            elif(force_intrinsic_coordinates):
                assert(x.shape[1]==self.total_target_dim_intrinsic)
               
                res, embedding_log_dets=self.transform_target_space_individual_subdims(res, transform_from="default", transform_to="intrinsic")
                
                tot_remaining_dim=0
                for pdf_index in range(max_iter+1):

                    tot_remaining_dim+=self.target_dims_intrinsic[pdf_index]
                    
                ## shorten res if necessary
                res=res[:,:tot_remaining_dim]

                ## copy logdet correction over 
                for k in logdet_per_manifold.keys():
                    logdet_per_manifold[k]=logdet_per_manifold[k]+embedding_log_dets[k]

            """
            ### 
            if(failsafe_crosscheck_tolerance):

                ## inverse and another forward

                raise Exception()
                
                with torch.no_grad():
                    new_base, bw_logdet_dict=self.all_layer_inverse_individual_subdims(res, 
                                                 data_summary, 
                                                 amortization_parameters=amortization_parameters, 
                                                 force_embedding_coordinates=force_embedding_coordinates, 
                                                 force_intrinsic_coordinates=force_intrinsic_coordinates,
                                                 sub_manifolds=sub_manifolds)

                    ## and another forward

                   second_res, fw_logdet_dict=self.all_layer_forward_individual_subdims(
                                       new_base,
                                       data_summary,
                                       failsafe_crosscheck_tolerance=failsafe_crosscheck_tolerance,
                                       force_embedding_coordinates=force_embedding_coordinates, 
                                       force_intrinsic_coordinates=force_intrinsic_coordinates,
                                       sub_manifolds=sub_manifolds,
                                       amortization_parameters=amortization_parameters
                                       )


            """




            return res, logdet_per_manifold

    def transform_target_space_individual_subdims(self, target, transform_from="default", transform_to="embedding"):
      

        new_target=target

        if(len(target.shape)==1):
            new_target=target.unsqueeze(0)

        ## transforming only makes sense if input has correct target shape
        if(transform_from=="default"):
            assert(new_target.shape[1]==self.total_target_dim)
        elif(transform_from=="intrinsic"):
            assert(new_target.shape[1]==self.total_target_dim_intrinsic)
        elif(transform_from=="embedding"):
            assert(new_target.shape[1]==self.total_target_dim_embedded) 

        potentially_transformed_vals=[]

        index=0

        individual_logdets=dict()
        tot_logdet=0.0

        for pdf_index, pdf_type in enumerate(self.pdf_defs_list):

            if(transform_from=="default"):
                this_dim=self.target_dims[pdf_index]
            elif(transform_from=="intrinsic"):
                this_dim=self.target_dims_intrinsic[pdf_index]
            elif(transform_from=="embedding"):
                this_dim=self.target_dims_embedded[pdf_index]
            
            this_target, this_log_det=self.layer_list[pdf_index][-1].transform_target_space(new_target[:,index:index+this_dim], log_det=0, transform_from=transform_from, transform_to=transform_to)
            
            potentially_transformed_vals.append(this_target)

            index+=this_dim

            individual_logdets[pdf_index]=this_log_det

            tot_logdet=tot_logdet+this_log_det

        individual_logdets["total"]=tot_logdet

        potentially_transformed_vals=torch.cat(potentially_transformed_vals, dim=1)

        if(transform_to=="default"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim)
        elif(transform_to=="intrinsic"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim_intrinsic), (potentially_transformed_vals.shape, self.total_target_dim_intrinsic)
        elif(transform_to=="embedding"):
            assert(potentially_transformed_vals.shape[1]==self.total_target_dim_embedded), (new_target.shape[1], self.total_target_dim_embedded)  


        if(len(target.shape)==1):
            potentially_transformed_vals=potentially_transformed_vals.squeeze(0)

        return potentially_transformed_vals, individual_logdets

    def obtain_current_dtype_n_device(self):

        ## peek into first parameter vector
        try:
            first = next(self.parameters())
        except StopIteration:
            first = None

        if(first is None):
            ## model contains no parameters
            return None, None

        else:
            return first.dtype, first.device

    def marginal_moments(self, 
                         conditional_input=None, 
                         samplesize=50, 
                         iterative_samplesize=10, 
                         max_iterative_batchsize=20,
                         mises_abs_precision=1e-7, 
                         calc_kl_diff_and_entropic_quantities=False,
                         failsafe_crosscheck_tolerance=None,
                         dtype=None,
                         device=None,
                         verbose=False,
                         s2_entropy_scanning=False,
                         s2_entropy_scan_nside=32):
        """
        Calculate the first and second central moments of the marginal distributions. For Euclidean manifolds it calculates a Gaussian approximation, for spherical distributions calculates
        a von-Mises approximation. Because these are the respective maximum entropy distributions, their entropy should always be larger than the original distribution.
        We can also calculate the kl divergence and cross entropy between the exact distribution and its approximation, by switching on the *calc_kl_diff_and_entropic_quantities* flag.

        Parameters:
            conditional_input (Tensor/list(Tensor)/None): If passed defines the input to the PDF.
            samplesize (int): Samplesize to use per event for moment approximation.
            iterative_samplesize (int): Number of target PDF samples evaluated simultaneously. Must be a divisor of samplesize.
            max_iterative_batchsize (int): The max number of batch samples evaluated simultaneously. 
            mises_abs_precision (float): The absolute precision used break out the loop for the mises distribution second moment estimation.
            calc_kl_diff_and_entropic_quantities (bool): Flag that determines if we also calculate the KL divergence between the respective marginal distribution and its respective 2nd-order approximation. Also includes the cross entropy.
            failsafe_crosscheck_tolerance (float / None): If set, is used to crosscheck forward/bakckward pass compatability and resample if necessary. Has been introduced for the v flow in particular, so it should not be necessary for other flows.
            dtype (torch dtype): If given, uses this dtype. Otherwise uses dtype from parameters.
            device (torch.device): If given, uses this device. Otherwise uses device from parameters.
            verbose (bool): Some extra print statements on runtime.
            s2_entropy_scanning (bool): Use a healpix scan to determine entropy .. can be faster for certain s2 distributions.

        Returns:

            dict
                Dictionary containing moments and potentially entropies and KL-divergence between approximation and marginal distributions for each marginal distribution.
                The index indicates which marginal distribution. Means on the sphere are given in embedding coordinates and in angle coordinates. For example:

                mean_0 = mean of first marginal in embedding coordinates
                mean_0_angles = mean of first marginal in angle coordinates
                varlike_0 = covariance (Gaussian/Euclidean) or concentration parameter (Fisher-von Mises/spherical)
                entropy_0 = entropy of first marginal distribution
                kl_diff_exact_approx_0 = KL divergence between exact distribution and 2nd-order approximation
                cross_entropy_0 = cross entropy between exact distribution and 2nd-order approximation
                ...
                mean_1 = mean of second marginal
                ...
                ...
                entropy_total = total entropy (only total quantity calculated because it is essentially free)

           
        """
        pass

    