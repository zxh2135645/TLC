Copyright 2018 Xinheng Zhang

__author__ = 'sf713420'

def get_config():
    """
    ===========================
    Configuration for Workflows
    ===========================
    plugin: "Linear", "SGE" for grid
    plugin_args: qsub args
    outputsink_directory: where to store results
    working_directory: where to run things
    """
    config = dict()

    #config["plugin"] = "Linear"#"SGE"
    #config["plugin_args"] = {}#{"qsub_args":"-q ms.q -l arch=lx24-amd64 -l h_stack=32M \
    #-l h_vmem=4G -l hostname=graf -v MKL_NUM_THREADS=1"}

    config["working_directory"] = "/working/henry_temp/keshavan/"
    config["output_directory"] = "/data/henry7/PBR/subjects"
    config["crash_directory"] = "/working/henry_temp/keshavan/crashes"
    config["mni_template"] = "/data/henry6/PBR/templates/OASIS-30_Atropos_template_in_MNI152.nii.gz"

    return config

def get_msid(name):
    msid = name.split("/")[-1].split('-')[0]
    return msid

def get_mseid(str1):
    mseid1 = str1.split("/")[-1].split('-')[1]
    # mseid2 = str2.split("/")[-1].split('-')[1]
    return mseid1

def mri_convert_like(t1_image, reorient_mask, working_dir):
    from subprocess import check_call
    import os
    from nipype.utils.filemanip import split_filename
    _, reorient_file, _ = split_filename(reorient_mask)
    output_file = os.path.join(working_dir, reorient_file + '_like.nii.gz')
    cmd = ["mri_convert", "--like", t1_image, reorient_mask, output_file]
    check_call(cmd)
    return output_file


def matrix_operation(t1_image, ventricle, cortex, flair_lesion, white_matter, working_dir):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import label
    from scipy.ndimage.morphology import binary_closing
    import os

    img = nib.load(t1_image)
    data, aff = img.get_data(), img.affine

    img2 = nib.load(ventricle)
    data2 = img2.get_data()

    img3 = nib.load(cortex)
    data3 = img3.get_data()

    img4 = nib.load(white_matter)
    data4 = img4.get_data()

    img5 = nib.load(flair_lesion)
    data5 = img5.get_data()

    csf = np.where(data2 >= 0.8)
    gray_matter = np.where(data3 >= 0.8)

    #ventricle_median = np.median(data[csf])
    ventricle_median = 0
    gray_matter85 = np.percentile(data[gray_matter], 85)
    gray_matter90 = np.percentile(data[gray_matter], 90)
    gray_matter95 = np.percentile(data[gray_matter], 95)
    gray_matter100 = np.percentile(data[gray_matter], 100)
    lesions85 = np.where(np.logical_and(data >= ventricle_median, data <= gray_matter85))
    lesions90 = np.where(np.logical_and(data >= ventricle_median, data <= gray_matter90))
    lesions95 = np.where(np.logical_and(data >= ventricle_median, data <= gray_matter95))
    lesions100 = np.where(np.logical_and(data >= ventricle_median, data <= gray_matter100))
    lesion_mask85 = np.zeros(data.shape)
    lesion_mask90 = np.zeros(data.shape)
    lesion_mask95 = np.zeros(data.shape)
    lesion_mask100 = np.zeros(data.shape)
    lesion_mask85[lesions85] = 1
    lesion_mask90[lesions90] = 1
    lesion_mask95[lesions95] = 1
    lesion_mask100[lesions100] = 1

    combined = np.where(np.logical_or(data4 >= 0.5, data5)) # Is this a good number to try?
    combined_mask = np.zeros(data.shape)
    combined_mask[combined] = 1

    true_lesions85 = np.where(np.logical_and(lesion_mask85, combined_mask))
    true_lesion_mask85 = np.zeros(data.shape)
    true_lesion_mask85[true_lesions85] = 1
    true_lesions90 = np.where(np.logical_and(lesion_mask90, combined_mask))
    true_lesion_mask90 = np.zeros(data.shape)
    true_lesion_mask90[true_lesions90] = 1
    true_lesions95 = np.where(np.logical_and(lesion_mask95, combined_mask))
    true_lesion_mask95 = np.zeros(data.shape)
    true_lesion_mask95[true_lesions95] = 1
    true_lesions100 = np.where(np.logical_and(lesion_mask100, combined_mask))
    true_lesion_mask100 = np.zeros(data.shape)
    true_lesion_mask100[true_lesions100] = 1

    subtraction85 = np.subtract(true_lesion_mask85, data5)
    subtraction90 = np.subtract(true_lesion_mask90, data5)
    subtraction95 = np.subtract(true_lesion_mask95, data5)
    subtraction100 = np.subtract(true_lesion_mask100, data5)

    labeled_img85, nlabels85 = label(true_lesion_mask85 > 0)
    labeled_img90, nlabels90 = label(true_lesion_mask90 > 0)
    labeled_img95, nlabels95 = label(true_lesion_mask95 > 0)
    labeled_img100, nlabels100 = label(true_lesion_mask100 > 0)


    # 85 percentile
    new_lesion85 = np.zeros(data.shape)
    lesion_size85 = np.bincount(labeled_img85.ravel())
    max_label85 = max(labeled_img85.ravel())

    for i in range(1, max_label85):
        coord = np.where(labeled_img85 == i)
        cluster_sum = int(np.sum(subtraction85[coord]))
        if lesion_size85[i] > cluster_sum:
            new_lesion85[coord] = 1

    # Remove larg non-lesion masks
    labeled_img285, nlabels285 = label(new_lesion85)
    lesion_size285 = np.bincount(labeled_img285.ravel())
    print("lesion size of 85 before filtering out large lesions are", lesion_size285,
          "\nAnd lesion counts for 85 are", len(lesion_size285) - 1)
    for i in range(1, nlabels285):
        coord = np.where(labeled_img285 == i)
        if lesion_size285[i] > 3000:
            labeled_img285[coord] = 0

    closing_img85 = np.zeros(data.shape)
    closing_img85[binary_closing(labeled_img285, structure=np.ones((2, 2, 2)))] = 1
    # using 2x2x2 kernel to close
    closing_img_out85 = nib.Nifti1Image(closing_img85, affine=aff)
    out_path85 = os.path.join(working_dir, 'new_lesion85.nii.gz')
    nib.save(closing_img_out85, out_path85)

    # repeat for 90 percentile
    new_lesion90 = np.zeros(data.shape)
    lesion_size90 = np.bincount(labeled_img90.ravel())
    max_label90 = max(labeled_img90.ravel())

    for i in range(1, max_label90):
        coord = np.where(labeled_img90 == i)
        cluster_sum = int(np.sum(subtraction90[coord]))
        if lesion_size90[i] > cluster_sum:
            new_lesion90[coord] = 1

    # Remove larg non-lesion masks
    labeled_img290, nlabels290 = label(new_lesion90)
    lesion_size290 = np.bincount(labeled_img290.ravel())
    print("lesion size of 90 before filtering out large lesions are", lesion_size290,
          "\nAnd lesion counts for 90 are", len(lesion_size290) - 1)
    for i in range(1, nlabels290):
        coord = np.where(labeled_img290 == i)
        if lesion_size290[i] > 3000:
            labeled_img290[coord] = 0

    closing_img90 = np.zeros(data.shape)
    closing_img90[binary_closing(labeled_img290, structure=np.ones((2, 2, 2)))] = 1
    # using 2x2x2 kernel to close
    closing_img_out90 = nib.Nifti1Image(closing_img90, affine=aff)
    out_path90 = os.path.join(working_dir, 'new_lesion90.nii.gz')
    nib.save(closing_img_out90, out_path90)

    # repeat for 95 percentile
    new_lesion95 = np.zeros(data.shape)
    lesion_size95 = np.bincount(labeled_img95.ravel())
    max_label95 = max(labeled_img95.ravel())

    for i in range(1, max_label95):
        coord = np.where(labeled_img95 == i)
        cluster_sum = int(np.sum(subtraction95[coord]))
        if lesion_size95[i] > cluster_sum:
            new_lesion95[coord] = 1

    labeled_img295, nlabels295 = label(new_lesion95)
    lesion_size295 = np.bincount(labeled_img295.ravel())
    for i in range(1, nlabels295):
        coord = np.where(labeled_img295 == i)
        #if lesion_size295[i] > 3000:
            #labeled_img295[coord] = 0

    closing_img95 = np.zeros(data.shape)
    closing_img95[binary_closing(labeled_img295, structure=np.ones((2, 2, 2)))] = 1
    closing_img_out95 = nib.Nifti1Image(closing_img95, affine=aff)
    out_path95 = os.path.join(working_dir, 'new_lesion95.nii.gz')
    nib.save(closing_img_out95, out_path95)

    # repeat for 100 percentile
    new_lesion100 = np.zeros(data.shape)
    lesion_size100 = np.bincount(labeled_img100.ravel())
    max_label100 = max(labeled_img100.ravel())

    for i in range(1, max_label100):
        coord = np.where(labeled_img100 == i)
        cluster_sum = int(np.sum(subtraction100[coord]))
        if lesion_size100[i] > cluster_sum:
            new_lesion100[coord] = 1

    # Remove larg non-lesion masks
    labeled_img300, nlabels300 = label(new_lesion100)
    lesion_size300 = np.bincount(labeled_img300.ravel())
    print("lesion size of 100 before filtering out large lesions are", lesion_size300,
          "\nAnd lesion counts for 100 are", len(lesion_size300) - 1)
    for i in range(1, nlabels300):
        coord = np.where(labeled_img300 == i)
        if lesion_size300[i] > 3000:
            labeled_img300[coord] = 0

    closing_img100 = np.zeros(data.shape)
    closing_img100[binary_closing(labeled_img300, structure=np.ones((2, 2, 2)))] = 1
    # using 2x2x2 kernel to close
    closing_img_out100 = nib.Nifti1Image(closing_img100, affine=aff)
    out_path100 = os.path.join(working_dir, 'new_lesion100.nii.gz')
    nib.save(closing_img_out100, out_path100)

    combined_mask_out = nib.Nifti1Image(combined_mask, affine=aff)
    out_path_combined = os.path.join(working_dir, 'combined_white_matter.nii.gz')

    return out_path85, out_path90, out_path95, out_path100, out_path_combined

def matrix_operation2(t1_image, ventricle, cortex, flair_lesion, white_matter, working_dir):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import label
    from scipy.ndimage.morphology import binary_closing
    import os

    img = nib.load(t1_image)
    data, aff = img.get_data(), img.affine

    img2 = nib.load(ventricle)
    data2 = img2.get_data()

    img3 = nib.load(cortex)
    data3 = img3.get_data()

    img4 = nib.load(white_matter)
    data4 = img4.get_data()

    img5 = nib.load(flair_lesion)
    data5 = img5.get_data()

    csf = np.where(data2 >= 0.8)
    gray_matter = np.where(data3 >= 0.8)

    #ventricle_median = np.median(data[csf])
    ventricle_median = 0
    gray_matter90 = np.percentile(data[gray_matter], 90)
    lesions90 = np.where(np.logical_and(data >= ventricle_median, data <= gray_matter90))
    lesion_mask90 = np.zeros(data.shape)
    lesion_mask90[lesions90] = 1

    gray_matter95 = np.percentile(data[gray_matter], 95)
    lesions95 = np.where(np.logical_and(data >= ventricle_median, data <= gray_matter95))
    lesion_mask95 = np.zeros(data.shape)
    lesion_mask95[lesions95] = 1

    gray_matter100 = np.percentile(data[gray_matter], 100)
    lesions100 = np.where(np.logical_and(data >= ventricle_median, data <= gray_matter100))
    lesion_mask100 = np.zeros(data.shape)
    lesion_mask100[lesions100] = 1

    combined = np.where(np.logical_or(data4 >= 0.5, data5))
    combined_mask = np.zeros(data.shape)
    combined_mask[combined] = 1

    # 90
    true_lesions90 = np.where(np.logical_and(lesion_mask90, combined_mask))
    true_lesion_mask90 = np.zeros(data.shape)
    true_lesion_mask90[true_lesions90] = 1

    intersection90 = np.where(np.logical_and(true_lesion_mask90, data5))
    intersection_mask90 = np.zeros(data.shape)
    intersection_mask90[intersection90] = 1


    labeled_img90, nlabels90 = label(intersection_mask90)
    lesion_size90 = np.bincount(labeled_img90.ravel())

    print("lesion size of 90 before filtering out large lesions are", lesion_size90,
          "\nAnd lesion counts for 90 are", len(lesion_size90) - 1)
    for i in range(1, nlabels90):
        coord = np.where(labeled_img90 == i)
        if lesion_size90[i] > 3000:
            labeled_img90[coord] = 0

    closing_img90 = np.zeros(data.shape)
    closing_img90[binary_closing(labeled_img90, structure=np.ones((2, 2, 2)))] = 1
    # using 2x2x2 kernel to close
    closing_img_out90 = nib.Nifti1Image(closing_img90, affine=aff)
    out_path90 = os.path.join(working_dir, 'new_lesion90.nii.gz')
    nib.save(closing_img_out90, out_path90)


    ### 95
    true_lesions95 = np.where(np.logical_and(lesion_mask95, combined_mask))
    true_lesion_mask95 = np.zeros(data.shape)
    true_lesion_mask95[true_lesions95] = 1

    intersection95 = np.where(np.logical_and(true_lesion_mask95, data5))
    intersection_mask95 = np.zeros(data.shape)
    intersection_mask95[intersection95] = 1


    labeled_img95, nlabels95 = label(intersection_mask95)
    lesion_size95 = np.bincount(labeled_img95.ravel())

    print("lesion size of 95 before filtering out large lesions are", lesion_size95,
          "\nAnd lesion counts for 95 are", len(lesion_size95) - 1)
    for i in range(1, nlabels95):
        coord = np.where(labeled_img95 == i)
        #if lesion_size95[i] > 3000:
            #labeled_img95[coord] = 0

    closing_img95 = np.zeros(data.shape)
    closing_img95[binary_closing(labeled_img95, structure=np.ones((2, 2, 2)))] = 1
    # using 2x2x2 kernel to close
    closing_img_out95 = nib.Nifti1Image(closing_img95, affine=aff)
    out_path95 = os.path.join(working_dir, 'new_lesion95.nii.gz')
    nib.save(closing_img_out95, out_path95)

    ### 100
    true_lesions100 = np.where(np.logical_and(lesion_mask100, combined_mask))
    true_lesion_mask100 = np.zeros(data.shape)
    true_lesion_mask100[true_lesions100] = 1

    intersection100 = np.where(np.logical_and(true_lesion_mask100, data5))
    intersection_mask100 = np.zeros(data.shape)
    intersection_mask100[intersection100] = 1


    labeled_img100, nlabels100 = label(intersection_mask100)
    lesion_size100 = np.bincount(labeled_img100.ravel())

    print("lesion size of 100 before filtering out large lesions are", lesion_size100,
          "\nAnd lesion counts for 100 are", len(lesion_size100) - 1)
    for i in range(1, nlabels100):
        coord = np.where(labeled_img100 == i)
        if lesion_size100[i] > 3000:
            labeled_img100[coord] = 0

    closing_img100 = np.zeros(data.shape)
    closing_img100[binary_closing(labeled_img100, structure=np.ones((2, 2, 2)))] = 1
    # using 2x2x2 kernel to close
    closing_img_out100 = nib.Nifti1Image(closing_img100, affine=aff)
    out_path100 = os.path.join(working_dir, 'new_lesion100.nii.gz')
    nib.save(closing_img_out100, out_path100)

    return out_path90, out_path95, out_path100

def create_tlc_workflow(config, t1_file, freesurf_parc, flair_lesion):
    """
    Inputs::
        config: Dictionary with PBR configuration options. See config.py
        t1_file: full path of t1 image
        freesurf_parc: full path of aparc+aseg.mgz from freesurfer
        flair_lesion: editted binary lesion mask based on FLAIR image (can also be labeled)
    Outputs::
        nipype.pipeline.engine.Workflow object
    """

    import nipype.interfaces.ants as ants
    from nipype.pipeline.engine import Node, Workflow, MapNode
    from nipype.interfaces.io import DataSink, DataGrabber
    from nipype.interfaces.utility import IdentityInterface, Function
    import nipype.interfaces.fsl as fsl
    from nipype.utils.filemanip import load_json
    import os
    import numpy as np
    from nipype.interfaces.freesurfer import Binarize, MRIConvert
    from nipype.interfaces.slicer.filtering import n4itkbiasfieldcorrection as n4
    from nipype.interfaces.fsl import Reorient2Std
    from nipype.interfaces.freesurfer import SegStats


    mse = get_mseid(t1_file)
    msid = get_msid(t1_file)
    working_dir = "tlc_{0}_{1}".format(msid, mse)

    register = Workflow(name=working_dir)
    register.base_dir = config["working_directory"]

    inputnode = Node(IdentityInterface(fields=["t1_image", "parc", "flair_lesion", "mse"]),
                     name="inputspec")
    inputnode.inputs.t1_image = t1_file
    inputnode.inputs.parc = freesurf_parc
    inputnode.inputs.flair_lesion = flair_lesion
    inputnode.inputs.mse = mse

    bin_math = Node(fsl.BinaryMaths(), name="Convert_to_binary")
    bin_math.inputs.operand_value = 1
    bin_math.inputs.operation = 'min'
    register.connect(inputnode, "flair_lesion", bin_math, "in_file")

    binvol1 = Node(Binarize(), name="binary_ventricle")
    binvol1.inputs.match = [4, 5, 11, 14, 15, 24, 43, 44, 50, 72, 213, 31, 63]
    #binvol1.inputs.match = [4, 5, 14, 15, 24, 43, 44, 72, 213]
    # every parcellation corresponds to ventricle CSF
    #binvol1.inputs.mask_thresh = 0.5
    binvol1.inputs.binary_file = os.path.join(config["working_directory"],
                                              working_dir, "binary_ventricle", "binarize_ventricle.nii.gz")
    register.connect(inputnode, "parc", binvol1, "in_file")

    binvol2 = Node(Binarize(), name="binary_gray_matter")
    binvol2.inputs.match = [3, 8, 42, 47, 169, 220, 702,
                            1878, 1915, 1979, 1993, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                            2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026,
                            2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035,
                            772, 833, 835, 896, 925, 936, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                            1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026,
                            1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035]
    binvol2.inputs.binary_file = os.path.join(config["working_directory"], working_dir,
                                              "binary_gray_matter", "binarize_cortex.nii.gz")
    #binvol2.inputs.mask_thresh = 0.5
    register.connect(inputnode, "parc", binvol2, "in_file")

    bias_corr = Node(n4.N4ITKBiasFieldCorrection(), name="BiasFieldCorrection")
    bias_corr.inputs.outputimage = os.path.join(config["working_directory"], working_dir,
                                                "BiasFieldCorrection", "bias_corrected.nii.gz")
    register.connect(inputnode, "t1_image", bias_corr, "inputimage")

    reo1 = Node(Reorient2Std(), name="reorient1")
    reo2 = Node(Reorient2Std(), name="reorient2")
    register.connect(binvol1, "binary_file", reo1, "in_file")
    register.connect(binvol2, "binary_file", reo2, "in_file")

    mri_convert1 = Node(Function(input_names=['t1_image', 'reorient_mask', 'working_dir'],
                                 output_names=['output_file'],
                                 function=mri_convert_like), name="mri_convert1")
    mri_convert2 = Node(Function(input_names=['t1_image', 'reorient_mask', 'working_dir'],
                                 output_names=['output_file'],
                                 function=mri_convert_like), name="mri_convert2")
    mri_convert1.inputs.working_dir = os.path.join(config["working_directory"], working_dir, 'mri_convert1')
    register.connect(bias_corr, "outputimage", mri_convert1, "t1_image")
    register.connect(reo1, "out_file", mri_convert1, "reorient_mask")
    mri_convert2.inputs.working_dir = os.path.join(config["working_directory"], working_dir, 'mri_convert2')
    register.connect(bias_corr, "outputimage", mri_convert2, "t1_image")
    register.connect(reo2, "out_file", mri_convert2, "reorient_mask")

    binvol3 = Node(Binarize(), name="binary_white_matter")
    binvol3.inputs.match = [2, 7, 16, 28, 41, 46, 60, 77, 78, 79, 251, 252, 253, 254, 255]
    #binvol3.inputs.match = [2, 7, 41, 46, 77, 78, 79]
    #binvol3.inputs.mask_thresh = 0.5
    binvol3.inputs.binary_file = os.path.join(config["working_directory"], working_dir,
                                              "binary_white_matter", "binarize_white_matter.nii.gz")
    register.connect(inputnode, "parc", binvol3, "in_file")
    reo3 = Node(Reorient2Std(), name="reorient3")
    register.connect(binvol3, "binary_file", reo3, "in_file")

    mri_convert3 = Node(Function(input_names=['t1_image', 'reorient_mask', 'working_dir'],
                                 output_names=['output_file'],
                                 function=mri_convert_like), name="mri_convert3")
    mri_convert3.inputs.working_dir = os.path.join(config["working_directory"], working_dir, 'mri_convert3')
    register.connect(reo3, "out_file", mri_convert3, "reorient_mask")
    register.connect(bias_corr, "outputimage", mri_convert3, "t1_image")

    get_new_lesion = Node(Function(input_names=['t1_image', 'ventricle', 'cortex', 'flair_lesion', 'white_matter',
                                                'working_dir'],
                                   output_names=['out_path85', 'out_path90', 'out_path95', 'out_path100', 'out_path_combined'],
                                   function=matrix_operation), name='get_new_lesion')
    get_new_lesion.inputs.working_dir = os.path.join(config["working_directory"], working_dir, 'get_new_lesion')
    register.connect(bias_corr, "outputimage", get_new_lesion, "t1_image")
    register.connect(mri_convert1, "output_file", get_new_lesion, "ventricle")
    register.connect(mri_convert2, "output_file", get_new_lesion, "cortex")
    register.connect(bin_math, "out_file", get_new_lesion, "flair_lesion")
    register.connect(mri_convert3, "output_file", get_new_lesion, "white_matter")


    cluster85 = Node(fsl.Cluster(threshold=0.0001,
                                 out_index_file = True,
                                 use_mm=True),
                     name="cluster85")
    register.connect(get_new_lesion, "out_path85", cluster85, "in_file")
    segstats85 = Node(SegStats(), name="segstats85")
    register.connect(cluster85, "index_file", segstats85, "segmentation_file")

    cluster90 = Node(fsl.Cluster(threshold=0.0001,
                                 out_index_file = True,
                                 use_mm=True),
                     name="cluster90")
    register.connect(get_new_lesion, "out_path90", cluster90, "in_file")
    segstats90 = Node(SegStats(), name="segstats90")
    register.connect(cluster90, "index_file", segstats90, "segmentation_file")

    cluster95 = Node(fsl.Cluster(threshold=0.0001,
                                 out_index_file = True,
                                 use_mm=True),
                     name="cluster95")
    register.connect(get_new_lesion, "out_path95", cluster95, "in_file")
    segstats95 = Node(SegStats(), name="segstats95")
    register.connect(cluster95, "index_file", segstats95, "segmentation_file")

    cluster100 = Node(fsl.Cluster(threshold=0.0001,
                                 out_index_file = True,
                                 use_mm=True),
                     name="cluster100")
    register.connect(get_new_lesion, "out_path100", cluster100, "in_file")
    segstats100 = Node(SegStats(), name="segstats100")
    register.connect(cluster100, "index_file", segstats100, "segmentation_file")

    get_new_lesion2 = Node(Function(input_names=['t1_image', 'ventricle', 'cortex', 'flair_lesion', 'white_matter',
                                                'working_dir'],
                                   output_names=['out_path90', 'out_path95', 'out_path100'],
                                   function=matrix_operation2), name='get_new_lesion2')
    get_new_lesion2.inputs.working_dir = os.path.join(config["working_directory"], working_dir, 'get_new_lesion2')
    register.connect(bias_corr, "outputimage", get_new_lesion2, "t1_image")
    register.connect(mri_convert1, "output_file", get_new_lesion2, "ventricle")
    register.connect(mri_convert2, "output_file", get_new_lesion2, "cortex")
    register.connect(bin_math, "out_file", get_new_lesion2, "flair_lesion")
    register.connect(mri_convert3, "output_file", get_new_lesion2, "white_matter")
    cluster_intersection90 = Node(fsl.Cluster(threshold=0.0001,
                                 out_index_file = True,
                                 use_mm=True),
                                 name="cluster_intersection90")
    register.connect(get_new_lesion2, "out_path90", cluster_intersection90, "in_file")
    segstats_intersection90 = Node(SegStats(), name="segstats_intersection90")
    register.connect(cluster_intersection90, "index_file", segstats_intersection90, "segmentation_file")

    cluster_intersection95 = Node(fsl.Cluster(threshold=0.0001,
                                 out_index_file = True,
                                 use_mm=True),
                                 name="cluster_intersection95")
    register.connect(get_new_lesion2, "out_path95", cluster_intersection95, "in_file")
    segstats_intersection95 = Node(SegStats(), name="segstats_intersection95")
    register.connect(cluster_intersection95, "index_file", segstats_intersection95, "segmentation_file")

    cluster_intersection100 = Node(fsl.Cluster(threshold=0.0001,
                                 out_index_file = True,
                                 use_mm=True),
                                 name="cluster_intersection100")
    register.connect(get_new_lesion2, "out_path100", cluster_intersection100, "in_file")
    segstats_intersection100 = Node(SegStats(), name="segstats_intersection100")
    register.connect(cluster_intersection100, "index_file", segstats_intersection100, "segmentation_file")

    sinker = Node(DataSink(), name="sinker")
    sinker.inputs.base_directory = os.path.join(config["output_directory"], mse, "tlc")
    sinker.inputs.container = '.'
    sinker.inputs.substitutions = []

    register.connect(get_new_lesion, "out_path85", sinker, "85.@lesion85")
    register.connect(get_new_lesion, "out_path90", sinker, "90.@lesion90")
    register.connect(get_new_lesion, "out_path95", sinker, "95.@lesion95")
    register.connect(get_new_lesion, "out_path100", sinker, "100.@lesion100")
    register.connect(get_new_lesion, "out_path_combined", sinker, "@WhiteMatterCombined")
    register.connect(get_new_lesion2, "out_path90", sinker, "intersection90.@lesion90")
    register.connect(get_new_lesion2, "out_path95", sinker, "intersection95.@lesion95")
    register.connect(get_new_lesion2, "out_path100", sinker, "intersection100.@lesion100")

    register.connect(segstats85, "summary_file", sinker, "85.@summaryfile85")
    register.connect(segstats90, "summary_file", sinker, "90.@summaryfile90")
    register.connect(segstats95, "summary_file", sinker, "95.@summaryfile95")
    register.connect(segstats100, "summary_file", sinker, "100.@summaryfile100")
    register.connect(segstats_intersection90, "summary_file", sinker, "intersection90.@summaryfile90")
    register.connect(segstats_intersection95, "summary_file", sinker, "intersection95.@summaryfile95")
    register.connect(segstats_intersection100, "summary_file", sinker, "intersection100.@summaryfile100")

    register.connect(cluster85, "index_file", sinker, "85.@index_file85")
    register.connect(cluster90, "index_file", sinker, "90.@index_file90")
    register.connect(cluster95, "index_file", sinker, "95.@index_file95")
    register.connect(cluster100, "index_file", sinker, "100.@index_file100")
    register.connect(cluster_intersection90, "index_file", sinker, "intersection90.@index_file90")
    register.connect(cluster_intersection95, "index_file", sinker, "intersection95.@index_file95")
    register.connect(cluster_intersection100, "index_file", sinker, "intersection100.@index_file100")

    return register
