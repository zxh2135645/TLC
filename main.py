from .tlc create_tlc_workflow
        
"""
An example for using TLC. With one configuration options, input image from T1 weighted and freesurfer parcellation masks,
as well as the editted binary lesion mask based on FLAIR image. 

Inputs::
    config: Dictionary with configuration options. You can create your own
    t1_file: full path of t1 image
    freesurf_parc: full path of aparc+aseg.mgz from freesurfer
    flair_lesion: editted binary lesion mask based on FLAIR image (can also be labeled)
    
Outputs::
   nipype.pipeline.engine.Workflow object
   
"""
        
wf = create_tlc_workflow(config,
                         t1,
                         freesurf_parc,
                         flair_lesion
                        )
wf.config = {"execution": {"crashdump_dir": os.path.join(config["crash_directory"], t1_filename + self.flag)}}
wf.run(plugin=self.inputs.plugin,
       plugin_args=self.inputs.plugin_args)
