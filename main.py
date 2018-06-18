        from .utils import get_msid, get_mseid, create_tlc_workflow
        
        wf = create_tlc_workflow(config,
                                 t1,
                                 freesurf_parc,
                                 flair_lesion
                                 )

        wf.config = {"execution": {"crashdump_dir": os.path.join(config["crash_directory"], t1_filename + self.flag)}}
        wf.run(plugin=self.inputs.plugin,
               plugin_args=self.inputs.plugin_args)
