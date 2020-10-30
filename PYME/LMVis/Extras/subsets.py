def time_block_subset(pipeline):
    from PYME.recipes import localisations
    
    recipe = pipeline.recipe
    
    clumper = localisations.TimeBlocks(recipe, input=pipeline.selectedDataSourceKey,
                                             output='time_blocked')
    if clumper.configure_traits(kind='modal'):
        recipe.add_modules_and_execute([clumper,])
        
        pipeline.selectDataSource(clumper.output)
        
        
def Plug(vis_fr):
    vis_fr.AddMenuItem('Extras', 'Split by time blocks for FRC', lambda e: time_block_subset(vis_fr.pipeline))