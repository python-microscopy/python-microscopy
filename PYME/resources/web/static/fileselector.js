function select_file(callback, message='Select a file', basedir=''){
    $('#file_select_modal').modal('show');

    $('#file_select_label').html(message);

    var tree = $('#file_selector_tree').jstree({
      'core' : {
        'data' : {
          'url' : '/files/_lite/',
          'data' : function (node) {
              //console.log(node);
              try{ return { 'path' : node.data.path };}
              catch (err) { return {'path' : ''}}

          }
        }
      }
    });

    $('#file_select_ok').click(function(){
        $('#file_select_modal').modal('hide');
        $('#file_select_ok').off('click');
        var path = $('#file_selector_tree').jstree(true).get_selected(true)[0].data.path;
        callback(path);
    })

}


function select_file_save(callback, message='Save as', clusterfilter='', basedir=''){
    // NOTE: basedir MUST have a trailing / if not empty
    $('#file_select_modal').modal('show');

    $('#file_select_label').html(message);

    $('#file_selector_tree').html(`
    <div class="form-group">
        <label class="control-label" for="saveURL">Recipe</label>
        <div class="input-group">
            <span class="input-group-addon">pyme-cluster://${clusterfilter}/${basedir}</span>
            <input type="text" class="form-control" id="saveURL" name="saveURL">
        </div>
    </div>
    `);


    $('#file_select_ok').click(function(){
        $('#file_select_modal').modal('hide');
        $('#file_select_ok').off('click');
        var path = basedir + $('#saveURL').val();
        callback(path);
    })

}
