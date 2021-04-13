
condense=function(y){
    var f = Math.floor(y);
    var g = y-f;

    return 5/8*(f/2) + g/4;
};

const zip = (arr2, arr1) => arr1.map((k, i) => [k, condense(arr2[i])]);
const input_type_mappings = {string : 'text', number : 'number', boolean : 'checkbox'};

var vm = undefined;
// jigna.models are ready only when the deferred called ready is done.
Vue.config.devtools = true;
//Vue.config.debug = true;

const cols = ['turquoise', 'emerald', 'peter-river', 'amethyst', 'wet-asphalt', 'sunflower', 'carrot', 'alizarin'];



Vue.component('trait-input',{
   props: ['value', 'type'],
   template: `
        <input :value="value" :checked="value" :type="input_type(type)" v-on:input="$emit('input', format_value($event))">`,
   methods: {
       input_type: function(type){return input_type_mappings[type];},
       format_value: function(event){
           var val = event.target.value;
           switch(this.type){
               case 'number':
                   return parseFloat(val);
               case 'boolean':
                   val = event.target.checked;
                   console.log(val, typeof(val));
                   return val;
               default:
                   return val;
           }
       }
   }

});

Vue.component('rec-module',{
   props: ['module'],
   template: `
    <div class="container">
            <div class="recipe_module">
                <h4 class="module_name">{{module.get_name()}}</h4>
            <table>
            <tr v-for="param in module.get_params()[2]">
                <td>{{param}}:</td><td><trait-input v-model="module[param]" :type="typeof(module[param])"></trait-input></td>
            </tr>
            </table>
            </div>
                </div>
        `,
   methods: {

   }

});


var editor;


jigna.ready.done(function() {
   vm = new Vue({
       el: '#app',
       data: jigna.models, // Attach to the body tag.
       methods: {
           threaded: function(obj, method_name, args) {
               jigna.threaded.apply(jigna, arguments);
           },
           line_fmt: function (line) {
               return zip(line[0], line[1]).join(' ');

           } ,
           input_type: function (param, module) {
               // take module argument for an improved future version which gets more trait info
               return input_type_mappings[typeof(param)];
           },
           cond: function(y){return condense(y);},
           get_colour: function(index) {return cols[index % 8]},
       },
       watch: {
           recipe: function(new_data, old_data){
               console.log('Model changed');
           }
       }
   });

   ace.config.set("basePath", "https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/");
   editor = ace.edit("editor");
   editor.setTheme("ace/theme/monokai");
   editor.session.setMode("ace/mode/yaml");
   editor.setValue(vm.recipe.toYAML());
   editor.clearSelection();
   editor.on('blur', function(){vm.recipe.update_from_yaml(editor.getValue());});

   jigna.add_listener('jigna', 'object_changed', function(event){editor.setValue(vm.recipe.toYAML());editor.clearSelection();});
});

load_recipe = function(){
    console.log('clicked load recipe');
    select_file(function(path){
     console.log('Selected: ', path);
     vm.recipe.update_from_file('PYME-CLUSTER:///' + path)
    }, 'Select a file');
};

save_recipe = function(){
    console.log('clicked save recipe');
    select_file_save(function(path){
     console.log('Selected: ', path);
     vm.recipe.save_yaml('PYME-CLUSTER:///' + path)
    }, 'Save as', '', 'RECIPES/');
};