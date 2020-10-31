### Imports ####
#from __future__ import print_function

from tornado.ioloop import IOLoop
from .traits import HasTraits, Int, Str
from jigna.web_app import WebApp
#from jigna.template import Template
from jigna.vue_template import VueTemplate
import tornado.web
import PYME.resources

#import jigna.web_server
# monkey-patch the js file location
# The reason we do this is so we can use a more recent version of vuejs. By default, jigna ships with a jigna-vue.js
# which contains a concatenation of jquery, vue=1.x, and jigna. In recipe_editor.html, we load a modified version of
# jigna-vue.js, jigna-vue-bare WITHOUT vue which lets us separately load a recent version of vue
#jigna.web_server.JIGNA_JS_FILE='/Users/david/dev/jigna/jigna/js/dist/jigna.js'

import sys
import os
from PYME.recipes import modules #force modules to load
from PYME.recipes.base import ModuleCollection

#### Domain model ####

class Person(HasTraits):
    name = Str
    age  = Int

#### UI layer ####

# body_html = """
#     <div>
#       Execute on invalidation: <input ng-model="recipe.execute_on_invalidation" type='checkbox'><br>
#
#       <div innerHTML="raw recipe.to_svg()"></div>
#       <div id='img'></div>
#
#       <div ng-repeat="module in recipe.modules">
#             <h4>{{module.get_name()}}</h4>
#             <table>
#             <tr ng-repeat="param in module.get_params()[2]">
#             <td>{{param}}:</td><td><input ng-model="module[param]"></td>
#             </tr>
#             </table>
#       </div>
#
#     </div>
# """

#template = Template(body_html=body_html)

template = VueTemplate(html_file=os.path.join(os.path.dirname(__file__), 'recipe_editor.html'))

#### Entry point ####

def main():
    # Start the tornado ioloop application
    ioloop = IOLoop.instance()
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'r') as f:
            recipe_yaml = f.read()
    else:
        recipe_yaml = ''
        
    recipe = ModuleCollection.fromYAML(recipe_yaml)
    
    print(recipe)

    # Instantiate the domain model
    fred = Person(name='Fred', age=42)

    # Create a web app serving the view with the domain model added to its
    # context.
    app = WebApp(template=template, context={'recipe': recipe}, handlers=[(r'/static/(.*)', tornado.web.StaticFileHandler, {'path': PYME.resources.get_web_static_dir()}),])
    app.listen(8000)

    # Start serving the web app on port 8000.
    #
    # Point your web browser to http://localhost:8000/ to connect to this jigna
    # web app. Any operation performed on the client directly update the
    # model attributes on the server.
    print('Serving on port 8000...')
    ioloop.start()

if __name__ == "__main__":
    main()