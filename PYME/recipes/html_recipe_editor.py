### Imports ####
#from __future__ import print_function

from tornado.ioloop import IOLoop
from .traits import HasTraits, Int, Str
from jigna.web_app import WebApp
from jigna.template import Template

import sys
from PYME.recipes import modules #force modules to load
from PYME.recipes.base import ModuleCollection

#### Domain model ####

class Person(HasTraits):
    name = Str
    age  = Int

#### UI layer ####

body_html = """
    <div>
      Execute on invalidation: <input ng-model="recipe.execute_on_invalidation" type='checkbox'><br>
      
      <div innerHTML="raw recipe.to_svg()"></div>
      <div id='img'></div>
      
      <div ng-repeat="module in recipe.modules">
            <h4>{{module.get_name()}}</h4>
            <table>
            <tr ng-repeat="param in module.get_params()[2]">
            <td>{{param}}:</td><td><input ng-model="module[param]"></td>
            </tr>
            </table>
      </div>
      
    </div>
"""

template = Template(body_html=body_html)

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
    app = WebApp(template=template, context={'recipe': recipe})
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