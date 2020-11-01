"""
A quick attempt to run both clusterUI and the recipe editor under tornado so that they can talk to each other.

NOTE/FIXME: This uses a single, global, recipe instance for recipe editing which means that things will go very wrong
if editing recipes in separate sessions. This is a major limitation which should be fixed..
"""

import os
import tornado.httpserver
import tornado.ioloop
import tornado.wsgi
from django.core.wsgi import get_wsgi_application
import PYME.resources
from tornado import web

from PYME.recipes import html_recipe_editor
from jigna import web_server as jignaws

from PYME.recipes.base import ModuleCollection

rec_text = '''
- processing.OpticalFlow:
    filterRadius: 0.5
    inputName: filtered_input
    outputNameX: flow_x
    outputNameY: flow_y
    regularizationLambda: 1.0
    supportRadius: 4.0
- filters.GaussianFilter:
    inputName: flow_x
    outputName: flow_xf
    processFramesIndividually: false
    sigmaZ: 5.0
- filters.GaussianFilter:
    inputName: flow_y
    outputName: flow_yf
    processFramesIndividually: false
    sigmaZ: 5.0
- base.JoinChannels:
    inputChan0: flow_xf
    inputChan1: flow_yf
    inputChan2: ''
    inputChan3: ''
    outputName: outFlow
- filters.GaussianFilter:
    inputName: input
    outputName: filtered_input
    sigmaX: 10.0
    sigmaY: 10.0
'''



class JignaWebApp(web.Application):
    """ A web based App to serve the jigna template with a given context over
    the web where it can be viewed using a regular web browser. """

    def __init__(self, handlers=None, default_host="", transforms=None,
                 context=None, template=None, trait_change_dispatch="same",
                 async=False, **kw):

        if template is not None:
            template.async = async
        self.context = context
        self.template = template
        self.trait_change_dispatch = trait_change_dispatch
        self.async = async

        if handlers is None:
            handlers = []
        handlers = self._create_handlers() + handlers

        super(JignaWebApp, self).__init__(handlers, default_host, transforms, **kw)

    #### Private protocol #####################################################

    def _create_handlers(self):
        """
        Create the web application serving the given context. Returns the
        tornado application created.
        """

        # Set up the WebServer to serve the domain models in context
        klass = jignaws.AsyncWebServer if self.async else jignaws.WebServer
        server = klass(
            base_url              = os.path.join(os.getcwd(), self.template.base_url),
            html                  = self.template.html,
            context               = self.context,
            trait_change_dispatch = self.trait_change_dispatch
        )

        server.handlers.pop(-1) #remove the html handler - we will take care of this elsewhere ...
        
        return server.handlers


def main():
    os.environ['DJANGO_SETTINGS_MODULE'] = 'clusterUI.settings' # path to your settings module
    application = get_wsgi_application()

    django_app = tornado.wsgi.WSGIContainer(application)
    tornado_app = JignaWebApp(handlers=[
        (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': PYME.resources.get_web_static_dir()}),
        #(r'/media/(.*)', tornado.web.StaticFileHandler, {'path': MEDIA_URL}),
        #(r'/recipe_editor/(.*)', tornado.web.StaticFileHandler, {'path': os.path.dirname(html_recipe_editor.__file__)}),
        (r'.*', tornado.web.FallbackHandler, dict(fallback=django_app)),
        
        ], template=html_recipe_editor.template, context={'recipe': ModuleCollection.fromYAML(rec_text)})
    #server = tornado.httpserver.HTTPServer(tornado_app)
    
    http_server = tornado.httpserver.HTTPServer(tornado_app)
    http_server.listen(8889)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()