import jinja2
from . import filters

import os
from PYME.config import get_plugin_template_dirs
import logging

logger = logging.getLogger(__name__)

def _plugin_template_path(template):
    plugin_template_dirs = get_plugin_template_dirs()  # get templates from plugins
    try:
        t_plugin, t_template = template.replace(os.sep, '/').split('/')
        return os.path.join(plugin_template_dirs[t_plugin], t_template)
    except (ValueError, KeyError):
        return ''

class UnifiedLoader(jinja2.BaseLoader):
    def get_source(self, environment, template):
        from PYME.IO import unifiedIO
        try:
            if os.path.exists(os.path.join(os.path.dirname(__file__), 'templates', template)):
                source = unifiedIO.read(os.path.join(os.path.dirname(__file__), template)).decode('utf-8')
            elif os.path.exists(_plugin_template_path(template)):
                source = unifiedIO.read(_plugin_template_path(template)).decode('utf-8')
            else:
                source = unifiedIO.read(template).decode('utf-8')
        except:
            logger.exception('Error loading template')
            raise jinja2.TemplateNotFound(template)

        return source, template, lambda: False


env = jinja2.Environment(loader=UnifiedLoader())
env.filters.update(filters.populate_filters())
