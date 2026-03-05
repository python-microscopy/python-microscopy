import jinja2
from . import filters

import os
from PYME.config import get_plugin_template_paths
import logging
import importlib

logger = logging.getLogger(__name__)

# def _plugin_template_path(template):
#     plugin_template_dirs = get_plugin_template_dirs()  # get templates from plugins
#     try:
#         t_plugin, t_template = template.replace(os.sep, '/').split('/')
#         return os.path.join(plugin_template_dirs[t_plugin], t_template)
#     except (ValueError, KeyError):
#         return ''

class UnifiedLoader(jinja2.BaseLoader):
    def get_source(self, environment, template):
        from PYME.IO import unifiedIO
        try:
            source = None

            # First try to load from built-in templates
            try:
                source = importlib.resources.read_text(__package__, 'templates/' + template)
            except:
                pass

            # Next try to load from plugin templates
            if source is None:
                for template_module in get_plugin_template_paths():
                    try:
                        source = importlib.resources.read_text(template_module, template)
                        break
                    except:
                        pass

            # Finally, try to load from filesystem path or PYMECluster URI
            if source is None:
                source = unifiedIO.read(template).decode('utf-8')
        except:
            logger.exception('Error loading template')
            raise jinja2.TemplateNotFound(template)

        return source, template, lambda: False


env = jinja2.Environment(loader=UnifiedLoader())
env.filters.update(filters.populate_filters())
