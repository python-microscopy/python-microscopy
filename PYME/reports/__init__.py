
import jinja2
import os
from PYME.reports import templates, filters
import logging

logger = logging.getLogger(__name__)

class UnifiedLoader(jinja2.BaseLoader):
    def get_source(self, environment, template):
        from PYME.IO import unifiedIO
        try:
            if os.path.exists(os.path.join(os.path.dirname(__file__), template)):
                source = unifiedIO.read(os.path.join(os.path.dirname(__file__), template)).decode('utf-8')
            else:
                source = unifiedIO.read(template).decode('utf-8')
        except:
            logger.exception('Error loading template')
            raise jinja2.TemplateNotFound
        return source, template, lambda: False

env = jinja2.Environment(loader=UnifiedLoader())

env.filters.update(filters.get_filters())
