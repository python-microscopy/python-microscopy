
import jinja2
import os
from PYME.config import get_report_templates
import logging

logger = logging.getLogger(__name__)

class UnifiedLoader(jinja2.BaseLoader):
    def get_source(self, environment, template):
        from PYME.IO import unifiedIO
        templates = get_report_templates()  # get templates from plugins
        try:
            if template in templates.keys():
                source = unifiedIO.read(templates[template]).decode('utf-8')
            elif os.path.exists(os.path.join(os.path.dirname(__file__), template)):
                source = unifiedIO.read(os.path.join(os.path.dirname(__file__), template)).decode('utf-8')
            else:
                source = unifiedIO.read(template).decode('utf-8')
        except:
            logger.exception('Error loading template')
            raise jinja2.TemplateNotFound(template)
        return source, template, lambda: False
