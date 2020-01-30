
import jinja2
from PYME.reports import templates, filters


env = jinja2.Environment(loader=templates.UnifiedLoader())
env.filters.update(filters.get_filters())
