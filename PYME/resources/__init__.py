"""
PYME.resources
==============

A storage area for non-code based resources.

"""
import importlib.resources
import os
import pathlib

import jinja2

dirname = os.path.dirname(__file__)

def getIconPath(name):
	"""Returns the full path to the icon with the given name"""

	return os.path.join(dirname, 'icons', name)

def get_web_static_dir():
	return os.path.join(dirname, 'web', 'static')

def get_web_dir():
	return os.path.join(dirname, 'web')

def get_test_data_dir():
	return os.path.join(dirname, 'test_datasets')


class PackageResourceLoader(jinja2.BaseLoader):
	def __init__(self, package, resource_root='templates'):
		self.package = package
		self.resource_root = resource_root

	def _resource_dir(self):
		resource_dir = importlib.resources.files(self.package)

		if self.resource_root:
			for part in pathlib.PurePosixPath(self.resource_root).parts:
				resource_dir = resource_dir.joinpath(part)

		return resource_dir

	def get_source(self, environment, template):
		template_file = self._resource_dir().joinpath(template)

		try:
			source = template_file.read_text(encoding='utf-8')
		except (FileNotFoundError, UnicodeDecodeError, OSError, AttributeError):
			raise jinja2.TemplateNotFound(template)

		return source, str(template_file), lambda: False