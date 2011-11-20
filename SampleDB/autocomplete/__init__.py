__version__ = '1.0'

def get_installed_version():
    import pkg_resources
    try:
        autocomplete = pkg_resources.get_distribution('django-autocomplete')
    except pkg_resources.DistributionNotFound:
        return __version__
    return autocomplete.version


def get_mercurial_version():
    import os
    path = os.path.join(__path__[0], os.pardir)
    try:
        from mercurial.hg import repository
        from mercurial.ui import ui
        from mercurial import node, error

        repo = repository(ui(), path)
    except:
        return None
    tip = repo.changelog.tip()
    rev = repo.changelog.rev(tip)
    return '%s.dev%d' % (__version__, rev)


def get_version():
    return get_mercurial_version() or get_installed_version()

