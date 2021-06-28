import matplotlib.pyplot as plt


class plotting_context(object):
    def __init__(self, backend='SVG'):
        self._backend = backend
        
    def __enter__(self):
        plt.ioff()
        self._old_backend = plt.get_backend()
        plt.switch_backend(self._backend)
    
    def __exit__(self, *args):
        plt.switch_backend(self._old_backend)
        plt.ion()

class Plot(object):
    """
    This is a class that defines a plot method that can be called to generate a matplotlib figure.
    
    The Plot class (or any subclass thereof) can be returned by recipe module with the actual plotting deferred until
    the plot() method is called. This facilitates the use of different matplotlib backends (as appropriate) depending
    on the context in which the recipe is running.
    
    The Plot class should either be subclassed with plot() over-ridden, or instantiated with a callable (e.g. a lambda
    function). This is conceptually similar to how the python Thread class behaves.
    
    The plot function defines a number of translators, for use in report templates or output modules
    
    """
    
    def __init__(self, plot_callable=None):
        """
        When the plot method has not been overridden, a callable function should be provided
         
        Parameters
        ----------
        plot_callable : a callable to generate the plots. Must return a matplotlib figure object
        """
        self._plot_callable = plot_callable
        
    def plot(self):
        """
        Do the plot in whatever context we have activated. Can be over-ridden in derived classes
        
        
        Returns
        -------
        
        a matplotlib figure instance

        """
        if self._plot_callable is None:
            raise RuntimeError('plot_callable not defined - either initialize the class with a callable, or over-ride plot method')
        else:
            return self._plot_callable()
            
    def as_html(self):
        """
        
        Returns
        -------
        
        The plot as an html string using mpld3

        """
        import mpld3
        import warnings
        if warnings.filters[0] == ('always', None, DeprecationWarning, None, 0):
            #mpld3 has messed with warnings - undo
            warnings.filters.pop(0)
        
        with plotting_context('SVG') as p:
            f = self.plot()

            ret = mpld3.fig_to_html(f, template_type='simple')
            plt.close(f)

            return ret
    
    def _savefig(self, fig, filename, format='png'):
        if filename is not None:
            fig.savefig(filename)
        else:
            from io import BytesIO
            b = BytesIO()
            fig.savefig(b)
            return b
    
    def as_png(self, filename=None):
        with plotting_context('agg') as p:
            f = self.plot()
        
            return self._savefig(f, filename, 'png')
    
    def as_pdf(self, filename=None):
        with plotting_context('pdf') as p:
            f = self.plot()
            
            return self._savefig(f, filename, 'pdf')
    
    def as_svg(self, filename=None):
        with plotting_context('svg') as p:
            f = self.plot()
        
            return self._savefig(f, filename, 'svg')

            