from django.forms.widgets import TextInput
from django.utils.encoding import force_text
from django.forms.utils import flatatt
from django.utils.html import format_html

class ClusterFileInput(TextInput):
    def render(self, name, value, attrs=None, renderer=None):
        if value is None:
            value = ''
        final_attrs = self.build_attrs(attrs, {'type': self.input_type, name: name})
        if value != '':
            # Only add the 'value' attribute if a value is non-empty.
            final_attrs['value'] = force_text(self.format_value(value))
        # return format_html('''<div class="input-group">
        #                         <span class="input-group-addon">pyme-cluster:///</span>
        #                         <input class="form-control" {} />
        #                         <span class="input-group-btn">
        #                             <button class="btn btn-default" type="button" onclick="select_file(function(path){{$('#{}').val('pyme-cluster:///' + path);}});">select</button>
        #                         </span>
        #                     </div>''', flatatt(final_attrs), attrs['id'].replace('.', r'\\.'))

        return format_html('''<div class="input-group">
            <input class="form-control" {} />
            <span class="input-group-btn">
                <button class="btn btn-default" type="button" onclick="select_file(function(path){{$('#{}').val('pyme-cluster:///' + path);}});">select</button>
            </span>
        </div>''', flatatt(final_attrs), attrs['id'].replace('.', r'\\.'))