(function( window, undefined ) {

var $ = window.jQuery || window.django.jQuery,
    django = window.django || {};

$.widget( "ui.djangoautocomplete", {
    options: {
        source: "../autocomplete/$name/",
        multiple: false,
        force_selection: true,
        renderItem: function( ul, item) {
            return $( "<li></li>" )
                .data( "item.autocomplete", item )
                .append( $( "<a></a>" ).append( item.label ) )
                .appendTo( ul );
        }

    },
    _create: function() {
        var self = this;
        this.hidden_input = this.element.prev( "input[type=hidden]" );
        this.name = this.hidden_input.attr( "name" );
        this.element.autocomplete({
            appendTo: this.element.parent(),
            select: function( event, ui ) {
                self.lastSelected = ui.item;
                if ( self.options.multiple ) {
                    if ( $.inArray( ui.item.id, self.values ) < 0 ) {
                        $('<li></li>')
                            .addClass( "ui-autocomplete-value" )
                            .data( "value.autocomplete", ui.item.id )
                            .append( ui.item.label+'<a href="#">x</a>' )
                            .appendTo( self.values_ul );
                        self.values.push( ui.item.id );
                    }
                    return false;
                }
            }
        }).data( "autocomplete" )._renderItem = this.options.renderItem;
        this._initSource();
        if ( this.options.multiple ) {
            this._initManyToMany();
        } else {
            this.lastSelected = {
                id: this.hidden_input.val(),
                value: this.element.val()
            };
        }
        if (this.options.force_selection) {
            this.element.focusout(function() {
                if ( self.element.val() != self.lastSelected.value ) {
                    self.element.val( "" );
                }
            });
        }
        this.element.closest( "form" ).submit(function() {
            if ( self.options.multiple ) {
                self.hidden_input.val( self.values.join(",") );
            } else if ( self.options.force_selection ) {
                self.hidden_input.val( self.element.val() ? self.lastSelected.id : "" );
            } else {
                self.hidden_input.val( self.element.val() );
            }
        });
    },
    
    destroy: function() {
        this.element.autocomplete( "destroy" );
        if ( this.options.multiple ) {
            this.values_ul.remove();
        }
		$.Widget.prototype.destroy.call( this );
    },

    _setOption: function( key, value ) {
		$.Widget.prototype._setOption.apply( this, arguments );
        if ( key === "source" ) {
            this._initSource();
        }
    },

    _initSource: function() {
        var source = typeof this.options.source === "string" ?
            this.options.source.replace( "$name", this.hidden_input.attr("name") ) :
            this.options.source;
        this.element.autocomplete( "option", "source", source );
    },

    _initManyToMany: function() {
        var self = this;
        this.element.bind( "autocompleteclose", function( event, ui ) {
            self.element.val( "" );
        });
        this.values = [];
        if ( this.hidden_input.val() !== "" ) {
            $.each(this.hidden_input.val().split( "," ), function(i, id) {
                self.values.push( parseInt(id, 10) );
            });
        }
        this.values_ul = this.element.nextAll( "ul.ui-autocomplete-values" );
        this.lastSelected = { id: null, value: null };
        if ( this.values.length && this.values_ul[0] ) {
            this.values_ul.children().each(function(i) {
                $(this)
                    .addClass( "ui-autocomplete-value" )
                    .data( "value.autocomplete", self.values[i] )
                    .append( '<a href="#">x</a>' );
            });
        } else {
            this.values_ul = $( "<ul></ul>" ).insertAfter( this.element );
        }
        this.values_ul.addClass( "ui-autocomplete-values" );
        $( ".ui-autocomplete-value a", this.values_ul[0] ).live( "click", function() {
            var span = $(this).parent();
            var id = span.data( "value.autocomplete" );
            $.each( self.values, function (i, v) {
                if (v === id) {
                    self.values.splice(i, 1);
                }
            });
            span.remove();
        });
    }
});


django.autocomplete = function (id, options) {
    return $(id).djangoautocomplete(options);
};

window.django = django;

})(window);
