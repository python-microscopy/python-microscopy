function bassistance_autocomplete(name, ac_url, force_selection) {
    $(document).ready(function () {
        var input = $('#id_' + name);
        var hidden_input = $('#id_hidden_' + name);
        input.autocomplete(ac_url, {
            limit: 10,
            matchSubset: false,
            dataType: 'json',
            parse: function(data) {
                var parsed = [];
                for (var i in data) {
                    row = {
                        data: data[i][1]+'|'+data[i][0],
                        value: data[i][0],
                        result: data[i][1]
                    };
                    parsed[parsed.length] = row;
                }
                return parsed;
            },    
            formatItem: function(data, i, total) {
                return data.split('|')[0];
            }
        });
        input.result(function(event, data, formatted) {
            hidden_input.val(data.split('|')[1]);
        });
        form = $("form:first");
        form.submit(function() {
            if (hidden_input.val() != input.val() && !force_selection) {
                hidden_input.val(input.val());
            }
        });
    });
}


function jquery_autocomplete(name, ac_url, force_selection) {

    this.name = name;
    this.ac_url = ac_url;
    this.force_selection = force_selection;

    this.source = function (request, response) {
        function success(data) {
            var parsed = [];
            for (var i in data) {
                parsed[parsed.length] = {
                    id: data[i][0],
                    value: data[i][1],
                    label: data[i][1],
                };
            }
            response(parsed);
        };
        $.ajax({
            url: this.ac_url,
            dataType: "json",
            data: {q: request.term},
            success: success
        });
    };

    this.select = function (event, ui) {
        // set the hidden input field.
        this.last_item = ui.item;
        this.hidden_input.val(ui.item.id);
    };

    this.close = function (event, ui) {
        alert(ui.toSource());  
    };

    this.setup = function () {
        this.input = $("#id_" + this.name);
        this.hidden_input = $("#id_hidden_" + this.name);
        this.last_item = {};
        this.input.autocomplete({
            // minLength: 2,
            source: jQuery.proxy(this.source, this),
            select: jQuery.proxy(this.select, this),
        });
        this.input.closest("form").submit(jQuery.proxy(function () {
            if ((!this.input.val()) || (this.hidden_input.val() != this.input.val()
                && !this.force_selection)) {
                this.hidden_input.val(this.input.val());
            }
        }, this));
        if (this.force_selection) {
            this.input.focusout(jQuery.proxy(function (event) {
                if (this.input.val() != this.last_item.value)
                    this.input.val("");
            }, this));
        }
    };

    $(document).ready(jQuery.proxy(this.setup, this));
};

autocomplete = jquery_autocomplete;
