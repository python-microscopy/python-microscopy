/**
 * Created by david on 12/04/20.
 */
// Long Polling to update image
// function html(strings){
//     return strings.raw;
// }

function poll_png(){
    $.ajax({ url: "/get_frame_png_b64", success: function(data){
        $("#cam_image").attr("src", "data:image/png;base64,"+data);
        }, dataType: "text", complete: function(jqXHR, status){
            if (status == 'success') {poll_png();} else {console.log('Error during image polling, make sure server is up and try refreshing the page');}
        }, timeout: 30000 });
}



var _min = 0;
var _max = 1e9;

function map_array(data, cmin, cmax){
    out = new Uint8ClampedArray(data.length*4);

    //record min and max
    _min = 1e9;
    _max = 0;

    for (j = 0; j< data.length; j++){
        k = j*4;
        v = data[j];
        _min = Math.min(v, _min);
        _max = Math.max(v,_max);
        v = (v - cmin)/(cmax-cmin);
        //console.log(v)
        v_ = 255*v; //simple grayscale map - FIXME
        out[k] = v_;
        out[k+1] = v_;
        out[k+2] = v_;
        out[k+3] = 255; //alpha
    }

    return out
}

// Long Polling to update image
function poll_array(){
    $.ajax({
        url: "/get_frame_pzf",
        success: function(data){
            //console.log(data);
            decoded = decode_pzf(data);
            //console.log(decoded);
            im = new ImageData(map_array(decoded.data, parseFloat($("#display_min").val()), parseFloat($("#display_max").val())), decoded.width, decoded.height);
            if ($("#display_autoscale").is(":checked")){
                $("#display_min").val(_min);
                $("#display_max").val(_max);
            }
            var zoom = parseFloat($("#display_zoom").val())/100.;
            var canvas = document.getElementById("cam_canvas");
            $("#cam_canvas").attr({width: decoded.width*zoom, height : decoded.height*zoom});
            var ctx = canvas.getContext('2d');
            //ctx.scale(zoom, zoom);

            createImageBitmap(im, options={resizeWidth: decoded.width*zoom, resizeHeight:decoded.height*zoom, resizeQuality:'pixelated'}).then(function(bmp){
                ctx.drawImage(bmp, 0, 0);
            });
            //console.log(im);
            //ctx.putImageData(im, 0, 0);

            },
        //dataType: "text",
        complete: function(jqXHR, status){
                if (status == 'success') {poll_array();} else {console.log('Error during image polling, make sure server is up and try refreshing the page');}
            },
        timeout: 30000,
        xhrFields: {responseType: 'arraybuffer'}
    });
}

poll_array();

function log_ajax_error(jqXHR, textStatus, errorThrown){
    console.log(textStatus);
    console.log(errorThrown);
    console.log(jqXHR);
}

function update_server_state(state){
    //console.log('updating state', state);
    $.ajax({
        url : "/update_scope_state",
        data : JSON.stringify(state),
        processData: false,
        type: 'POST',
        error: log_ajax_error,
    })
}

function update_stack_settings(settings){
    console.log('updating stack', settings);
    $.ajax({
        url : "/stack_settings/update",
        data : JSON.stringify(settings),
        processData: false,
        type: 'POST',
        error: log_ajax_error,
    })
}

function update_spooler_settings(settings){
    console.log('updating spooler', settings);
    $.ajax({
        url : "/spool_controller/settings",
        data : JSON.stringify(settings),
        processData: false,
        type: 'POST',
        error: log_ajax_error,
    })
}

//

function start_spooling(filename=null,max_frames=null){
    //console.log('updating state', state);
    $.ajax({
        url : "/spool_controller/start_spooling",
        //data : JSON.stringify(state),
        processData: false,
        type: 'GET',
        error: log_ajax_error,
    })
}

Vue.component('position-control', {
    props: {'value' : Number, 'axis' : String, 'delta' : {type:[Number,], default: 1.0}},
    template: /* html */ `<div class="input-group input-group-sm">
                
                <label class="form-control-sm"> {{ axis }} [um]:&nbsp;
                <button type="button" class="btn btn-dark" v-on:click="set_position(axis, value - delta)">&lt;</button>
                    <input type="number" v-bind:value="value" style="width: 80px"
                    v-on:change="set_position(axis, $event.target.value);$emit('input', $event.target.value)" class="form-control form-control-sm">
                    
                    <button type="button" class="btn btn-dark" v-on:click="set_position(axis, value + delta)">&gt;</button>
                </label>
                </div>`,
    methods:{
        set_position: function(axis, value){
            //console.log(delta);
            //update_server_state(dict_fill('Positioning.' + axis, parseFloat(value)));
            update_server_state({['Positioning.' + axis]: parseFloat(value)});
        }
    }
});

Vue.component('laser-control', {
    props: ['power', 'on','name', 'max_power'],
    template: /* html */`<div class="input-group input-group-sm">
                <label class="form-control-sm">{{ name }}&nbsp;
                    <input type="range" class="form-control form-control-sm"
                            :value="power"
                            :max="max_power"
                            v-on:change="set_laser_power(name, $event.target.value)">&nbsp;

                    <input type="number" class="form-control form-control-sm" style="width: 50px"
                            :value="power" v-on:change="set_laser_power(name, $event.target.value)">
                </label>

                <div class="form-check">
                    <input class="form-check-input" type="checkbox"
                            :value="on" :id="'laser-check-' + name"
                            v-on:change="set_laser_on(name, $event.target.checked)">
                    <label class="form-check-label" :for="'laser-check-' + name">On</label>
                </div>

            </div>`,
    methods: {
        set_laser_power: function (lname, value) {
            //update_server_state(dict_fill('Lasers.' + lname + '.Power', parseFloat(value)));
            update_server_state({['Lasers.' + lname + '.Power']: parseFloat(value)});
        },
        set_laser_on: function (lname, value) {
            //console.log('turning ' + lname + ' on: ' + value);
            //update_server_state(dict_fill('Lasers.' + lname + '.On', value));
            update_server_state({['Lasers.' + lname + '.On']: value});
        },
    }
});

Vue.component('stack-settings', {
    props: {value : Object,
            show_dwell_time: {type: Boolean, default: false}},
    template: /* html */`<div>
                <div class="input-group input-group-sm">
                    <span class="form-control-sm">Axis:&nbsp&nbsp</span>
                    <select  class="form-control form-control-sm" :value="value.ScanPiezo" v-on:change="update_setting('ScanPiezo', $event.target.value)">
                        <option selected>z</option>
                        <option>...</option>
                        </select>
                </div>
                <div class="input-group input-group-sm">
                <span class="form-control-sm">Mode: </span>
                <select class="form-control form-control-sm" :value="value.ScanMode"  v-on:change="update_setting('ScanMode', $event.target.value)">
                        <option>Middle and Number</option>
                        <option>Start and End</option> </select>
                </div>
                <div class="form-row">
                <div class="input-group input-group-sm col">
                <span class="form-control-sm">Start: </span>
                <input class="form-control form-control-sm" type="number" step=.01 :value="value.StartPos"  v-on:change="update_setting('StartPos', $event.target.value)">
                <div class="input-group-append"><button class="btn-light">Set</button> </div>
                </div>
                <div class="input-group input-group-sm col">
                <span class="form-control-sm">End: </span>
                <input class="form-control form-control-sm" type="number" step=.01 :value="value.EndPos"  v-on:change="update_setting('EndPos', $event.target.value)">
                <div class="input-group-append"><button class="btn-light">Set</button> </div>
                </div>
                </div>
                <div class="form-row">
                <div class="input-group input-group-sm col">
                <span class="form-control-sm">Step size [um]: </span>
                <input class="form-control form-control-sm" type="number" :value="value.StepSize"  v-on:change="update_setting('StepSize', $event.target.value)">
                </div>
                <div class="input-group input-group-sm col">
                <span class="form-control-sm">Num slices: </span>
                <input class="form-control form-control-sm" type="number" :value="value.NumSlices"  v-on:change="update_setting('NumSlices', $event.target.value)">
                </div>
                <div class="input-group input-group-sm col" v-if="show_dwell_time">
                <span class="form-control-sm">Dwell time: </span>
                <input class="form-control form-control-sm" type="number" step=.01 :value="value.DwellFrames"  v-on:change="update_setting('DwellFrames', $event.target.value)">
                </div>
</div>
                </div>
                
                `,
    methods: {
        update_setting: function(propname, value){
            update_stack_settings({[propname] : value});
        }

    }
});


Vue.component('palm-storm-settings', {
    props: {spooler: Object, stack: Object},
    computed: {
        spool_z_stepping: {
            get: function (){
                if (this.spooler.settings.z_stepped){
                    return 'true';
                } else return false;
            },
            set: function(newValue){
                update_spooler_settings({'z_stepped': newValue=='true'});
                //this.spooler.settings.z_stepped = (newValue == 'true');
            }
        }
    },
    template: /* html */`
    <div>
    <h6>Protocol</h6>
    <form class="form">
        <div class="input-group input-group-sm">
            <span class="form-control-sm">Protocol File: </span>
            <select id="inputState" class="form-control form-control-sm">
                <option selected>None chosen...</option>
                <option>...</option>
                </select>
        </div>
        <div class="input-group input-group-sm ml-2 mt-1">
            <div class="form-check-inline">
            <input class="form-check-input" type="radio" id="aq_standard" v-model="spool_z_stepping" value=false>
            <label class="form-check-label" for="aq_standard">Standard</label>
            </div>
            <div class="form-check-inline">
            <input class="form-check-input" type="radio" id="aq_zs" v-model="spool_z_stepping" value=true>
            <label class="form-check-label" for="aq_zs">Z stepped</label>
            </div>
        </div>

        <stack-settings v-if="spooler.settings.z_stepped" v-model="stack" class="mt-1"></stack-settings>

    </form>
    </div>
    `
})


Vue.component('simulation-settings', {
    props: {simcontrol: Object},
    template: /* html */`
    <div>
    <h6>Simulation</h6>
    <form class="form">
    </form>
    </div>
    `
})

var scope_state = {};
scope_state['Camera.IntegrationTime']=0.1; //default start option


var app = new Vue({
    el: '#app',
    data: {
        //message: 'Hello Vue!',
        state: scope_state,
        spooler : {status:{spooling:false,},
                    settings:{z_stepped:false}},
        stack : {},
        },
    computed: {
        integration_time_ms: function () {
            return this.state['Camera.IntegrationTime']*1000;
            },

        laser_names: function () {
            lks = Object.keys(this.state).filter(function(key){return key.startsWith('Lasers') && key.endsWith('On');})
            laser_info = lks.map(function(k){
                lname= k.split('.')[1];
                return lname;
            });
            return laser_info;
            },
        spool_z_stepping: {
            get: function (){
                if (this.spooler.settings.z_stepped){
                    return 'true';
                } else return false;
            },
            set: function(newValue){
                update_spooler_settings({'z_stepped': newValue=='true'});
                //this.spooler.settings.z_stepped = (newValue == 'true');
            }
        }
        },
    methods:{
        update_server_state : update_server_state,
        set_laser_power: function(lname, value){var key = 'Lasers.' + lname + '.Power';
                                        var state = {};
                                        state[key] = parseFloat(value);
                                        update_server_state(state);} ,
        set_laser_on: function(lname, value){var key = 'Lasers.' + lname + '.On';update_server_state({key: value});},
    }
});

//get initial values
$.ajax({url: "/get_scope_state", success: function(data){app.state=data;}});
$.ajax({url: "/spool_controller/info", success: function(data){app.spooler = data;}});
$.ajax({url: "/stack_settings/settings", success: function(data){app.stack = data;}});

function poll_updates(url, attrib){
    var _poll = function(){
        $.ajax({
            url: url,
            success: function(data) {
                        app[attrib] = data;
                        console.log('updated ' + attrib + ' on ' + app);

                },
            complete: function(jqXHR, status){if (status == 'success') {_poll();} else {console.log('Error whilst polling ' + url + ', make sure server is up and try refreshing the page');}}
        })
    };
    _poll();
}


poll_updates("/scope_state_longpoll", 'state');
poll_updates("/spool_controller/info_longpoll",  'spooler');
poll_updates("/stack_settings/settings_longpoll", 'stack');


$(window).on('load', function(){$("#home-tab").tab('show');});