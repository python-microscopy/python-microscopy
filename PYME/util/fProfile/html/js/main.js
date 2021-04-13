var display_params = {
    OVERVIEW_HEIGHT : 20,
    ZOOM_HEIGHT : 100
};

//### - from d3-flame-graph
var colorMapper = function(d) {
    return d.highlight ? "#E600E6" : colorHash(d.name);
};

function generateHash(name) {
    // Return a vector (0.0->1.0) that is a hash of the input string.
    // The hash is computed to favor early characters over later ones, so
    // that strings with similar starts have similar vectors. Only the first
    // 6 characters are considered.
    var hash = 0, weight = 1, max_hash = 0, mod = 10, max_char = 6;
    if (name) {
        for (var i = 0; i < name.length; i++) {
            if (i > max_char) { break; }
            hash += weight * (name.charCodeAt(i) % mod);
            max_hash += weight * (mod - 1);
            weight *= 0.70;
        }
        if (max_hash > 0) { hash = hash / max_hash; }
    }
    return hash;
}

/**
 * Converts an HSL color value to RGB. Conversion formula
 * adapted from http://en.wikipedia.org/wiki/HSL_color_space.
 * Assumes h, s, and l are contained in the set [0, 1] and
 * returns r, g, and b in the set [0, 255].
 *
 * @param   {number}  h       The hue
 * @param   {number}  s       The saturation
 * @param   {number}  l       The lightness
 * @return  {Array}           The RGB representation
 */
function hslToRgb(h, s, l){
    var r, g, b;

    if(s == 0){
        r = g = b = l; // achromatic
    }else{
        var hue2rgb = function hue2rgb(p, q, t){
            if(t < 0) t += 1;
            if(t > 1) t -= 1;
            if(t < 1/6) return p + (q - p) * 6 * t;
            if(t < 1/2) return q;
            if(t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        }

        var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        var p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }

    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

String.prototype.hashCode = function() {
    var hash = 0, i, chr, len;
    if (this.length === 0) return hash;
    for (i = 0, len = this.length; i < len; i++) {
        chr   = this.charCodeAt(i);
        hash  = ((hash << 5) - hash) + chr;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
};

var divisor = Math.pow(2, 32);

function colorHash(name) {
    // Return an rgb() color string that is a hash of the provided name,
    // and with a warm palette.
    var vector = 0;
    if (name) {
        name = name.replace(/.*`/, "");		// drop module name if present
        name = name.replace(/\(.*/, "");	// drop extra info
        vector = generateHash(name);
    }
    //vector = name.hashCode()/divisor;
    //vector = Math.abs(vector/Math.pow(2, 32));

    var r = 0 + Math.round(55 * vector);
    var g = 0 + Math.round(230 * (1 - vector));
    var b = 200 + Math.round(55 * (vector));
    /*var col = hslToRgb(vector, 0.5, 0.5);
    var r = col[0],
        g = col[1],
        b = col[2];*/
    /*var r = (name.charCodeAt(0) - 90)*2 + 150,
        g = (name.charCodeAt(1) - 90)*2 + 150,
        b = (name.charCodeAt(2) - 90)*2 + 150;*/
    return "rgb(" + r + "," + g + "," + b + ")";
    //return "hsl(" + (100*vector) + "%, 50%, 50%)";
}

function colorHashWarm(thread) {
    // Return an rgb() color string that is a hash of the provided name,
    // and with a warm palette.
    var vector = 0;
    //
    vector = (thread % 10)/10.0;

    //vector = name.hashCode()/divisor;
    //vector = Math.abs(vector/Math.pow(2, 32));

    var b = 0 + Math.round(55 * vector);
    var g = 0 + Math.round(230 * (1 - vector));
    var r = 200 + Math.round(55 * (vector));
    /*var col = hslToRgb(vector, 0.5, 0.5);
    var r = col[0],
        g = col[1],
        b = col[2];*/
    /*var r = (name.charCodeAt(0) - 90)*2 + 150,
        g = (name.charCodeAt(1) - 90)*2 + 150,
        b = (name.charCodeAt(2) - 90)*2 + 150;*/
    return "rgb(" + r + "," + g + "," + b + ")";
    //return "hsl(" + (100*vector) + "%, 50%, 50%)";
}
//### - end from d3-flame-graph

function label_0(d) {
    return d.n + '\n' + ((d.tf -d.ts)*1000).toPrecision(2) + ' ms\n\n' + d.ns + '\n' + d.f
}

function label_1(d) {
    return d.f + ' - ' + d.n
}

d3.select("#chart_d3 svg").remove();
var svg = d3.select("#chart_d3").append("svg")
    .style("position", "relative")
    .attr("width", 100 + "%")
    .attr("height", 600 + "px");



var xs = d3.scale.linear().range([0, $("#chart_d3").width()]);
var xs_z = d3.scale.linear().range([0, $("#chart_d3").width()]);
var ys = d3.scale.linear().range([5, display_params.OVERVIEW_HEIGHT]);
var ys_z = d3.scale.linear().range([20, display_params.ZOOM_HEIGHT]);

//Overview
var xAxis_1 = d3.svg.axis()
    .scale(xs)
    .orient("top");

var overview = svg.append('g');
overview.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + 20 + ")")
    .call(xAxis_1);



//zoomed view
var xAxis_z = d3.svg.axis()
    .scale(xs_z)
    .orient("top");

var zoomed = svg.append('g');



function load_data(url) {
    d3.json(url, function (error, prof_data) {
        var callstack = prof_data.callstack;
        var threadNames = prof_data.threadNames;
        var threadData = prof_data.threadLines;

        var nthreads = prof_data.maxConcurrentThreads;//threadNames.length;

        svg.attr("height", 60 + nthreads * (display_params.OVERVIEW_HEIGHT + display_params.ZOOM_HEIGHT));

        var Y_ZOOM_START = (nthreads + 1) * display_params.OVERVIEW_HEIGHT;
        //console.log(nthreads, Y_ZOOM_START);

        /*var tip = d3.tip()
         .direction("s")
         .offset([8, 0])
         .attr('class', 'd3-tip')
         .html(function(d) { return label(d); });*/


        /*d3.select("#chart_d3").call(tip)*/


        /*var data = crossfilter(prof_data),
         duration = data.dimension(function(d) { return d.tf - d.ts; }),
         ts = data.dimension(function (d) {return d.ts;}),
         tf = data.dimension(function (d) {return d.tf;});

         console.log(data)*/


        var x0 = 0;
        var x1 = d3.max(callstack, function (d) {
            return d.tf;
        });
        //console.log('x1: ' + x1);

        xs.domain([0, x1]);
        xs_z.domain([0, x1]);
        ys.domain([0, 8]);
        ys_z.domain([0, 10]);

        zoomed.select(".x.axis").call(xAxis_z);
        overview.select(".x.axis").call(xAxis_1);

        /*var zoom = d3.behavior.zoom()
         .scaleExtent([1, Infinity])
         //.translateExtent([[0,0], [2,0]])
         .x(xs)
         .on("zoom", function () {
         //overview.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")");
         //print("zoom");
         //x0 = xs.domain[0];
         //x1 = xs.domain[1];
         //x1 = x1*d3.event.scale
         //xs.domain([x0, x1]);
         //var zoomed_domain = xs.domain();
         //xs.domain([Math.max(zoomed_domain[0], 0), Math.min(zoomed_domain[1], 90)])
         overview.select(".x.axis").call(xAxis_1);

         overview.selectAll("rect")
         .attr("x", function (d) {
         return xs(d.ts);
         })
         .attr("width", function (d) {
         return xs(d.tf) - xs(d.ts);
         })

         //console.log("here", d3.event.translate, d3.event.scale);
         }) //.append('g')*/

        //duration.filter(xs.range()[1]/(xs.domain()[1] - xs.domain()[0]), Infinity)

        var min_duration = (xs.domain()[1] - xs.domain()[0]) / xs.range()[1];

        var filt_data = callstack.filter(function (d) {
            return (d.tf - d.ts) > min_duration
        });

        var sel = overview.selectAll(".bar")
            .data(filt_data);

        var bar = sel.enter().append("g")
            .attr("class", "bar")
            .attr("x", function (d) {
                return xs(d.ts);
            })
            .attr("y", function (d) {
                return ys(d.l);
            });

        sel.exit().remove();

        var rect = bar.append("rect")
            .attr("x", function (d) {
                return xs(d.ts);
            })
            .attr("y", function (d) {
                return ys(d.l) + d.tl * display_params.OVERVIEW_HEIGHT;
            })
            .attr("width", function (d) {
                return xs(d.tf) - xs(d.ts);
            })
            .attr("height", function (d) {
                return ys(1) - ys(0);
            })
            .append("svg:title").text(function (d) {
                return label_0(d);
            });

        //Now for the thread bars

        var sel_t = overview.selectAll(".threadbar")
            .data(threadData);

        var bar_t = sel_t.enter().append("g")
            .attr("class", "threadbar")
            .attr("x", function (d) {
                return xs(d.ts);
            })
            .attr("y", function (d) {
                return ys(d.tl);
            });

        sel_t.exit().remove();

        var rect_t = bar_t.append("rect")
            .attr("x", function (d) {
                return xs(d.ts);
            })
            .attr("y", function (d) {
                return d.tl* display_params.OVERVIEW_HEIGHT;
            })
            .attr("width", function (d) {
                return xs(d.tf) - xs(d.ts);
            })
            .attr("height", function (d) {
                return 1;
            })
            .attr("fill", function (d, i) {
                        return colorHashWarm(i);
                    });
            /*.append("svg:title").text(function (d) {
                return label_0(d);
            });*/

        //brush
        var brush_ = d3.svg.brush()
            .x(xs)
            .on("brush", brushed_);

        overview.append("g")
            .attr("class", "x brush")
            .call(brush_)
            .selectAll("rect")
            .attr("y", 0)
            .attr("height", Y_ZOOM_START);

        //axis for zoomed view
        zoomed.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (Y_ZOOM_START + 30) + ")")
            .call(xAxis_z);

        //zoomed view
        function update_zoomed() {
            zoomed.selectAll(".bar").remove();
            zoomed.selectAll(".threadbar").remove();

            var x0 = xs_z.domain()[0];
            var x1 = xs_z.domain()[1];

            var min_duration = (x1 - x0) / xs_z.range()[1];
            var filt_data = callstack.filter(function (d) {
                return ((d.tf - d.ts) > min_duration) && (d.tf > x0) && (d.ts < x1)
            });

            //console.log(min_duration)

            var sel = zoomed.selectAll(".bar")
                .data(filt_data);

            var bar_1 = sel.enter().append("g")
                .attr("class", "bar")
                .attr("x", function (d) {
                    return xs_z(d.ts);
                })
                .attr("y", function (d) {
                    return ys_z(d.l) + Y_ZOOM_START + d.tl * display_params.ZOOM_HEIGHT;
                });


            var rect_1 = bar_1.append("rect")
                    .attr("x", function (d) {
                        return xs_z(d.ts);
                    })
                    .attr("y", function (d) {
                        return ys_z(d.l) + Y_ZOOM_START + d.tl * display_params.ZOOM_HEIGHT;
                    })
                    .attr("width", function (d) {
                        return xs_z(d.tf) - xs_z(d.ts);
                    })
                    .attr("height", function (d) {
                        return ys_z(1) - ys_z(0);
                    })
                    .attr("fill", function (d) {
                        return colorHash(d.n);
                    })
                    .on("mouseover", function (d) {
                        this.style.fill = "brown";
                    })
                    .on("mouseout", function (d) {
                        this.style.fill = colorHash(d.n);
                    })
                    .append("svg:title").text(function (d) {
                        return label_0(d);
                    })

                ;

            /*var upd = sel.update().selectAll("rect")
             .attr("x", function (d) {return xs_z(d.ts);})
             .attr("width", function (d) {return xs_z(d.tf) - xs_z(d.ts);});*/
            /*.on('mouseover', function(d) {
             if(!d.dummy) {
             $("#info").text(function(d) {return(label(d));});
             }
             }).on('mouseout', function(d) {
             $("#info").text('');
             });*/
            var label = bar_1.append("text")
                .attr("x", "10px")//function(d) { return xs(d.ts); })
                .attr("y", function (d) {
                    return d.tl * display_params.ZOOM_HEIGHT + Y_ZOOM_START + 20;
                })//function(d) { ; })
                //.attr("dy", ".35em")
                .text(function (d) {
                    return label_1(d);
                });

            //Now for the thread bars

            var sel_t = zoomed.selectAll(".threadbar")
                .data(threadData);

            var bar_t = sel_t.enter().append("g")
                .attr("class", "threadbar")
                .attr("x", function (d) {
                    return xs_z(d.ts);
                })
                .attr("y", function (d) {
                    return ys_z(d.tl) + Y_ZOOM_START + d.tl * display_params.ZOOM_HEIGHT;
                });

            //sel_t.exit().remove();

            var rect_t = bar_t.append("rect")
                .attr("x", function (d) {
                    return xs_z(d.ts);
                })
                .attr("y", function (d) {
                    return ys_z(0) + Y_ZOOM_START + d.tl * display_params.ZOOM_HEIGHT
                })
                .attr("width", function (d) {
                    return xs_z(d.tf) - xs_z(d.ts);
                })
                .attr("height", function (d) {
                    return 3;
                })
                .attr("fill", function (d, i) {
                            return colorHashWarm(i);
                        });

            //sel.exit().remove();

            /*var sel2 = zoomed.selectAll(".bar").attr("x", function (d) {return xs_z(d.ts);})
             .attr("width", function (d) {return xs_z(d.tf) - xs_z(d.ts);});*/
        }

        function brushed_() {
            xs_z.domain(brush_.empty() ? xs.domain() : brush_.extent());
            //focus.select(".area").attr("d", area);
            zoomed.select(".x.axis").call(xAxis_z);
            /*zoomed.selectAll("rect")
             .attr("x", function (d) {
             return xs_z(d.ts);
             })
             .attr("width", function (d) {
             return xs_z(d.tf) - xs_z(d.ts);
             })*/
            update_zoomed();
        }

        update_zoomed();

        var status = document.getElementById('load_status');
        status.innerHTML = "data loaded";


        //svg.call(zoom);

        /*svg.selectAll(".label")
         .data(prof_data)
         .enter().append('text')
         .attr("x", function(d) { return xs(d.ts) + 5; })
         .attr("y", function(d) { return ys(d.l+1) -5; })
         .text(function(d){return d.n;})*/
    });
}

console.log('script run');

function OnFileUpload() {
    var selectedFile = document.getElementById('file_upload').files[0];
    if (selectedFile) {
        var status = document.getElementById('load_status');
        status.innerHTML = "loading data ...";
        var objectURL = window.URL.createObjectURL(selectedFile);
        load_data(objectURL);
    }
}