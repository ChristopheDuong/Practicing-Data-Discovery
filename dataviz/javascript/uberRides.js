// Setup Google Map View
var googleOpt = {
    zoom: 12,
    center: new google.maps.LatLng(37.77926, -122.41934),
    mapTypeId: google.maps.MapTypeId.ROADMAP
};
var map = new google.maps.Map(d3.select("#google-map-view").node(), googleOpt);

// Setup Canvas for WebGL
var canvasLayerOptions = {
    map: map,
    resizeHandler: resize,
    animate: false,
    updateHandler: update,
};

var canvasLayer = new CanvasLayer(canvasLayerOptions);
var gl = null;
var pointProgram = null;
var pi_180 = Math.PI / 180.0;
var pi_4 = Math.PI * 4;
var POINT_COUNT = 0;
var pixelsToWebGLMatrix = new Float32Array(16);
var mapMatrix = new Float32Array(16);

// initialize WebGL
try {
    gl = canvasLayer.canvas.getContext('webgl') ||
        canvasLayer.canvas.getContext('experimental-webgl');
    createShaderProgram("shader/point.vert", "shader/point.frag", ready);
} catch (e) {}

var rawData = null;
var timeData = null;

function reload() {
    createShaderProgram("shader/point.vert", "shader/point.frag", null);
    displayData(rawData, 3, 0.3);
}

function createShaderProgram(vertexFile, fragmentFile, readyCallback) {
    var vertexShader;
    var fragmentShader;
    d3.text(vertexFile, "text/plain", function(error, data) {
        if (error != null) {
            console.warn(error);
        } else {
            vertexShader = gl.createShader(gl.VERTEX_SHADER);
            gl.shaderSource(vertexShader, data);
            gl.compileShader(vertexShader);

            if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
                console.error(gl.getShaderInfoLog(vertexShader));
            }
        }
        d3.text(fragmentFile, "text/plain", function(error, data) {
            if (error != null) {
                console.warn(error);
            } else {
                fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
                gl.shaderSource(fragmentShader, data);
                gl.compileShader(fragmentShader);

                if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
                    console.error(gl.getShaderInfoLog(fragmentShader));
                }

                // link shaders to create our program
                pointProgram = gl.createProgram();
                gl.attachShader(pointProgram, vertexShader);
                gl.attachShader(pointProgram, fragmentShader);
                gl.linkProgram(pointProgram);

                if (!gl.getProgramParameter(pointProgram, gl.LINK_STATUS)) {
                    console.error("Unable to initialize the shader program.");
                    return;
                }

                gl.useProgram(pointProgram);

                if (readyCallback != null) {
                    readyCallback();
                }
            }
        });
    });
}

function resize() {
    var width = canvasLayer.canvas.width;
    var height = canvasLayer.canvas.height;
    gl.viewport(0, 0, width, height);

    // matrix which maps pixel coordinates to WebGL coordinates
    pixelsToWebGLMatrix.set([2 / width, 0, 0, 0, 0, -2 / height, 0, 0,
        0, 0, 0, 0, -1, 1, 0, 1
    ]);
}


function update() {
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (POINT_COUNT > 0) {
        var mapProjection = map.getProjection();
        /**
         * We need to create a transformation that takes world coordinate
         * points in the pointArrayBuffer to the coodinates WebGL expects.
         * 1. Start with second half in pixelsToWebGLMatrix, which takes pixel
         *     coordinates to WebGL coordinates.
         * 2. Scale and translate to take world coordinates to pixel coords
         * see https://developers.google.com/maps/documentation/javascript/maptypes#MapCoordinate
         */

        // copy pixel->webgl matrix
        mapMatrix.set(pixelsToWebGLMatrix);

        // Scale to current zoom (worldCoords * 2^zoom)
        var scale = Math.pow(2, map.zoom);
        scaleMatrix(mapMatrix, scale, scale);

        // translate to current view (vector from topLeft to 0,0)
        var offset = mapProjection.fromLatLngToPoint(canvasLayer.getTopLeft());
        translateMatrix(mapMatrix, -offset.x, -offset.y);

        // attach matrix value to 'mapMatrix' uniform in shader
        var matrixLoc = gl.getUniformLocation(pointProgram, 'mapMatrix');
        gl.uniformMatrix4fv(matrixLoc, false, mapMatrix);

        // draw!
        gl.drawArrays(gl.POINTS, 0, POINT_COUNT);
    }
}

function scaleMatrix(matrix, scaleX, scaleY) {
    // scaling x and y, which is just scaling first two columns of matrix
    matrix[0] *= scaleX;
    matrix[1] *= scaleX;
    matrix[2] *= scaleX;
    matrix[3] *= scaleX;

    matrix[4] *= scaleY;
    matrix[5] *= scaleY;
    matrix[6] *= scaleY;
    matrix[7] *= scaleY;
}

function translateMatrix(matrix, tx, ty) {
    // translation is in last column of matrix
    matrix[12] += matrix[0] * tx + matrix[4] * ty;
    matrix[13] += matrix[1] * tx + matrix[5] * ty;
    matrix[14] += matrix[2] * tx + matrix[6] * ty;
    matrix[15] += matrix[3] * tx + matrix[7] * ty;
}

function displayData(data, numComponents, pointSize) {
    POINT_COUNT = data.length / numComponents;

    // create webgl buffer, bind it, and load rawData into it
    var pointArrayBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pointArrayBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

    // enable the 'worldCoord' attribute in the shader to receive buffer
    var attributeLoc = gl.getAttribLocation(self.pointProgram, 'worldCoord');
    gl.enableVertexAttribArray(attributeLoc);
    gl.vertexAttribPointer(attributeLoc, numComponents, gl.FLOAT, false, 0, 0);

    var textureSizeLocation = gl.getUniformLocation(self.pointProgram, "pointSize");
    gl.uniform2f(textureSizeLocation, pointSize, pointSize);

    refreshTimeInterval(0, 1.0);
}

function refreshTimeInterval(minValue, maxValue) {
    var timeInterval = gl.getUniformLocation(self.pointProgram, "timeInterval");
    gl.uniform2f(timeInterval, minValue, maxValue);

    update();
}

function LatLongToPixelXY(latitude, longitude) {
    var sinLatitude = Math.sin(latitude * this.pi_180);
    var pixelY = (0.5 - Math.log((1 + sinLatitude) / (1 - sinLatitude)) / (this.pi_4)) * 256;
    var pixelX = ((longitude + 180) / 360) * 256;
    var pixel = {
        x: pixelX,
        y: pixelY
    };
    return pixel;
}

function ready() {
    // If we don't have a GL context, give up now
    if (!gl) {
        alert("Unable to initialize WebGL. Your browser may not support it.");
        gl = null;
    } else {
        var n = noty({
            id: 'loading',
            text: "Loading Uber Ride data (about 65MB). Please wait...",
            layout: 'center',
            theme: 'relax'
        });

        d3.csv('http://christopheduong.github.io/data/SanFrancisco/all.csv', function(error, data) {
            if (error != null) {
                console.warn(error);
            } else {
                timeData = [];
                rawData = new Float32Array(3 * data.length);
                var i = 0;
                data.forEach(function(row) {
                    var point = LatLongToPixelXY(parseFloat(row.lat), parseFloat(row.lon));
                    rawData[0 + i * 3] = point.x;
                    rawData[1 + i * 3] = point.y;
                    rawData[2 + i * 3] = parseFloat(row.normTime);
                    timeData.push(row.normTime);
                    ++i;
                });
                n.close();
                displayHistogram();
                displayData(rawData, 3, 0.3);
            }
        });
    }
}

var brush = null;
var x = null;

function displayHistogram() {
    var values = timeData;

    var margin = {
            top: 10,
            right: 30,
            bottom: 30,
            left: 30
        },
        width = 1500 - margin.left - margin.right,
        height = 200 - margin.top - margin.bottom;

    x = d3.scale.linear()
        .domain([0, 1])
        .range([0, width]);

    brush = d3.svg.brush()
        .x(x)
        .on("brush", brushed);

    // Generate a histogram using twenty uniformly-spaced bins.
    var data = d3.layout.histogram()
        .bins(x.ticks(200))
        (values);

    var y = d3.scale.linear()
        .domain([0, d3.max(data, function(d) {
            return d.y;
        })])
        .range([height, 0]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom");

    var body = d3.select("body");

    body.on("keydown", onKeyPress);

    var svg = body.append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var bar = svg.selectAll(".bar")
        .data(data)
        .enter().append("g")
        .attr("class", "bar")
        .attr("transform", function(d) {
            return "translate(" + x(d.x) + "," + y(d.y) + ")";
        });

    bar.append("rect")
        .attr("x", 1)
        .attr("width", x(data[0].dx) - 1)
        .attr("height", function(d) {
            return height - y(d.y);
        });

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    svg.append("g")
        .attr("class", "x brush")
        .call(brush)
        .selectAll("rect")
        .attr("y", -6)
        .attr("height", height + 7);
}

function brushed() {
    if (brush != null && x != null) {
        var interval = brush.empty() ? x.domain() : brush.extent();
        refreshTimeInterval(interval[0], interval[1]);
    }
}

function onKeyPress() {
    if (brush != null) {
        var b = brush.extent();
        var key = d3.event.keyCode;
        var shift = 0.0001;
        if (key == 39) {
            d3.select(".brush").call(brush.extent([b[0] + shift, b[1] + shift]));
            brushed();
        } else if (key == 37) {
            d3.select(".brush").call(brush.extent([b[0] - shift, b[1] - shift]));
            brushed();
        } else if (key == 40) {
            d3.select(".brush").call(brush.extent([b[0] + shift * 10, b[1] + shift * 10]));
            brushed();
        } else if (key == 38) {
            d3.select(".brush").call(brush.extent([b[0] - shift * 10, b[1] - shift * 10]));
            brushed();
        } 
    }
}