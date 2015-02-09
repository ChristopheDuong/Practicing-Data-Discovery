// Setup Google Map View
var googleOpt = {
    zoom: 13,
    center: new google.maps.LatLng(37.77926, -122.41934),
    mapTypeId: google.maps.MapTypeId.HYBRID
};

var shaderFragment = "shader/point.frag";
var shaderVertex = "shader/point.vert";

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
        var file = '../../data/SanFrancisco/all.csv';
        d3.csv(file, function(error, data) {
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