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
            text: "Loading Uber trips and San Francisco Bus data (about 6MB). Please wait...",
            layout: 'center',
            theme: 'relax'
        });
        var file = '../../data/SanFrancisco/distance.tsv';
        d3.tsv(file, function(error, distData) {
            if (error != null) {
                console.warn(error);
            } else {
                nearby = d3.map();
                busSet = d3.set(['30', '22']);
                distData.forEach(function(row) {
                    // if busline of interest
                    if (busSet.has(row.busline)) {
                        nearby.set(row.tripId, row.distance);
                    }
                });
                file = '../../data/SanFrancisco/trips.csv';
                d3.csv(file, function(error, data) {
                    if (error != null) {
                        console.warn(error);
                    } else {
                        timeData = [];
                        rawData = new Float32Array(2 * 4 * data.length);
                        var i = 0;
                        var count = 0;
                        data.forEach(function(row) {
                            var point = LatLongToPixelXY(parseFloat(row.startLat), parseFloat(row.startLon));
                            rawData[0 + i * 4] = point.x;
                            rawData[1 + i * 4] = point.y;
                            rawData[2 + i * 4] = parseFloat(row.normTime);
                            if (nearby.has(row.tripId)) {
                                rawData[3 + i * 4] = 1.0;
                                ++count;
                            } else {
                                rawData[3 + i * 4] = 0.0;
                            }
                            timeData.push(row.normTime);
                            ++i;
                            point = LatLongToPixelXY(parseFloat(row.endLat), parseFloat(row.endLon));
                            rawData[0 + i * 4] = point.x;
                            rawData[1 + i * 4] = point.y;
                            rawData[2 + i * 4] = parseFloat(row.normTime);
                            if (nearby.has(row.tripId)) {
                                rawData[3 + i * 4] = 1.0;
                                ++count;
                            } else {
                                rawData[3 + i * 4] = 0.0;
                            }
                            timeData.push(row.normTime);
                            ++i;
                        });
                        n.close();
                        displayHistogram();
                        displayData(rawData, componentsNb, pointSize);
                        console.log(count * 100.0 / i);
                    }
                });
            }
        });
    }
}