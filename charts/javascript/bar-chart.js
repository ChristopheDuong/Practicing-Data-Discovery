var chart;
var data;

function init(file) {
    d3.csv(file, function(error, d) {
        if (error != null) {
            console.warn(error);
            console.log(data);
        } else {
            data = d;
            chart = initPlot(extractColumnNames(data));
            setTimeout(function() {
                refreshData(convertCSVtoChartData(data));
            }, 1000);
        }
    });
}

// build the c3.js chart with the desired functionalities
function initPlot(features) {
    var chart = c3.generate({
        bindto: '#chart',
        data: {
            x: 'x',
            columns: [],
            order: null,
            type: 'bar'
        },
        axis: {
            rotated: true,
            x: {
                type: 'category',
                label: {
                    text: 'Variable Name',
                    position: 'outer-middle'
                }
            },
        },
        grid: {
            x: {
                show: false
            },
            y: {
                show: true,
                lines: [{
                    value: 0
                }]
            },
            focus: {
                show: false
            }
        },
        tooltip: {
            format: {
                title: function(d) {
                	var shown = chart.data.shown();
                	var name = Object.keys(data[d])[0];
                    return data[d][name] + " = " + sumValues(data[d], shown).toFixed(2);
                }
            }
        },
        legend: {
            item: {
                onclick: function(id) {
                    var $$ = this;
                    if ($$.d3.event.altKey) {
                        $$.api.hide();
                        $$.api.show(id);
                    } else {
                        $$.api.toggle(id);
                        $$.isTargetToShow(id) ? $$.api.focus(id) : $$.api.revert();
                    }
                    setTimeout(function() {
                        refreshData(convertCSVtoChartData(data));
                    }, 500);
                }
            }
        }
    });
    chart.groups(features);
    return chart;
}

// change the data that is displayed
function refreshData(data) {
    chart.load({
        columns: data
    });
}

// sum all selected features and compare totals
function comparePoints(a, b) {
    var shown = chart.data.shown();
    var aTotal = sumValues(a, shown);
    var bTotal = sumValues(b, shown);
    return aTotal < bTotal;
}

function isNumber(n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
}

function isShown(key, shown) {
    var result = false;
    for (var i = 0; i < shown.length; ++i) {
        if (shown[i]['id'] == key) {
            result = true;
        }
    }
    return result;
}

function sumValues(x, shown) {
    var total = 0;
    var keys = Object.keys(x);
    for (var i = 1; i < keys.length; ++i) {
        if (shown == null || shown.length == 0 || isShown(keys[i], shown)) {
            var key = keys[i];
            if (isNumber(x[key])) {
                total += parseFloat(x[key]);
            }
        }
    }
    return total;
}

// convert the data from CSV format into data format expected by C3.js
function convertCSVtoChartData(data) {
    var result = [];
    if (data != null && data.length > 0) {
        data.sort(comparePoints);
        var keys = Object.keys(data[0]);
        result.push(['x']);
        for (var i = 1; i < keys.length; ++i) {
            result.push([keys[i]]);
        }
        for (var i = 0; i < data.length; ++i) {
            for (var j = 0; j < keys.length; ++j) {
                result[j].push(data[i][keys[j]]);
            }
        }
    }
    return result;
}

// return the list of columns that should be stacked together (all features)
function extractColumnNames(data) {
    var result = [];
    if (data != null && data.length > 0) {
        var keys = Object.keys(data[0]);
        // first one is the category name, skip it
        for (var i = 1; i < keys.length; ++i) {
            result.push(keys[i]);
        }
    }
    return [result];
}