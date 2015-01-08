function init() {
    d3.csv("sample.csv", function(error, data) {
        if (error != null) {
            console.warn(error);
            console.log(data);
        } else {

        }
    });
}