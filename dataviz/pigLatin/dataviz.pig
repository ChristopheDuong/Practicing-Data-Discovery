
raw_bus = LOAD 'bus.csv' using PigStorage(',') as (
	stopTag: chararray,
	busLineTag: chararray,
	lat: float,
	lon: float,
	title: chararray,
	direction: chararray,
	stopOrder: int
);
/* remove csv header row... any better ways? */
raw_bus = FILTER raw_bus by stopTag != 'stopTag';
bus = FOREACH raw_bus GENERATE busLineTag, lat, lon;

raw_trips = LOAD 'trips.csv' using PigStorage(',') as (
	tripId: int,
	duration: chararray,
	endLat: float,
	endLon: float,
	startLat: float,
	startLon: float,
	startTime: chararray,
	dayofweek: int,
	normTime: float
);
/* remove csv header row... any better ways? */
raw_trips = FILTER raw_trips by duration != 'duration';
trips = FOREACH raw_trips GENERATE tripId, startLat, startLon, endLat, endLon;

clean_bus = FILTER bus by (
	lat is not null and
	lon is not null);
clean_trips = FILTER trips by (
	startLat is not null and 
	startLon is not null and 
	endLat is not null and 
	endLon is not null);
/* produce all pairs between bus stops and trips points */
bus_trips = CROSS clean_trips, clean_bus;
/* compute distance between each bus stop to each trip start and end points */
distanced = FOREACH bus_trips GENERATE
	busLineTag, tripId, 
	(SQRT((lat - startLat)*(lat - startLat) + 
		(COS(lat*3.14159265359/180)*(lon - startLon)) * 
		(COS(lat*3.14159265359/180)*(lon - startLon)))) as distToStart,
	(SQRT((lat - endLat)*(lat - endLat) + 
		(COS(lat*3.14159265359/180)*(lon - endLon)) *
		(COS(lat*3.14159265359/180)*(lon - endLon)))) as distToEnd;
grouped = GROUP distanced BY (tripId, busLineTag);
shortest = FOREACH grouped GENERATE 
	flatten(group), 
	MIN(distanced.distToStart) as startBus,
	MIN(distanced.distToEnd) as endBus;
nearest = FOREACH  shortest GENERATE 
	busLineTag, tripId, 
	(startBus + endBus) as totalDistance;
ordered = ORDER nearest BY totalDistance ASC;

/* remove folder from hdfs and store a new copy of the results */
fs -rm -r -f nearest
store ordered into 'nearest';