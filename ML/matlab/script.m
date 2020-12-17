% ref1
m = csvread('data/ref1.csv', 1, 0);
cmd_m = zeros(size(m, 1), 1);

for row = 1 : size(m, 1)
    currLat = deg2km(m(row, 1)) * 1000;
    currLon = deg2km(m(row, 2)) * 1000;
    currCrs = m(row, 4);
    v = m(row, 3);
    refLat = deg2km(m(row, 5)) * 1000;
    refLon = deg2km(m(row, 6)) * 1000;
    refCrs = m(row, 7);
    refPose = [refLon, refLat, refCrs];
    currPose = [currLon, currLat, currCrs];
    steerCmd = lateralControllerStanley(refPose, currPose, v);
    cmd_m(row, 1) = steerCmd;
end

modified_m = [m cmd_m];
table = array2table(modified_m, 'VariableNames', {'LAT', 'LON', 'GSPEED', 'CRS', 'NLAT', 'NLON', 'NCRS', 'CMD'});
writetable(table, 'data/ref1_v2.csv');


% traces1
m = csvread('data/traces1.csv', 1, 0);
cmd_m = zeros(size(m, 1), 1);

for row = 1 : size(m, 1)
    currLat = deg2km(m(row, 1)) * 1000;
    currLon = deg2km(m(row, 2)) * 1000;
    currCrs = m(row, 4);
    v = m(row, 3);
    refLat = deg2km(m(row, 5)) * 1000;
    refLon = deg2km(m(row, 6)) * 1000;
    refCrs = m(row, 7);
    refPose = [refLon, refLat, refCrs];
    currPose = [currLon, currLat, currCrs];
    steerCmd = lateralControllerStanley(refPose, currPose, v);
    cmd_m(row, 1) = steerCmd;
end

modified_m = [m cmd_m];
table = array2table(modified_m, 'VariableNames', {'LAT', 'LON', 'GSPEED', 'CRS', 'NLAT', 'NLON', 'NCRS', 'CMD'});
writetable(table, 'data/traces1_v2.csv');


% ref2
m = csvread('data/ref2.csv', 1, 0);
cmd_m = zeros(size(m, 1), 1);

for row = 1 : size(m, 1)
    currLat = deg2km(m(row, 1)) * 1000;
    currLon = deg2km(m(row, 2)) * 1000;
    currCrs = m(row, 4);
    v = m(row, 3);
    refLat = deg2km(m(row, 5)) * 1000;
    refLon = deg2km(m(row, 6)) * 1000;
    refCrs = m(row, 7);
    refPose = [refLon, refLat, refCrs];
    currPose = [currLon, currLat, currCrs];
    steerCmd = lateralControllerStanley(refPose, currPose, v);
    cmd_m(row, 1) = steerCmd;
end

modified_m = [m cmd_m];
table = array2table(modified_m, 'VariableNames', {'LAT', 'LON', 'GSPEED', 'CRS', 'NLAT', 'NLON', 'NCRS', 'CMD'});
writetable(table, 'data/ref2_v2.csv');


% traces2-1
m = csvread('data/traces2-1.csv', 1, 0);
cmd_m = zeros(size(m, 1), 1);

for row = 1 : size(m, 1)
currLat = deg2km(m(row, 1)) * 1000;
currLon = deg2km(m(row, 2)) * 1000;
currCrs = m(row, 4);
v = m(row, 3);
refLat = deg2km(m(row, 5)) * 1000;
refLon = deg2km(m(row, 6)) * 1000;
refCrs = m(row, 7);
refPose = [refLon, refLat, refCrs];
currPose = [currLon, currLat, currCrs];
steerCmd = lateralControllerStanley(refPose, currPose, v);
cmd_m(row, 1) = steerCmd;
end

modified_m = [m cmd_m];
table = array2table(modified_m, 'VariableNames', {'LAT', 'LON', 'GSPEED', 'CRS', 'NLAT', 'NLON', 'NCRS', 'CMD'});
writetable(table, 'data/traces2-1_v2.csv');


% traces2-2
m = csvread('data/traces2-2.csv', 1, 0);
cmd_m = zeros(size(m, 1), 1);

for row = 1 : size(m, 1)
    currLat = deg2km(m(row, 1)) * 1000;
    currLon = deg2km(m(row, 2)) * 1000;
    currCrs = m(row, 4);
    v = m(row, 3);
    refLat = deg2km(m(row, 5)) * 1000;
    refLon = deg2km(m(row, 6)) * 1000;
    refCrs = m(row, 7);
    refPose = [refLon, refLat, refCrs];
    currPose = [currLon, currLat, currCrs];
    steerCmd = lateralControllerStanley(refPose, currPose, v);
    cmd_m(row, 1) = steerCmd;
end

modified_m = [m cmd_m];
table = array2table(modified_m, 'VariableNames', {'LAT', 'LON', 'GSPEED', 'CRS', 'NLAT', 'NLON', 'NCRS', 'CMD'});
writetable(table, 'data/traces2-2_v2.csv');
