% okay lets do this
stepSize = .01;
index = [0:stepSize:2*pi];
memlen = length(index);
index = [0:stepSize:2*pi];

% distance between estimator samples
sampleSpread = 1; 
%
decay = 1;
%
weightDecay = 1;
% sw = sin(index)+2*cos(index*2+pi/2);
% sw = sin(index);

% sw = sin(index) .* sin(1/4 * index)...
% + cos(index*10) - sin(index*4 -pi/3)...
% + log(index+1);

sw = index;
% sw = sw + .5*randn(1,length(index));
% sw = 1*randn(1,length(index));
% sw = [sw,sw,sw,sw,sw,sw,sw];
dsw = discreteDerivative(sw);

% learn the conditional expectation...

[mx,mn,md,ad] = characterize(sw);
[dmx,dmn,dmd,dad] = characterize(dsw);

% ce = load("sinCE.mat");
% condExp = ce.condExp;
condExp = trainCE(mx,mn,md,ad,memlen,sw,decay,sampleSpread);
% dce = trainCE(mx,mn,md,ad,memlen,dsw,decay);

% plot(1:length(condExp(1,:)),condExp(1,:))

% displayCE(condExp);
dce3d(condExp);

swindow = sw;
% sw = index;
% now predict
while true
% for i = [1:10]
	% swindow = dispGen(sw,memlen,mx,mn,ad,weightDecay,condExp);
	% plot(1:length(swindow),swindow);

	% dsw = dispGen(dsw,memlen,mx,mn,ad,weightDecay,dce);
	% plot(1:length(dswindow),dswindow);
	figure(1);
	swindow = [swindow(2:length(swindow)),nextStep(swindow,memlen,mx,mn,ad,weightDecay,condExp,sampleSpread)];
	sw = [sw(1:length(sw)),nextStep(sw,memlen,mx,mn,ad,weightDecay,condExp,sampleSpread)];
	% plot(1:length(swindow),swindow);
	plot(1:length(sw),sw);

	pause(.00001);

	% figure(2);
	% plot(1:length(discreteDerivative(sw)),discreteDerivative(sw));
	% dswindow = [dsw(2:length(dsw)),nextStep(dsw,memlen,dmx,dmn,dad,weightDecay,dce)];
	% dsw = [dsw(1:length(dsw)),nextStep(dsw,memlen,dmx,dmn,dad,weightDecay,dce)];
	% plot(1:length(dswindow),dswindow);
	% plot(1:length(discreteDerivative(sw)),discreteDerivative(sw));

end

function dce3d(ce)

	figure(3)
	hold on 
	for i = 1:size(ce,1)
        for j = 1:size(ce,2)
            plot3(i,j,ce(i,j)); 
        end
    end
    hold off
end

function displayCE(ce)

	figure(2)
	hold on
	for i = 1:size(ce,1)

		plot(ce(i,:))

	end
	hold off
end

function window = dispGen(sw,memlen,mx,mn,ad,weightDecay,condExp)
	% for i = [1:1000]
	% while true

	swindow = [sw(2:length(sw)),nextStep(sw,memlen,mx,mn,ad,weightDecay,condExp)];
	sw = [sw(1:length(sw)),nextStep(sw,memlen,mx,mn,ad,weightDecay,condExp)];
	% plot(1:length(swindow),swindow);
	% plot(1:length(sw),sw);

	window = sw;


end


function condExp = trainCE(mx,mn,md,ad,memlen,sw,decay,sampleSpread)

	condExp = zeros(length([0:ad:mx-mn]),memlen);
	counter = condExp;

	for i = [1:length(sw)]

		% new value is read in
		newval = sw(i);
		% iterate backwards through the layers
		for j = [1:sampleSpread:memlen*sampleSpread]

			if (j < i)
			% update the expectation at that value (scaling)
				ce = condExp(idxGet(sw(i-j),mx,mn,ad),(j-1+sampleSpread)/sampleSpread);
				ce = ce * counter(idxGet(sw(i-j),mx,mn,ad),(j-1+sampleSpread)/sampleSpread);
				counter(idxGet(sw(i-j),mx,mn,ad),(j-1+sampleSpread/sampleSpread)) = counter(idxGet(sw(i-j),mx,mn,ad),(j-1+sampleSpread)/sampleSpread) + 1;
				ce = ce*decay + sw(i);
				ce = ce / counter(idxGet(sw(i-j),mx,mn,ad),(j-1+sampleSpread)/sampleSpread);
				condExp(idxGet(sw(i-j),mx,mn,ad),(j-1+sampleSpread)/sampleSpread) = ce;

			end

		end

	end

end


function dd = discreteDerivative(v)

    dd = [];

    for i = 1:length(v)-1
        
        dd = [dd,v(i)-v(i+1)];
        
    end
    

end


% average (weighted) the expectations
function avgVal = nextStep(sw,memlen,mx,mn,ad,weightDecay,condExp,sampleSpread)

	ws = fliplr(sw);
	nextVal = [];
	avgVal = 0;

	for i = 1:sampleSpread:memlen*sampleSpread

		nextVal = [nextVal,condExp(idxGet(ws(i),mx,mn,ad),i,sampleSpread)];

		avgVal = avgVal + nextVal(i) / (weightDecay^i);

	end

	avgVal = avgVal/memlen;

	%nextVal = mean(nextVal)

end

function idx = idxGet(val,mx,mn,ad)

	vals = [mn:ad:mx];
	va = vals*0+val;
	[m,idx] = min(abs(vals-va));

end

function mindiff = minimumDifference(sw)

	mindiff = abs(max(sw)-min(sw));

	for i = [1:length(sw)-1]

		if (abs(sw(i) - sw(i+1)) < mindiff)
			mindiff = abs(sw(i) - sw(i+1));
		end

	end

end

function avgdiff = avgDifference(sw)

	avgdiff = abs(max(sw)-min(sw));

	for i = [1:length(sw)-1]

		avgdiff = avgdiff + abs(sw(i) - sw(i+1));

	end

	avgdiff = avgdiff / (length(sw)-1);

end


function [maximum,minimum,minDiff,avgDiff] = characterize(sw)

	maximum = max(sw);
	minimum = min(sw);

	minDiff = minimumDifference(sw);

	avgDiff = avgDifference(sw);

end

