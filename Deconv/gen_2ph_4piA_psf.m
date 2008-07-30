function h = gen_2ph_4piA_psf(size, voxelsize, ex_lambda, em_lambda, ex_na, em_na, n, pinhole, use4pi)

o = kSimPSF( {'lambdaEm',ex_lambda;'Pi4Em',use4pi;'na',ex_na;'ri',n;'sX',size(1);'sY',size(2);'sZ',size(3);'scaleX',voxelsize.x*1e3;'scaleY',voxelsize.y*1e3;'scaleZ',voxelsize.z*1e3;'lambdaEx',488;'pinhole',1;'confocal',0;'nonorm',0;'Pi4Ex',0;'computeASF',0;'circPol',0;'scalarTheory',0;'o',''});

o = o.^2;

q = kSimPSF( {'lambdaEm',em_lambda;'Pi4Em',0;'na',em_na;'ri',n;'sX',size(1);'sY',size(2);'sZ',size(3);'scaleX',voxelsize.x*1e3;'scaleY',voxelsize.y*1e3;'scaleZ',voxelsize.z*1e3;'lambdaEx',488;'pinhole',1;'confocal',0;'nonorm',0;'Pi4Ex',0;'computeASF',0;'circPol',0;'scalarTheory',0;'o',''});

[X, Y,Z] = meshgrid(((1:size(1))-size(1)/2 -1)*voxelsize.x,((1:size(2))-size(2)/2 -1)*voxelsize.y,((1:size(3))-size(3)/2 -1).*voxelsize.z);

g = ((X.^2 + Y.^2) < (pinhole*1.22*.488/(2*1.35)).^2).*(abs(Z) < voxelsize.z);

g = g./sum(g(:));

%dip_image(g)

%q = ifftn(fftn(single(q)).*fftn(g));

h = single(o).*q;

h = h./max(h(:));