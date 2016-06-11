path = 'subsample/';
nclasses = 10;
folders = dir(path);

%images = cell(length(folders)-2,1);
Amat = [];
Labelmat = [];
for i = 3:length(folders)
    k = i-2;
    files = dir([path, folders(i).name,'/*.jpg']);
    images{k} = cell(length(files),1);
    for j=1:length(files)
        %images{k}{j} = 
        img = imread([path,folders(i).name,'/',files(j).name]);
        img = rgb2gray(img);
        Amat = [Amat reshape(img, [numel(img),1])];
        Labelmat = [Labelmat k];
    end
end

m = size(Amat,2);
nsqr = size(Amat,1);
for i=1:m
    Amat(:,i) = Amat(:,i) - mean(Amat,2)
end


mdash = int16(m/2);
[u,~] = eig(Amat'*Amat);
u = fliplr(u);
u = u(:,1:mdash);       %m x m'
eigvecs = Amat*u;           %n2 x m'

cv = cvpartition(m, 'Holdout');
XY = [Amat; Labelmat];
XYTrain = Amat(:,cv.training);
XYTest = Amat(:,cv.test);

classreps = zeros(nn,nclasses);
for i=1:nclasses
    classreps(i)= mean(XTrain(:,XTrain(end,:) == i,1:end-1),2);
end

omegaclasses = eigenvecs'*classreps;


TestYs = size(1, XYTest);
TestpYs = size(1, XYTest);
for i=size(XYTest,2)
    omega = eigenvecs'* XYTest(1:end-1,i);  %m' x 1
    TestYs(i) = XYTest(end,i);
    d = zeros(nClasses,1);
    for j = 1:nClasses
        d(j) = pdist([omega' ; omegaclasses(:,j)']);
    end
    [~,TestpYs(i) ] = min(d);
end

confusionmat(TestYs, TestpYs);