xvs = 70*arange(-2.01, 2,.1)
yvs = 70*arange(-2.01, 2,.1)
zvs = arange(-500, 500, 100)

PsfFitCSIR.setModel(seriesName, None)

misf = zeros([len(xvs), len(yvs), len(zvs)])
m0 = PsfFitCSIR.f_Interp3d([1,0,0,-300,0], X, Y, Z)
for i in range(len(xvs)):
    for j in range(len(yvs)):
        for k in range(len(zvs)):
            misf[i,j, k] = ((m0 -PsfFitCSIR.f_Interp3d([1,xvs[i],yvs[j],zvs[k],0], X, Y, Z))**2).sum()