def rotVects(x1,y1, xc, yc, theta):
    dx = xc - x1
    dy = yc - y1
    sx = 2*sin(theta/2)*(sin(theta/2)*dx - cos(theta/2)*dy)
    sy = 2*sin(theta/2)*(cos(theta/2)*dx + sin(theta/2)*dy)
    return (sx, sy)
