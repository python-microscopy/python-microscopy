#include <metal_stdlib>

using namespace metal;
kernel void vect_polar_feat_3d(const device int *uniform [[ buffer(0) ]],
                const device float *ax1 [[ buffer(1) ]],const device float *ay1 [[ buffer(2) ]],const device float *az1 [[ buffer(3) ]],
                const device float *x2 [[ buffer(4) ]],const device float *y2 [[ buffer(5) ]],const device float *z2 [[ buffer(6) ]],
                device int  *out [[ buffer(7) ]],
                uint j [[ thread_position_in_grid ]]) 
{
    
    int n_bins_r = uniform[0];
    int n_bins_angle = uniform[1];
    int n_bins_phi = n_bins_angle/2;
    float rBinSize = (float) 1.0/uniform[2]; 
    int x2_len = uniform[3];
    float rBinAngle = ((float)n_bins_angle)/(2*M_PI_F);
    
    int unwrapped_size = n_bins_r*n_bins_angle*n_bins_phi;
    
    float x1 = (float) ax1[j];
    float y1 = (float) ay1[j];
    float z1 = (float) az1[j];
    
    device int *outj = &(out[j*unwrapped_size]); 

    for (int i2 = 0; i2 < x2_len; i2++)
    {

        //calculate the delta
        float dx = x1 - (float)x2[i2];
        float dy = y1 - (float)y2[i2];
        float dz = z1 - (float)z2[i2];

        //and the distance
        float d = sqrt(dx*dx + dy*dy + dz*dz);

        //calculate angles
        float theta = atan2(dy, dx) + M_PI_F;
        float phi = atan(dz/sqrt(dx*dx + dy*dy)) + M_PI_2_F;

        //convert distance to bin index
        int r_id = (int)(d*rBinSize);
        int id = r_id*n_bins_angle*n_bins_phi + n_bins_phi*(int)(theta*rBinAngle) + (int)(phi*rBinAngle);

        if ((id < unwrapped_size) && (id >= 0)) outj[id] +=1;
        
    }

}