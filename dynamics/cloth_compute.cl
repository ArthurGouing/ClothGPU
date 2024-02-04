__kernel void step(__global float *positions,
                   __global float *velocity, 
                   __global float *all_l0,
                   __global int *neighbours,
                   const float dt,
                   const float E,
                   const float k,
                   const float r)
{
    // Parameters
    size_t v_id = 3 * get_global_id(0);
    float3 sphere_center = (float3)(0);

    // Fetch values from global memory
    float3 pos   = (float3)(positions[v_id], positions[v_id+1], positions[v_id+2]);
    float3 vel   = (float3)(velocity[v_id],  velocity[v_id+1],  velocity[v_id+2]);
    // Create force
    float3 force;

    // Add volumic forces
    force = (float3)(0.f, 0.f, -9.81f);

    // Add surfacic forces
    float dist = distance(pos, sphere_center);
    if (dist < r)
    {
        float3 dir = normalize(pos-sphere_center);
        force += E * sqrt(r) * (r-dist)*sqrt(r-dist) * dir;
    }

    // Add internal forces
    short n_neighbours = 6;
    for (size_t i=0; i<n_neighbours; i++)
    {
        size_t neigh_id = 3 * neighbours[n_neighbours*get_global_id(0) + i];
        if (neigh_id!=-3) {
            float3 neighbour_pos = (float3)(positions[neigh_id], positions[neigh_id+1], positions[neigh_id+2]);
            float l0 = all_l0[n_neighbours*get_global_id(0) + i];
            float l = length(neighbour_pos - pos);
            float3 dir = (neighbour_pos - pos)/l;
            force += k * (l-l0) * dir;
        }
    }

    // Update values
    float alpha = 0.001;
    vel += dt * force - alpha*vel;
    pos += dt * vel;

    // Update values in global memory
    if (v_id!=10)
    {
        positions[v_id]   = pos.x;
        positions[v_id+1] = pos.y;
    }
    positions[v_id+2] = pos.z;
    velocity[v_id]   = vel.x;
    velocity[v_id+1] = vel.y;
    velocity[v_id+2] = vel.z;
    }