#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> 
#endif

__constant int samples = 3000;
__constant int dim = 2;

//*******************************************************************************
//  # Function    : max_value_calc
//  # Description : Function to calculate max value from dataframe
//  # Input       : dataframe
//  # Output      : max_value
//*******************************************************************************/

__kernel void max_value_calc(__global float *dataframe, __global float *max_value)
{
    size_t j = get_global_id(0);
    float max = 0.0;

    for (int i = 0; i < samples; i++)
    {
        if (max < dataframe[(i*dim) + j])
        {
            max = dataframe[(i*dim) +j];
        }
    }
    max_value[j] = max;
}

//*******************************************************************************
//  # Function    : clusters_initialization
//  # Description : Function to initialise clusters.
//  # Input       : max_value
//  # Output      : Buffer containing clusters initial values: cluster_init
//*******************************************************************************/

__kernel void clusters_initialization(__global float *cluster_init, __global float *max_value)
{
    int j;
    size_t i = get_global_id(0);
    int rand = i+1;

    for(j = 0; j < dim; j++)
    {
        cluster_init[(i*dim) + j] = (max_value[j] * (rand^7)) / ((rand+1)^13);
    }
    
}

//*******************************************************************************
//  # Function    : calculate_distance
//  # Description : Function to calculate distance.
//  # Input       : dataframe, clusters
//  # Output      : Buffer containing distance values: d_output2,distances
//*******************************************************************************/

__kernel void calculate_distance(__global float *dataframe, __global float *clusters, 
                                __global float *distances)
{
    size_t i = get_global_id(0);
    size_t num_clusters = get_global_size(0);
    int j, k;

    for(j = 0; j < samples; j++)
    {
        float distance = 0;
        for(k = 0; k < dim; k++)
        {
            distance = distance + ((dataframe[(j*dim) + k] - clusters[(i*(dim)) + k])
            * (dataframe[(j*dim) + k] - clusters[(i*(dim)) + k]));
        }
        distances[(j*num_clusters) + i] = distance/dim;
    }
}

//*******************************************************************************
//  # Function    : assign_cluster
//  # Description : Function to assign clusters to each sample point
//  # Input       : dataframe, distances
//  # Output      : Assigned cluster: cluster_number
//*******************************************************************************/

__kernel void assign_cluster(__global float *dataframe, __global float *distances,
                             __global float *clusters,__global float *cluster_number, __global int *changed)
{
    size_t id = get_global_id(0);
    int i, j, min, cluster, k;
    int count = 0;
    size_t num_clusters = get_global_size(0);
    float new_cluster[dim];

    for(i = 0; i < dim; i++)
    {
        new_cluster[i] = 0;
    }

    for(i = 0; i < samples; i++)
    {
        min = distances[(i*num_clusters)];
        cluster = 0;
        for(j = 0; j < num_clusters; j++)
        {
            if(min > distances[(i*num_clusters) + j])
            {
                min = distances[(i*num_clusters) + j];
                cluster = j;
            }
        }
        if(cluster == id)
        {
            cluster_number[i] = cluster;
            for(k = 0; k < dim; k++)
            {
                new_cluster[k] += dataframe[(i*dim) + k];
            }
            count++;
        }
    }

    int flag = 0;
    for(i = 0; i < dim; i++)
    {
        if(count != 0)
        {
            new_cluster[i] /= count;
        }
        else
        {
            new_cluster[i] = 100;
        }
        if(new_cluster[i] != clusters[(id*(dim)) + i])
        {
            flag = 1;
        }           
        clusters[(id*(dim)) + i] = new_cluster[i];
    }
    changed[id] = flag;
}