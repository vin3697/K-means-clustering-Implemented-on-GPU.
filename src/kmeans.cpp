#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>


constexpr auto Num_Cluster = 20;
constexpr auto N_Samples = 3000;
constexpr auto Dimension = 2;

//////////////////////////////////////////////////////////////////////////////
/* Write Result to File */
//////////////////////////////////////////////////////////////////////////////
void write_CPU_results_to_file(float* cluster_number, int nsamples)
{
    std::ofstream result_file;
    result_file.open("C:/University/Course_Subjects/Semester_3/Lab/Project_K_Means_Clustering/Results/Kmeans_CPU_result.csv");
    int i;
    result_file << "Point,Cluster\n";
    for (i = 0; i < nsamples; i++)
    {
        result_file << i + 1 << "," << cluster_number[i] << "\n";
    }
}

void write_GPU_results_to_file(float* cluster_number, int nsamples)
{
    std::ofstream result_file;
    result_file.open("C:/University/Course_Subjects/Semester_3/Lab/Project_K_Means_Clustering/Results/Kmeans_GPU_result.csv");
    int i;
    result_file << "Point,Cluster\n";
    for (i = 0; i < nsamples; i++)
    {
        result_file << i + 1 << "," << cluster_number[i] << "\n";
    }
}

//////////////////////////////////////////////////////////////////////////////
/* Get Data from CSV for GPU */
//////////////////////////////////////////////////////////////////////////////
float* csv_to_float_matrix_GPU()
{
    std::ifstream csv_file;
    csv_file.open("C:/University/Course_Subjects/Semester_3/Lab/Project_K_Means_Clustering/data/dataset.csv", std::ios::in);
    char str[210];
    int start = 0, end, row_num = 0, col_num, first = 1;
    std::string row_values, substr, delim;
    int nsamples = N_Samples, dims = Dimension;
    float* feature_matrix = new float[nsamples * dims];
    while (csv_file.getline(str, 210))
    {
        if (first)
        {
            first = 0;
            continue;
        }
        row_values = str;
        delim = ",";
        start = 0;
        end = row_values.find(delim);
        col_num = 0;
        while (end != std::string::npos)
        {
            substr = row_values.substr(start, end - start);
            feature_matrix[(row_num * dims) + col_num] = stof(substr);
            col_num++;
            start = end + delim.length();
            end = row_values.find(delim, start);
        }
        feature_matrix[(row_num * dims) + col_num] = stof(row_values.substr(start, end));
        row_num++;
        col_num++;
    }
    return feature_matrix;
}

//////////////////////////////////////////////////////////////////////////////
/* CPU implementation */
//////////////////////////////////////////////////////////////////////////////
float euclidean_distance(float point[], float centre[], int d)
{
    int i;
    float distance = 0;
    for (i = 0; i < d; i++)
    {
        distance = distance + ((point[i] - centre[i]) * (point[i] - centre[i]));
    }
    return distance / d;
}

int assign_clusters(float distances[], int k)
{
    float min;
    min = distances[0];
    int i, cluster = 0;
    for (i = 1; i < k; i++)
    {
        if (min > distances[i])
        {
            cluster = i;
            min = distances[i];
        }
    }
    return cluster;
}

float** init_float_matrix(int rows, int cols)
{
    float** mat;
    mat = new float* [rows];
    for (int i = 0; i < rows; i++)
    {
        mat[i] = new float[cols];
    }
    return mat;
}

/*__________Get Data from CSV for GPU___________________*/
float** csv_to_float_matrix_CPU()
{
    std::ifstream csv_file;
    csv_file.open("C:/University/Course_Subjects/Semester_3/Lab/Project_K_Means_Clustering/data/dataset.csv", std::ios::in);
    char str[210];
    int start = 0, end, row_num = 0, col_num, first = 1;
    std::string row_values, substr, delim;
    float** feature_matrix = init_float_matrix(N_Samples, Dimension);
    while (csv_file.getline(str, 210))
    {
        if (first)
        {
            first = 0;
            continue;
        }
        row_values = str;
        delim = ",";
        start = 0;
        end = row_values.find(delim);
        col_num = 0;
        while (end != std::string::npos)
        {
            substr = row_values.substr(start, end - start);
            feature_matrix[row_num][col_num++] = stof(substr);
            start = end + delim.length();
            end = row_values.find(delim, start);
        }
        feature_matrix[row_num++][col_num++] = stof(row_values.substr(start, end));
    }
    return feature_matrix;
}

/*______ MAX Value function to return max values from samples _____________*/
float* max_values(float** feauture_matrix, int nsamples, int dims)
{
    float* max_vals = new float[dims];
    float max;
    for (int j = 0; j < dims; j++)
    {
        max = 0.0;
        for (int i = 0; i < nsamples; i++)
        {
            if (max < feauture_matrix[i][j])
            {
                max = feauture_matrix[i][j];
            }
        }
        max_vals[j] = max;
    }
    return max_vals;
}

/*______ Initial Clusters _____________*/
float** init_clusters(float* rand_max, int num_clusters, int dims)
{
    float** clusters = init_float_matrix(num_clusters, dims);
    for (int i = 0; i < num_clusters; i++)
    {
        for (int j = 0; j < dims; j++)
        {
            clusters[i][j] = rand_max[j] >= 1.0 ? (float)(rand() % (int)rand_max[j]) : 0.0;
        }
    }
    return clusters;
}

/*______ Cluster Assignments _____________*/
void kmeans_clustering(float** dataframe, float** clusters, int nsamples, int num_clusters, int dims)
{
    int changed = 1, i, j, cluster, inter = 0;
    float* cluster_number = (float*)malloc(N_Samples * sizeof(float));
    while (changed)
    {
        float distances[Num_Cluster], point_summation[Num_Cluster][Dimension + 1];
        for (i = 0; i < num_clusters; i++)
        {
            for (j = 0; j < dims + 1; j++)
            {
                point_summation[i][j] = 0;
            }
        }
        for (int i = 0; i < nsamples; i++)
        {
            for (j = 0; j < num_clusters; j++)
            {
                distances[j] = euclidean_distance(dataframe[i], clusters[j], dims);
            }
            cluster = assign_clusters(distances, num_clusters);
            cluster_number[i] = cluster;
            point_summation[cluster][0]++;
            for (j = 0; j < dims; j++)
            {
                point_summation[cluster][j + 1] += dataframe[i][j];
            }
        }
        changed = 0;
        for (i = 0; i < num_clusters; i++)
        {
            for (j = 0; j < dims; j++)
            {
                float new_value = point_summation[i][0] != 0 ? point_summation[i][j + 1] / point_summation[i][0] : 100;
                if (new_value != clusters[i][j])
                {
                    changed = 1;
                    clusters[i][j] = new_value;
                }
            }
        }

        inter++;
    }

    std::cout << "Number of iterations for assigment of cluster CPU:" << inter << std::endl;
    write_CPU_results_to_file(cluster_number, nsamples);
}


//////////////////////////////////////////////////////////////////////////////
/* Main function */
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    float** dataframe = csv_to_float_matrix_CPU();
    float* dataframe_GPU = csv_to_float_matrix_GPU();
    int nsamples = N_Samples, dims = Dimension, num_clusters = Num_Cluster;

    int changed = 1;
    int changed_flag[Num_Cluster];
    int inter = 0;

    /////////////////// CPU EXECUTION ////////////////////////////////////////

    double run_time_CPU = clock();
    float** cluster_centres = init_clusters(max_values(dataframe, nsamples, dims), num_clusters, dims);
    kmeans_clustering(dataframe, cluster_centres, nsamples, num_clusters, dims);
    run_time_CPU = (clock() - run_time_CPU) / CLOCKS_PER_SEC;
    std::cout << "CPU_TIME:" << run_time_CPU << std::endl;


    ////////////////// GPU EXECUTION /////////////////////////////////////////

    // Create a context	
    //cl::Context context(CL_DEVICE_TYPE_GPU);
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cerr << "No platforms found" << std::endl;
        return 1;
    }
    int platformId = 0;
    for (size_t i = 0; i < platforms.size(); i++) {
        if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
            platformId = i;
            break;
        }
    }
    cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
    std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
    cl::Context context(CL_DEVICE_TYPE_GPU, prop);


    // Get a device of the context
    int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
    std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
    ASSERT(deviceNr > 0);
    ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
    std::vector<cl::Device> devices;
    devices.push_back(device);
    OpenCL::printDeviceInfo(std::cout, device);

    // Create a command queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load the source code
    cl::Program program = OpenCL::loadProgramSource(context, "C:/University/Course_Subjects/Semester_3/Lab/Project_K_Means_Clustering/src/kmeans.cl ");
    // Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
    OpenCL::buildProgram(program, devices);

    // Create a kernel object
    cl::Kernel kernel(program, "max_value_calc");
    cl::Kernel kernel1(program, "clusters_initialization");
    cl::Kernel kernel2(program, "calculate_distance");
    cl::Kernel kernel3(program, "assign_cluster");

    // Declare some values
    std::size_t wgSize = 1;
    std::size_t samples = (N_Samples * sizeof(float));
    std::size_t No_Cluster = (Num_Cluster * sizeof(int));
    std::size_t dimension = ((Dimension) * sizeof(float));
    std::size_t data_size = (N_Samples * Dimension * sizeof(float));
    std::size_t matrix = (N_Samples * Num_Cluster * sizeof(float));
    std::size_t cluster = (Num_Cluster * (Dimension) * sizeof(float));// Size of data in bytes

    // Allocate space for output data from GPU on the host
    float* clusters = (float*)malloc(cluster);
    float* distances = (float*)malloc(matrix);
    float* max_value = (float*)malloc(dimension);
    float* assigned_cluster = (float*)malloc(samples);

    // Initialize memory to 0x0000
    memset(clusters, 0x0000, cluster);
    memset(distances, 0x0000, matrix);

    // Allocate space for input and output data on the device
    cl::Buffer input_data_frame(context, CL_MEM_READ_WRITE, data_size);
    cl::Buffer output_cluster(context, CL_MEM_READ_WRITE, No_Cluster);
    cl::Buffer output_max_value(context, CL_MEM_READ_WRITE, dimension);
    cl::Buffer cluster_init(context, CL_MEM_READ_WRITE, cluster);
    cl::Buffer output_distance_value(context, CL_MEM_READ_WRITE, matrix);
    cl::Buffer cluster_number(context, CL_MEM_READ_WRITE, samples);

    //Write Buffer
    cl::Event Write_Dataframe;
    queue.enqueueWriteBuffer(input_data_frame, true, 0, data_size, dataframe_GPU, NULL, &Write_Dataframe);

    // Launch kernel on the device for Max Value
    cl::Event Exec_Max_Value_Time;
    kernel.setArg<cl::Buffer>(0, input_data_frame);
    kernel.setArg<cl::Buffer>(1, output_max_value);
    queue.enqueueNDRangeKernel(kernel, 0, Dimension, wgSize, NULL, &Exec_Max_Value_Time);

    cl::Event Read_Max_Value;
    queue.enqueueReadBuffer(output_max_value, true, 0, dimension, max_value, NULL, &Read_Max_Value);

    // Launch kernel1 on the device for Cluster Initialization
    cl::Event Exec_Init_Cluster_Time;
    kernel1.setArg<cl::Buffer>(0, cluster_init);
    kernel1.setArg<cl::Buffer>(1, output_max_value);
    queue.enqueueNDRangeKernel(kernel1, 0, Num_Cluster, wgSize, NULL, &Exec_Init_Cluster_Time);

    // Copy output data back to host
    cl::Event Read_Cluster;
    queue.enqueueReadBuffer(cluster_init, true, 0, cluster, clusters, NULL, &Read_Cluster);

    // Events
    cl::Event Exec_Calc_Distance;
    cl::Event Exec_Assign_Cluster;
    cl::Event Read_Change_Flag;

    while (changed)
    {
        // Launch kernel2 on the device for Distance Calculation
        kernel2.setArg<cl::Buffer>(0, input_data_frame);
        kernel2.setArg<cl::Buffer>(1, cluster_init);
        kernel2.setArg<cl::Buffer>(2, output_distance_value);
        queue.enqueueNDRangeKernel(kernel2, 0, Num_Cluster, wgSize, NULL, &Exec_Calc_Distance);

        // Launch kernel3 on the device for Cluster Assignment
        kernel3.setArg<cl::Buffer>(0, input_data_frame);
        kernel3.setArg<cl::Buffer>(1, output_distance_value);
        kernel3.setArg<cl::Buffer>(2, cluster_init);
        kernel3.setArg<cl::Buffer>(3, cluster_number);
        kernel3.setArg<cl::Buffer>(4, output_cluster);
        queue.enqueueNDRangeKernel(kernel3, 0, Num_Cluster, wgSize, NULL, &Exec_Assign_Cluster);

        // Copy output data back to host
        queue.enqueueReadBuffer(output_cluster, true, 0, No_Cluster, changed_flag, NULL, &Read_Change_Flag);

        int i;
        changed = 0;
        for (i = 0; i < Num_Cluster; i++)
        {
            if (changed_flag[i])
            {
                changed = 1;
                break;
            }
        }

        inter++;
    }

    std::cout << "Number of iterations for assigment of cluster GPU:" << inter << std::endl;

    // Event to read assigned clusters
    cl::Event Read_Assigned_Cluster;
    queue.enqueueReadBuffer(cluster_number, true, 0, samples, assigned_cluster, NULL, &Read_Assigned_Cluster);

    //write GPU kmeans data to cvs
    write_GPU_results_to_file(assigned_cluster, nsamples);

    // Performance data
    Core::TimeSpan copy1 = OpenCL::getElapsedTime(Write_Dataframe);
    Core::TimeSpan copy2 = OpenCL::getElapsedTime(Read_Max_Value);
    Core::TimeSpan copy3 = OpenCL::getElapsedTime(Read_Cluster);
    Core::TimeSpan copy4 = OpenCL::getElapsedTime(Read_Change_Flag);
    Core::TimeSpan copy5 = OpenCL::getElapsedTime(Read_Assigned_Cluster);
    Core::TimeSpan exec1 = OpenCL::getElapsedTime(Exec_Max_Value_Time);
    Core::TimeSpan exec2 = OpenCL::getElapsedTime(Exec_Init_Cluster_Time);
    Core::TimeSpan exec3 = OpenCL::getElapsedTime(Exec_Calc_Distance);
    Core::TimeSpan exec4 = OpenCL::getElapsedTime(Exec_Assign_Cluster);
    Core::TimeSpan total_copy_time = copy1 + copy2 + copy3 + copy4 + copy5;
    Core::TimeSpan total_exec_time = exec1 + exec2 + exec3 + exec4;
    Core::TimeSpan overallGpuTime = (total_copy_time + total_exec_time);
    std::cout << "GPU Time: " << overallGpuTime.toString() << std::endl;

    return 0;
}