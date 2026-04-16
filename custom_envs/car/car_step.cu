// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause
#include <math.h>
__constant__ float kPi = 3.141592654;
__constant__ float kTwoPi = 6.283185308;
__constant__ float kEpsilon = 1.0e-10;  // to prevent indeterminate cases
__constant__ float square_root_2 = 1.41422;
extern "C" {
  // Device helper function to compute distances between two agents
  __device__ float ComputeAgentDistance(
    float * agent_size_arr,
    float * loc_x_arr,
    float * loc_y_arr,
    const int kThisAgentId1,
    const int kThisAgentId2,
    const int kEnvId,
    int kNumAgents,
    int * agent_types_arr
  ) {
    const int index1 = kEnvId * kNumAgents + kThisAgentId1;
    const int index2 = kEnvId * kNumAgents + kThisAgentId2;
    float min_dis = sqrt(
      pow(loc_x_arr[index1] - loc_x_arr[index2], 2) +
      pow(loc_y_arr[index1] - loc_y_arr[index2], 2)
    ) - agent_size_arr[kThisAgentId1] - agent_size_arr[kThisAgentId2];
    return min_dis;
  }

  __device__ float MapDx(
    float * grid_map_z,
    const float grid_length_max_y,
    const float grid_length_height,
    int map_x,
    int map_y
  ) {
    float point_dx;
    if (map_x < 1) point_dx = -1 * grid_length_height;
    else if(map_x >= grid_length_max_y - 1) point_dx = grid_length_height;
    else if(map_y > 0 && map_y < grid_length_max_y) 
    {
      int big_dx_id = grid_length_max_y * (map_x + 1) + map_y;
      int small_dx_id = grid_length_max_y * (map_x - 1) + map_y;
      point_dx = 0.5 * (grid_map_z[big_dx_id] - grid_map_z[small_dx_id]);
    }
    else point_dx = 0;
    return point_dx;
  }

  __device__ float MapDy(
    float * grid_map_z,
    const float grid_length_max_y,
    const float grid_length_height,
    int map_x,
    int map_y
  ) {
    float point_dy;
    if (map_y < 1) point_dy = -1 * grid_length_height;
    else if(map_y >= grid_length_max_y - 1) point_dy = grid_length_height;
    else if(map_x > 0 && map_x < grid_length_max_y) 
    {
      int big_dy_id = grid_length_max_y * map_x  + map_y + 1;
      int small_dy_id = grid_length_max_y * map_x  + map_y - 1;
      point_dy = 0.5 * (grid_map_z[big_dy_id] - grid_map_z[small_dy_id]);
    }
    else point_dy = 0;
    return point_dy;
  }

  __device__ void CudaTagContinuousGenerateObservation(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_x_arr,
    float * speed_y_arr,
    float * direction_arr,
    float * acceleration_x_arr,
    float * acceleration_y_arr,
    float * obs_arr,
    float * neighbor_distances_arr,
    float * agent_size_arr,
    float * grid_map_z,
    float * different_point_angle_distance,
    const float Training,
    const float kGridLength,
    const float kGridDensity,
    const float kNumOtherAgentsObserved_,
    const float spot_lidar_angle,
    const float kMaxSpeed_x,
    const float kMaxSpeed_y,
    const float MaxAcceleration_x,
    const float MaxAcceleration_y,
    const float grid_length_max_y,
    const float grid_length_height,
    int * still_in_the_game_arr,
    int * neighbor_ids_sorted_by_distance_arr,
    int * nearest_neighbor_ids,
    int * agent_types_arr,
    int * env_timestep_arr,
    const int kNumAgents,
    const int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {
    int num_features = 8;
    __sync_env_threads();
    if (kThisAgentId < kNumAgents) {
        
        int num_angle_lidar = int(360 / spot_lidar_angle);
        const int kNumOtherAgentsObserved = kNumOtherAgentsObserved_;
        const int kThisAgentIdxOffset = kEnvId * kNumAgents *
          (num_features * kNumOtherAgentsObserved + 20 + num_angle_lidar) +
          kThisAgentId * (num_features * kNumOtherAgentsObserved + 20 + num_angle_lidar);
        
        //printf("4");
        obs_arr[kThisAgentIdxOffset] = 0.0;  // acceleration x
        obs_arr[kThisAgentIdxOffset + 1] = 0.0;  // acceleration y
        obs_arr[kThisAgentIdxOffset + 2] = 0.0;  // speed x
        obs_arr[kThisAgentIdxOffset + 3] = 0.0;  // speed y
        obs_arr[kThisAgentIdxOffset + 4] = 0.0;  // dir
        obs_arr[kThisAgentIdxOffset + 5] = 0.0;  // size
        obs_arr[kThisAgentIdxOffset + 6] = 0.0;  // goal x
        obs_arr[kThisAgentIdxOffset + 7] = 0.0;  // gaol y
        obs_arr[kThisAgentIdxOffset + 8] = 0.0;  // gaol z
        obs_arr[kThisAgentIdxOffset + 9] = 0.0;  // agent x
        obs_arr[kThisAgentIdxOffset + 10] = 0.0;  // agent y
        obs_arr[kThisAgentIdxOffset + 11] = 0.0;  // front pitch
        obs_arr[kThisAgentIdxOffset + 12] = 0.0;  // front row
        obs_arr[kThisAgentIdxOffset + 13] = 0.0;  // back pitch
        obs_arr[kThisAgentIdxOffset + 14] = 0.0;  // back row
        obs_arr[kThisAgentIdxOffset + 15] = 0.0;  // left pitch
        obs_arr[kThisAgentIdxOffset + 16] = 0.0;  // left row
        obs_arr[kThisAgentIdxOffset + 17] = 0.0;  // right pitch
        obs_arr[kThisAgentIdxOffset + 18] = 0.0;  // right row
        obs_arr[kThisAgentIdxOffset + 19] = 0.0;  // time
        //printf("5");
        for (int idx = 0; idx < kNumOtherAgentsObserved; idx++) {
          obs_arr[kThisAgentIdxOffset + 20 + 0 * kNumOtherAgentsObserved + idx] = 0.0;  // other LOC_X
          obs_arr[kThisAgentIdxOffset + 20 + 1 * kNumOtherAgentsObserved + idx] = 0.0;  // other LOC_Y
          obs_arr[kThisAgentIdxOffset + 20 + 2 * kNumOtherAgentsObserved + idx] = 0.0;  // other SP_X
          obs_arr[kThisAgentIdxOffset + 20 + 3 * kNumOtherAgentsObserved + idx] = 0.0;  // other SP_Y
          obs_arr[kThisAgentIdxOffset + 20 + 4 * kNumOtherAgentsObserved + idx] = 0.0;  // other DIR
          obs_arr[kThisAgentIdxOffset + 20 + 5 * kNumOtherAgentsObserved + idx] = 0.0;  // other agent type
          obs_arr[kThisAgentIdxOffset + 20 + 6 * kNumOtherAgentsObserved + idx] = 0.0;  // other still in game
          obs_arr[kThisAgentIdxOffset + 20 + 7 * kNumOtherAgentsObserved + idx] = 0.0;  // other agent size
        }

        for (int angle = 0; angle < num_angle_lidar; angle++){
          obs_arr[kThisAgentIdxOffset + 20 + num_features * kNumOtherAgentsObserved + angle] = 0.0;  // local map infomation
        }
        
        // Update obs for agents still in the game
        if (still_in_the_game_arr[kThisAgentArrayIdx]) {
          int distance_arr_idx;
          int i_index;
          int j_index;
          int neighbor_ids_sorted_by_distance_arr_idx;

          // Find the nearest agents
          const int kThisAgentArrayIdxOffset = kEnvId * kNumAgents *
            (kNumAgents - 1) + kThisAgentId * (kNumAgents - 1);

          // Initialize neighbor_ids_sorted_by_distance_arr
          // other agents that are still in the same
          int num_valid_other_agents = 0;
          for (int other_agent_id = 0; other_agent_id < kNumAgents; other_agent_id++) {
            if ((other_agent_id != kThisAgentId) &&
                (still_in_the_game_arr[kEnvId * kNumAgents + other_agent_id])) {
              neighbor_ids_sorted_by_distance_arr_idx =
                kThisAgentArrayIdxOffset + num_valid_other_agents;
              neighbor_ids_sorted_by_distance_arr[
                neighbor_ids_sorted_by_distance_arr_idx] = other_agent_id;
              num_valid_other_agents++;
            }
          }
          //neighbor_ids_sorted_by_distance_arr have this env no sort but delete no in the game agent
          // First, find distances to all the valid agents
          for (int idx = 0; idx < num_valid_other_agents; idx++) {
            distance_arr_idx = kThisAgentArrayIdxOffset + idx;
            neighbor_distances_arr[distance_arr_idx] = ComputeAgentDistance(
              agent_size_arr,
              loc_x_arr,
              loc_y_arr,
              kThisAgentId,
              neighbor_ids_sorted_by_distance_arr[distance_arr_idx],
              kEnvId,
              kNumAgents,
              agent_types_arr);
          }
          

          // sort neighbor agent indices
          // add some "if" to let car agent at least see 1 goal 
          float goal_x_ = 0;
          float goal_y_ = 0;
          float goal_z_ = 0;

          int give_car_one_fix_goal = 1;
          bool check_car_can_look_goal = false;

          for (int i = 0; i < min(num_valid_other_agents,
            kNumOtherAgentsObserved); i++) {
            i_index = kThisAgentArrayIdxOffset + i;

            for (int j = i + 1; j < num_valid_other_agents; j++) {
              j_index = kThisAgentArrayIdxOffset + j;

              if (neighbor_distances_arr[j_index] <
                neighbor_distances_arr[i_index]) {
                float tmp1 = neighbor_distances_arr[i_index];
                neighbor_distances_arr[i_index] =
                    neighbor_distances_arr[j_index];
                neighbor_distances_arr[j_index] = tmp1;

                int tmp2 = neighbor_ids_sorted_by_distance_arr[i_index];
                neighbor_ids_sorted_by_distance_arr[i_index] =
                  neighbor_ids_sorted_by_distance_arr[j_index];
                neighbor_ids_sorted_by_distance_arr[j_index] = tmp2;
              }
            }
            if (agent_types_arr[neighbor_ids_sorted_by_distance_arr[i_index]] == 0) {
              check_car_can_look_goal = true;
            }
            if (i == kNumOtherAgentsObserved -1 &&
              check_car_can_look_goal == false &&
              agent_types_arr[kThisAgentId] == 1
            ){
              int k = num_valid_other_agents;
              for(int j = i + 1; j < k; j++) {
                j_index = kThisAgentArrayIdxOffset + j;
                if (agent_types_arr[neighbor_ids_sorted_by_distance_arr[j_index]] == 0){
                  k = 0;
                  neighbor_distances_arr[i_index] = neighbor_distances_arr[j_index];
                  neighbor_ids_sorted_by_distance_arr[i_index] = neighbor_ids_sorted_by_distance_arr[j_index];
                }
              }
            }
          }
          
          // Save nearest neighbor ids.
          for (int idx = 0; idx < min(num_valid_other_agents,
            kNumOtherAgentsObserved); idx++) {
            const int kNearestNeighborsIdx =
              kEnvId * kNumAgents * kNumOtherAgentsObserved +
              kThisAgentId * kNumOtherAgentsObserved +
              idx;
            nearest_neighbor_ids[kNearestNeighborsIdx] =
              neighbor_ids_sorted_by_distance_arr[kThisAgentArrayIdxOffset + idx];
          }
          //nearest_neighbor_ids have sort agent ids

          // Update observation
          for (int idx = 0; idx < min(num_valid_other_agents,
            kNumOtherAgentsObserved); idx++) {
            const int kNearestNeighborsIdx =
              kEnvId * kNumAgents * kNumOtherAgentsObserved +
              kThisAgentId * kNumOtherAgentsObserved +
              idx;
            const int kOtherAgentId = nearest_neighbor_ids[
              kNearestNeighborsIdx];
            const int kOtherAgentArrayIdx = kEnvId * kNumAgents + kOtherAgentId;
            int agent_types = 0;
            float agent_sizes = agent_size_arr[kOtherAgentId] + agent_size_arr[kThisAgentId] ;

            if (agent_types_arr[kOtherAgentId] == 0)
            {
              agent_types = 0;
            }
            else
            {
              agent_types = 1;
            }
       
            if (give_car_one_fix_goal == 1 && agent_types_arr[kOtherAgentId] == 0 && goal_x_ == 0 && goal_y_ == 0){
              give_car_one_fix_goal = 0;
              goal_x_ = static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] - loc_x_arr[kThisAgentArrayIdx]);
              goal_y_ = static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] - loc_y_arr[kThisAgentArrayIdx]);
              int other_map_x = int(loc_x_arr[kOtherAgentArrayIdx] / kGridDensity + 0.5);
              int other_map_y = int(loc_y_arr[kOtherAgentArrayIdx] / kGridDensity + 0.5);
              float other_map_z = grid_map_z[int(grid_length_max_y * other_map_x  + other_map_y)];
              int agent_map_x = int(loc_x_arr[kThisAgentArrayIdx] / kGridDensity + 0.5);
              int agent_map_y = int(loc_y_arr[kThisAgentArrayIdx] / kGridDensity + 0.5);
              float agent_map_z = grid_map_z[int(grid_length_max_y * agent_map_x  + agent_map_y)];

              goal_z_ = static_cast<float>(other_map_z - agent_map_z);

            }
            else if (idx == min(num_valid_other_agents,kNumOtherAgentsObserved) - 1 && give_car_one_fix_goal == 1 ){
              goal_x_ = static_cast<float>(0.0);
              goal_y_ = static_cast<float>(0.0);
              goal_z_ = static_cast<float>(0.0);
            }

            obs_arr[kThisAgentIdxOffset + 20 + 0 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] -
              loc_x_arr[kThisAgentArrayIdx]);
            obs_arr[kThisAgentIdxOffset + 20 + 1 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] -
              loc_y_arr[kThisAgentArrayIdx]);
            obs_arr[kThisAgentIdxOffset + 20 + 2 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(speed_x_arr[kOtherAgentArrayIdx]) / (kMaxSpeed_x + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 20 + 3 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(speed_y_arr[kOtherAgentArrayIdx]) / (kMaxSpeed_y + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 20 + 4 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(direction_arr[kOtherAgentArrayIdx]) / (kTwoPi);
            obs_arr[kThisAgentIdxOffset + 20 + 5 * kNumOtherAgentsObserved + idx] =
              agent_types;
            obs_arr[kThisAgentIdxOffset + 20 + 6 * kNumOtherAgentsObserved + idx] =
              still_in_the_game_arr[kOtherAgentArrayIdx];
            obs_arr[kThisAgentIdxOffset + 20 + 7 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(agent_sizes);

          }
          int angle_ = int(0.5 + direction_arr[kThisAgentArrayIdx] * 360 / (kTwoPi) / spot_lidar_angle);
          float dir_dx = cos(direction_arr[kThisAgentArrayIdx]);
          float dir_dy = sin(direction_arr[kThisAgentArrayIdx]);
          int front_agent_grid_x = int((loc_x_arr[kThisAgentArrayIdx] + dir_dx) / kGridDensity + 0.5);
          int front_agent_grid_y = int((loc_y_arr[kThisAgentArrayIdx] + dir_dy) / kGridDensity + 0.5);
          int back_agent_grid_x = int((loc_x_arr[kThisAgentArrayIdx] - dir_dx) / kGridDensity + 0.5);
          int back_agent_grid_y = int((loc_y_arr[kThisAgentArrayIdx] - dir_dy) / kGridDensity + 0.5);
          int left_agent_grid_x = int((loc_x_arr[kThisAgentArrayIdx] - dir_dx) / kGridDensity + 0.5);
          int left_agent_grid_y = int((loc_y_arr[kThisAgentArrayIdx] + dir_dy) / kGridDensity + 0.5);
          int right_agent_grid_x = int((loc_x_arr[kThisAgentArrayIdx] + dir_dx) / kGridDensity + 0.5);
          int right_agent_grid_y = int((loc_y_arr[kThisAgentArrayIdx] - dir_dy) / kGridDensity + 0.5);
          float front_pitch = MapDx(grid_map_z, grid_length_max_y, grid_length_height, front_agent_grid_x, front_agent_grid_y) * dir_dx +
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, front_agent_grid_x, front_agent_grid_y) * dir_dy;
          float front_row = MapDx(grid_map_z, grid_length_max_y, grid_length_height, front_agent_grid_x, front_agent_grid_y) * dir_dy -
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, front_agent_grid_x, front_agent_grid_y) * dir_dx;
          float back_pitch = -1 * MapDx(grid_map_z, grid_length_max_y, grid_length_height, back_agent_grid_x, back_agent_grid_y) * dir_dx -
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, back_agent_grid_x, back_agent_grid_y) * dir_dy;
          float back_row = -1 * MapDx(grid_map_z, grid_length_max_y, grid_length_height, back_agent_grid_x, back_agent_grid_y) * dir_dy +
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, back_agent_grid_x, back_agent_grid_y) * dir_dx;

          float left_pitch = -1 * MapDx(grid_map_z, grid_length_max_y, grid_length_height, left_agent_grid_x, left_agent_grid_y) * dir_dy +
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, left_agent_grid_x, left_agent_grid_y) * dir_dx;
          float left_row = MapDx(grid_map_z, grid_length_max_y, grid_length_height, left_agent_grid_x, left_agent_grid_y) * dir_dx +
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, left_agent_grid_x, left_agent_grid_y) * dir_dy;
          float right_pitch = MapDx(grid_map_z, grid_length_max_y, grid_length_height, right_agent_grid_x, right_agent_grid_y) * dir_dy -
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, right_agent_grid_x, right_agent_grid_y) * dir_dx;
          float right_row = -1 * MapDx(grid_map_z, grid_length_max_y, grid_length_height, right_agent_grid_x, right_agent_grid_y) * dir_dx -
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, right_agent_grid_x, right_agent_grid_y) * dir_dy;
          



          obs_arr[kThisAgentIdxOffset] = static_cast<float>(acceleration_x_arr[kThisAgentArrayIdx]) / (MaxAcceleration_x + kEpsilon); 
          obs_arr[kThisAgentIdxOffset + 1] = static_cast<float>(acceleration_y_arr[kThisAgentArrayIdx]) / (MaxAcceleration_y + kEpsilon); 
          obs_arr[kThisAgentIdxOffset + 2] = static_cast<float>(speed_x_arr[kThisAgentArrayIdx] / (kMaxSpeed_x + kEpsilon));
          obs_arr[kThisAgentIdxOffset + 3] = static_cast<float>(speed_y_arr[kThisAgentArrayIdx] / (kMaxSpeed_y + kEpsilon));
          obs_arr[kThisAgentIdxOffset + 4] = static_cast<float>(direction_arr[kThisAgentArrayIdx]) / (kTwoPi);
          obs_arr[kThisAgentIdxOffset + 5] = static_cast<float>(agent_size_arr[kThisAgentId]);
          obs_arr[kThisAgentIdxOffset + 6] = goal_x_;
          obs_arr[kThisAgentIdxOffset + 7] = goal_y_;
          obs_arr[kThisAgentIdxOffset + 8] = goal_z_;
          obs_arr[kThisAgentIdxOffset + 9] = static_cast<float>(loc_x_arr[kThisAgentArrayIdx]) / (kGridLength);
          obs_arr[kThisAgentIdxOffset + 10] = static_cast<float>(loc_y_arr[kThisAgentArrayIdx]) / (kGridLength);
          obs_arr[kThisAgentIdxOffset + 11] = static_cast<float>(front_pitch);
          obs_arr[kThisAgentIdxOffset + 12] = static_cast<float>(front_row);
          obs_arr[kThisAgentIdxOffset + 13] = static_cast<float>(back_pitch);
          obs_arr[kThisAgentIdxOffset + 14] = static_cast<float>(back_row);
          obs_arr[kThisAgentIdxOffset + 15] = static_cast<float>(left_pitch);
          obs_arr[kThisAgentIdxOffset + 16] = static_cast<float>(left_row);
          obs_arr[kThisAgentIdxOffset + 17] = static_cast<float>(right_pitch);
          obs_arr[kThisAgentIdxOffset + 18] = static_cast<float>(right_row);
          obs_arr[kThisAgentIdxOffset + 19] = static_cast<float>(env_timestep_arr[kEnvId]) / kEpisodeLength;

          int left_new_agent_grid_x = int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity));
          int down_new_agent_grid_y = int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity));
          int right_new_agent_grid_x = min(left_new_agent_grid_x + 1, int(grid_length_max_y) - 1);
          int up_new_agent_grid_y = min(down_new_agent_grid_y + 1, int(grid_length_max_y) - 1);

          float precise_new_agent_grid_x = loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) - left_new_agent_grid_x;
          float precise_new_agent_grid_y = loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) - down_new_agent_grid_y;
          for (int i = 0; i < num_angle_lidar; i++)
          {
            int real_angle = int(i + angle_);
            if (real_angle >= num_angle_lidar) real_angle = real_angle - num_angle_lidar;
            if (real_angle < 0) real_angle = real_angle + num_angle_lidar;
            float left_down_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * left_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + real_angle)];
            float left_up_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * left_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + real_angle)];
            float right_down_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * right_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + real_angle)];
            float right_up_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * right_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + real_angle)];
            float precise_angle_distance = precise_new_agent_grid_x * precise_new_agent_grid_y * right_up_angle_distance +
             (1 - precise_new_agent_grid_x) * precise_new_agent_grid_y * left_up_angle_distance +
             precise_new_agent_grid_x * (1 - precise_new_agent_grid_y) * right_down_angle_distance +
             (1 - precise_new_agent_grid_x) * (1 - precise_new_agent_grid_y) * left_down_angle_distance;
            float max_angle_distance = max(max(left_down_angle_distance, left_up_angle_distance), max(right_down_angle_distance, right_up_angle_distance));
            float min_angle_distance = min(min(left_down_angle_distance, left_up_angle_distance), min(right_down_angle_distance, right_up_angle_distance));
            float smallest_distance = min(min(precise_new_agent_grid_x, 1 - precise_new_agent_grid_x), min(precise_new_agent_grid_y, 1 - precise_new_agent_grid_y));
            if (max_angle_distance - min_angle_distance > square_root_2 * kGridDensity)
            {
              obs_arr[kThisAgentIdxOffset + 20 + num_features * kNumOtherAgentsObserved + i] = min_angle_distance - smallest_distance;
            }
            else
            {
              obs_arr[kThisAgentIdxOffset + 20 + num_features * kNumOtherAgentsObserved + i] = precise_angle_distance;
            }
          }
        }
    }
  }

  // Device helper function to compute rewards
  __device__ void CudaTagContinuousComputeReward(
    float * grid_map_z,
    float * different_point_angle_distance,
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_x_arr,
    float * speed_y_arr,
    float * direction_arr,
    float * acceleration_x_arr,
    float * acceleration_y_arr,
    float * rewards_arr,
    float * agent_size_arr,
    float * skill_levels_arr,
    float * display_reward,
    float delta_turn,
    const float Training,
    const float TrainingPhase,
    const float kGridLength,
    const float kGridDensity,
    const float spot_lidar_angle,
    const float MaxTurn,
    const float kMaxSpeed_x,
    const float kMaxSpeed_y,
    const float MaxAcceleration_x,
    const float MinAcceleration_x,
    const float MaxAcceleration_y,
    const float MinAcceleration_y,
    const float kRewardSpotGetGoal,
    const float kRewardSpotFloorDirection,
    const float kRewardSpotFloorDirectionBase,
    const float kPenaltyAgentHit,
    const float kCostOperate,
    const float kCostSpeedYRateWithAccX,
    const float kCostTurnRateWithAccX,
    const float kRewardGoalStep,
    const float kRewardGoalSpeed,
    const float kRewardOnFloor,
    const float grid_length_max_y,
    const float grid_length_height,
    int * still_in_the_game_arr,
    int * agent_types_arr,
    int * num_hit_obstacle,
    int * num_get_goal,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {

    if (kThisAgentId < kNumAgents) {

      // initialize rewards
      rewards_arr[kThisAgentArrayIdx] = 0.0;
      __sync_env_threads();
      bool is_goal = agent_types_arr[kThisAgentId] == 0;
      bool is_spot = agent_types_arr[kThisAgentId] == 1;
      
      //bool is_obstacle = agent_types_arr[kThisAgentId] == 2;
      float dist_obstacle = kGridLength * sqrt(2.0);
      float dist_goal = dist_obstacle;
      float dist_spot = dist_obstacle;
      int nearest_obstacle_idx = 0;
      int nearest_goal_idx = 0;
      int nearest_spot_idx = 0;
      const int kThisAgentIdxOffset = kEnvId * kNumAgents * 13 + kThisAgentId * 13;
      const int num_angle_lidar = int(360 / spot_lidar_angle);

      if (still_in_the_game_arr[kThisAgentArrayIdx]) {

        for (int other_agent_id = 0; other_agent_id < kNumAgents; other_agent_id++) {
          bool other_is_goal = agent_types_arr[other_agent_id] == 0;
          bool other_is_spot = agent_types_arr[other_agent_id] == 1;
          bool other_is_obstacle = agent_types_arr[other_agent_id] == 2;
          const float dist = ComputeAgentDistance(
            agent_size_arr,
            loc_x_arr,
            loc_y_arr,
            kThisAgentId,
            other_agent_id,
            kEnvId,
            kNumAgents,
            agent_types_arr
          );
          const int OtherAgentArrayIdx = kEnvId * kNumAgents + other_agent_id;
          if (other_is_obstacle && dist < dist_obstacle && other_agent_id != kThisAgentId && still_in_the_game_arr[OtherAgentArrayIdx]){
            dist_obstacle = dist;
            nearest_obstacle_idx = OtherAgentArrayIdx;
          }
          if(other_is_goal && dist < dist_goal && other_agent_id != kThisAgentId && still_in_the_game_arr[OtherAgentArrayIdx]){
            dist_goal = dist;
            nearest_goal_idx = OtherAgentArrayIdx;
          }
          if(other_is_spot && dist < dist_spot && other_agent_id != kThisAgentId && still_in_the_game_arr[OtherAgentArrayIdx]){
            dist_spot = dist;
            nearest_spot_idx = OtherAgentArrayIdx;
          }
        }
        if(is_goal)
        {
          // 動態障礙物
          if(dist_obstacle <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
            display_reward[kThisAgentIdxOffset] += kPenaltyAgentHit;
          }

          // 靜態障礙物
          
          int new_agent_grid_x = int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5);
          int new_agent_grid_y = int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5);
          for (int i = 0; i < num_angle_lidar; i++)
          {
            if(different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * new_agent_grid_x + num_angle_lidar * new_agent_grid_y + i)] < agent_size_arr[kThisAgentId])
            {
              rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
              display_reward[kThisAgentIdxOffset + 1] += kPenaltyAgentHit;
              break;
            }
          }
          int left_new_agent_grid_x = int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity));
          int down_new_agent_grid_y = int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity));
          int right_new_agent_grid_x = min(left_new_agent_grid_x + 1, int(grid_length_max_y) - 1);
          int up_new_agent_grid_y = min(down_new_agent_grid_y + 1, int(grid_length_max_y) - 1);

          float precise_new_agent_grid_x = loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) - float(left_new_agent_grid_x); 
          float precise_new_agent_grid_y = loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) - float(down_new_agent_grid_y); 
          for (int i = 0; i < num_angle_lidar; i++)
          {
            float left_down_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * left_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + i)];
            float left_up_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * left_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + i)];
            float right_down_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * right_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + i)];
            float right_up_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * right_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + i)];
            float precise_angle_distance = precise_new_agent_grid_x * precise_new_agent_grid_y * right_up_angle_distance +
             (1 - precise_new_agent_grid_x) * precise_new_agent_grid_y * left_up_angle_distance +
             precise_new_agent_grid_x * (1 - precise_new_agent_grid_y) * right_down_angle_distance +
             (1 - precise_new_agent_grid_x) * (1 - precise_new_agent_grid_y) * left_down_angle_distance;
            float max_angle_distance = max(max(left_down_angle_distance, left_up_angle_distance), max(right_down_angle_distance, right_up_angle_distance));
            float min_angle_distance = min(min(left_down_angle_distance, left_up_angle_distance), min(right_down_angle_distance, right_up_angle_distance));
            float smallest_distance = min(min(precise_new_agent_grid_x, 1 - precise_new_agent_grid_x), min(precise_new_agent_grid_y, 1 - precise_new_agent_grid_y));
            if (max_angle_distance - min_angle_distance > square_root_2 * kGridDensity)
            {
              if (min_angle_distance <= agent_size_arr[kThisAgentId] + smallest_distance)
              {
                rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
                display_reward[kThisAgentIdxOffset + 1] += kPenaltyAgentHit;
                break;
              }
            }
            else
            {
              if (precise_angle_distance <= agent_size_arr[kThisAgentId])
              {
                rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
                display_reward[kThisAgentIdxOffset + 1] += kPenaltyAgentHit;
                break;
              }
            }
          }

          // SPOT reward
          if(dist_spot <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
            display_reward[kThisAgentIdxOffset + 2] += kPenaltyAgentHit;
          }

          // Step reward
          rewards_arr[kThisAgentArrayIdx] += kRewardGoalStep;
          display_reward[kThisAgentIdxOffset + 3] += kRewardGoalStep;

          // Move reward
          float goal_max_speed = kMaxSpeed_x * skill_levels_arr[kThisAgentId];
          float goal_max_acceleration = MaxAcceleration_x * skill_levels_arr[kThisAgentId];
          float nomal_speed_x = abs(speed_x_arr[kThisAgentArrayIdx]) / goal_max_speed;
          float nomal_acc_x = abs(acceleration_x_arr[kThisAgentArrayIdx]) / goal_max_acceleration;
          float goal_move_reward = 0.5 * kRewardGoalSpeed * (nomal_speed_x + 1 - nomal_acc_x);
          rewards_arr[kThisAgentArrayIdx] += goal_move_reward;
          display_reward[kThisAgentIdxOffset + 4] += goal_move_reward;

          // Floor reward
          if (TrainingPhase > 2)
          {
            int self_agent_id = int(grid_length_max_y * new_agent_grid_x + new_agent_grid_y);

            if( abs(grid_map_z[self_agent_id]) > 0 && abs(grid_map_z[self_agent_id]) < grid_length_height)
            {
              rewards_arr[kThisAgentArrayIdx] += kRewardOnFloor;
              display_reward[kThisAgentIdxOffset + 5] += kRewardOnFloor;
            }
          }
        }
        else if(is_spot)
        {
          // 動態障礙物
          if(dist_obstacle <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
            display_reward[kThisAgentIdxOffset + 6] += kPenaltyAgentHit;
            num_hit_obstacle[kEnvId] += 1;
            if (Training == 1) 
            {
              still_in_the_game_arr[kThisAgentArrayIdx] = 0;
            }
          }
          if(dist_spot <= 0)
          {
            if (Training == 1 && still_in_the_game_arr[kThisAgentArrayIdx] == 1) 
            {
              rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
              rewards_arr[nearest_spot_idx] += kPenaltyAgentHit;
              display_reward[kThisAgentIdxOffset + 6] += 2 * kPenaltyAgentHit;
              still_in_the_game_arr[nearest_spot_idx] = 0;
              still_in_the_game_arr[kThisAgentArrayIdx] = 0;

            }
            num_hit_obstacle[kEnvId] += 2;
          }

          // 靜態障礙物 
          int left_new_agent_grid_x = int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity));
          int down_new_agent_grid_y = int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity));
          int right_new_agent_grid_x = min(left_new_agent_grid_x + 1, int(grid_length_max_y) - 1);
          int up_new_agent_grid_y = min(down_new_agent_grid_y + 1, int(grid_length_max_y) - 1);

          float precise_new_agent_grid_x = loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) - float(left_new_agent_grid_x); 
          float precise_new_agent_grid_y = loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) - float(down_new_agent_grid_y); 
          for (int i = 0; i < num_angle_lidar; i++)
          {
            float left_down_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * left_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + i)];
            float left_up_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * left_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + i)];
            float right_down_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * right_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + i)];
            float right_up_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * right_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + i)];
            float precise_angle_distance = precise_new_agent_grid_x * precise_new_agent_grid_y * right_up_angle_distance +
             (1 - precise_new_agent_grid_x) * precise_new_agent_grid_y * left_up_angle_distance +
             precise_new_agent_grid_x * (1 - precise_new_agent_grid_y) * right_down_angle_distance +
             (1 - precise_new_agent_grid_x) * (1 - precise_new_agent_grid_y) * left_down_angle_distance;
            float max_angle_distance = max(max(left_down_angle_distance, left_up_angle_distance), max(right_down_angle_distance, right_up_angle_distance));
            float min_angle_distance = min(min(left_down_angle_distance, left_up_angle_distance), min(right_down_angle_distance, right_up_angle_distance));
            float smallest_distance = min(min(precise_new_agent_grid_x, 1 - precise_new_agent_grid_x), min(precise_new_agent_grid_y, 1 - precise_new_agent_grid_y));
            if (max_angle_distance - min_angle_distance > square_root_2 * kGridDensity)
            {
              if (min_angle_distance < agent_size_arr[kThisAgentId] + smallest_distance)
              {
                rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
                display_reward[kThisAgentIdxOffset + 7] += kPenaltyAgentHit;
                still_in_the_game_arr[kThisAgentArrayIdx] = 0;
                num_hit_obstacle[kEnvId] += 1;
                break;
              }
            }
            else
            {
              if (precise_angle_distance < agent_size_arr[kThisAgentId])
              {
                rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
                display_reward[kThisAgentIdxOffset + 7] += kPenaltyAgentHit;
                still_in_the_game_arr[kThisAgentArrayIdx] = 0;
                num_hit_obstacle[kEnvId] += 1;
                break;
              }
            }
          }

          // Goal reward
          if(dist_goal <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kRewardSpotGetGoal;
            display_reward[kThisAgentIdxOffset + 8] += kRewardSpotGetGoal;
            num_get_goal[kEnvId] += 1;
            if (Training == 1) 
            {
              still_in_the_game_arr[nearest_goal_idx] = 0;
            }
            else
            {
              loc_x_arr[nearest_goal_idx] = float(int(loc_y_arr[nearest_goal_idx] + 21.02) % int(kGridLength));
              loc_y_arr[nearest_goal_idx] = float(int(loc_x_arr[nearest_goal_idx] + 2.35 + speed_x_arr[nearest_obstacle_idx]) % int(kGridLength));
            }
          }
          
          // 運動模型
          if (TrainingPhase > 1)
          {
            float nomal_turn_ = abs(delta_turn) / MaxTurn;
            float nomal_acc_x = abs(acceleration_x_arr[kThisAgentArrayIdx]) / max(abs(MinAcceleration_x), MaxAcceleration_x);
            float nomal_speed_y = abs(speed_y_arr[kThisAgentArrayIdx]) / kMaxSpeed_y;
            float display_reward_ = 0;
            display_reward_ = kCostOperate * (1 - nomal_acc_x) * (1 - nomal_acc_x);
            display_reward_ += kCostOperate * kCostSpeedYRateWithAccX * (1 - nomal_speed_y);
            display_reward_ += kCostOperate * kCostTurnRateWithAccX * (1 - nomal_turn_) * (1 - nomal_turn_);
            rewards_arr[kThisAgentArrayIdx] += display_reward_;
            display_reward[kThisAgentIdxOffset + 9] += display_reward_;
          }
          else
          {
            float display_reward_ = 0;
            if (speed_x_arr[kThisAgentArrayIdx] >= 0)
            {
              display_reward_ = kCostOperate * 20;
            }
            rewards_arr[kThisAgentArrayIdx] += display_reward_;
            display_reward[kThisAgentIdxOffset + 9] += display_reward_;
          }
          // Floor reward
          int new_agent_grid_x = int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5);
          int new_agent_grid_y = int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5);
          int self_agent_id = int(grid_length_max_y * new_agent_grid_x + new_agent_grid_y);

          if( abs(grid_map_z[self_agent_id]) > 0 && abs(grid_map_z[self_agent_id]) < grid_length_height)
          {
            float kRewardOnFloor;
            float dir_dx = cos(direction_arr[kThisAgentArrayIdx]);
            float dir_dy = sin(direction_arr[kThisAgentArrayIdx]);
            float agent_pitch = MapDx(grid_map_z, grid_length_max_y, grid_length_height, new_agent_grid_x, new_agent_grid_y) * dir_dx +
              MapDy(grid_map_z, grid_length_max_y, grid_length_height, new_agent_grid_x, new_agent_grid_y) * dir_dy;
            if (agent_pitch > 0)  kRewardOnFloor = kRewardSpotFloorDirection;
            else 
            {
              kRewardOnFloor = - 1 * kRewardSpotFloorDirectionBase;
              still_in_the_game_arr[kThisAgentArrayIdx] = 0;
              num_hit_obstacle[kEnvId] += 1;
            }
            rewards_arr[kThisAgentArrayIdx] += kRewardOnFloor;
            display_reward[kThisAgentIdxOffset + 10] += kRewardOnFloor;
          }
        }
      }

      display_reward[kThisAgentIdxOffset + 11] = float(num_hit_obstacle[kEnvId]);
      display_reward[kThisAgentIdxOffset + 12] = float(num_get_goal[kEnvId]);

      __sync_env_threads();
      // Use only agent 0's thread to set done_arr
      if (kThisAgentId == 0) {
        if (env_timestep_arr[kEnvId] == kEpisodeLength) {
          done_arr[kEnvId] = 1;
          
        }
      }
    }
    
  }

  __global__ void CudaCarStep(
    float * grid_map_z,
    float * different_point_angle_distance,
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_x_arr,
    float * speed_y_arr,
    float * direction_arr,
    float * acceleration_x_arr,
    float * acceleration_y_arr,
    float * obs_arr,
    float * rewards_arr,
    float * turn_actions_arr,
    float * acceleration_actions_x_arr,
    float * acceleration_actions_y_arr,
    float * neighbor_distances_arr,
    float * agent_size_arr,
    float * skill_levels_arr,
    float * display_reward,
    const float Testing_type,
    const float Training,
    const float TrainingPhase,
    const float kGridLength,
    const float kGridDensity,
    const float kNumOtherAgentsObserved,
    const float spot_lidar_angle,
    const float MaxTurn,
    const float kMaxSpeed_x,
    const float kMinSpeed_x,
    const float kMaxSpeed_y,
    const float kMinSpeed_y,
    const float MaxAcceleration_x,
    const float MinAcceleration_x,
    const float MaxAcceleration_y,
    const float MinAcceleration_y,
    const float kRewardSpotGetGoal,
    const float kRewardSpotFloorDirection,
    const float kRewardSpotFloorDirectionBase,
    const float kPenaltyAgentHit,
    const float kCostOperate,
    const float kCostSpeedYRateWithAccX,
    const float kCostTurnRateWithAccX,
    const float kRewardGoalStep,
    const float kRewardGoalSpeed,
    const float kRewardOnFloor,
    const float grid_length_max_y,
    const float grid_length_height,
    int kNumAgents,
    int kEpisodeLength,
    int * action_indices_arr,
    int * still_in_the_game_arr,
    int * neighbor_ids_sorted_by_distance_arr,
    int * nearest_neighbor_ids,
    int * agent_types_arr,
    int * done_arr,
    int * env_timestep_arr,
    int * num_spot,
    int * num_hit_obstacle,
    int * num_get_goal

  ) {

    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;
    const int kNumActions = 3;

    float delta_turn = kTwoPi;
    const int num_angle_lidar = int(360 / spot_lidar_angle);
    
    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == 0) {
      env_timestep_arr[kEnvId] += 1;
    }
    /*
    bool data_not_finish = true;
    
    //cheak data load finish
    while (data_not_finish) {
      int last_different_point_angle_distance_id = int(num_angle_lidar * grid_length_max_y * (grid_length_max_y - 1) + num_angle_lidar * (grid_length_max_y - 1) + num_angle_lidar - 1);
      if (different_point_angle_distance[last_different_point_angle_distance_id] == different_point_angle_distance[last_different_point_angle_distance_id - 100]) 
      {
        data_not_finish = false; 
      }
      if (done_arr[kEnvId] != 1 && done_arr[kEnvId] != 0)
      {
        printf("\n\ndone_arr error\n\n");
        data_not_finish = true;
      }
    }
    */

    // Wait here until timestep has been updated
    __sync_env_threads();

    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <= kEpisodeLength);
    
    if (kThisAgentId < kNumAgents) {
      int kThisAgentActionIdxOffset = kEnvId * kNumAgents * kNumActions +
        kThisAgentId * kNumActions;
      float delta_acceleration_x = acceleration_actions_x_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 0]];
      float delta_acceleration_y = acceleration_actions_y_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 1]];
      delta_turn = turn_actions_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 2]];

      acceleration_x_arr[kThisAgentArrayIdx] += delta_acceleration_x;
      acceleration_y_arr[kThisAgentArrayIdx] += delta_acceleration_y;

      direction_arr[kThisAgentArrayIdx] = fmod(
        direction_arr[kThisAgentArrayIdx] + delta_turn, kTwoPi) *
        still_in_the_game_arr[kThisAgentArrayIdx];
      if (direction_arr[kThisAgentArrayIdx] < 0) {
        direction_arr[kThisAgentArrayIdx] = kTwoPi + direction_arr[
          kThisAgentArrayIdx];
      }
      
      // Speed clipping
      speed_x_arr[kThisAgentArrayIdx] = min(
          kMaxSpeed_x * skill_levels_arr[kThisAgentId],
          max(
            kMinSpeed_x * skill_levels_arr[kThisAgentId],
            speed_x_arr[kThisAgentArrayIdx] + acceleration_x_arr[
              kThisAgentArrayIdx])) * still_in_the_game_arr[kThisAgentArrayIdx];
      
      speed_y_arr[kThisAgentArrayIdx] = min(
          kMaxSpeed_y * skill_levels_arr[kThisAgentId],
          max(
            kMinSpeed_y * skill_levels_arr[kThisAgentId],
            speed_y_arr[kThisAgentArrayIdx] + acceleration_y_arr[
              kThisAgentArrayIdx])) * still_in_the_game_arr[kThisAgentArrayIdx];

      // acceleration clipping
      if ((speed_x_arr[kThisAgentArrayIdx] <= kMinSpeed_x * skill_levels_arr[kThisAgentId]) ||
        (speed_x_arr[kThisAgentArrayIdx] >=
        (kMaxSpeed_x * skill_levels_arr[kThisAgentId]))) {
          acceleration_x_arr[kThisAgentArrayIdx] = 0.0;
      }

      if ((speed_y_arr[kThisAgentArrayIdx] <= kMinSpeed_y * skill_levels_arr[kThisAgentId]) ||
        (speed_y_arr[kThisAgentArrayIdx] >=
        (kMaxSpeed_y * skill_levels_arr[kThisAgentId]))) {
          acceleration_y_arr[kThisAgentArrayIdx] = 0.0;
      }

      if (agent_types_arr[kThisAgentId] != 1)  // not spot
      {
        
        speed_y_arr[kThisAgentArrayIdx] = 0.0;
        acceleration_y_arr[kThisAgentArrayIdx] = 0.0;
        if (agent_types_arr[kThisAgentId] == 2)  // obstacles
        {
          bool need_cheak_obstacles_hit_wall = true;
          int random_seed = 5;
          while (need_cheak_obstacles_hit_wall)
          {
            random_seed += 1;
            float _new_loc_x_arr = loc_x_arr[kThisAgentArrayIdx] + speed_x_arr[kThisAgentArrayIdx] * cos(direction_arr[kThisAgentArrayIdx]) ;
            float _new_loc_y_arr = loc_y_arr[kThisAgentArrayIdx] + speed_x_arr[kThisAgentArrayIdx] * sin(direction_arr[kThisAgentArrayIdx]) ;
            int new_agent_grid_x = int(_new_loc_x_arr / kGridDensity + 0.5);
            int new_agent_grid_y = int(_new_loc_y_arr / kGridDensity + 0.5);
            int new_hit_wall_ = 0;
            int or_agent_grid_x = int(loc_x_arr[kThisAgentArrayIdx] / kGridDensity + 0.5);
            int or_agent_grid_y = int(loc_y_arr[kThisAgentArrayIdx] / kGridDensity + 0.5);
            int or_hit_wall_ = 0;

            for (int i = 0; i < num_angle_lidar; i++)
            {
              if(different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * or_agent_grid_x + num_angle_lidar * or_agent_grid_y + i)] < agent_size_arr[kThisAgentId])
              {
                or_hit_wall_ = 1;
              }
              if(different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * new_agent_grid_x + num_angle_lidar * new_agent_grid_y + i)] < agent_size_arr[kThisAgentId])
              {
                new_hit_wall_ = 1;
                break;
              }
            }

            if (new_hit_wall_ == 1 && or_hit_wall_ == 0)
            {
              direction_arr[kThisAgentArrayIdx] = fmod(direction_arr[kThisAgentArrayIdx] + kPi + int(loc_y_arr[kThisAgentArrayIdx + random_seed]), kTwoPi);  // 隨機角度
            }
            if (or_hit_wall_ == 1 || new_hit_wall_ == 0)
            {
              break;
            }
          }
        }
        loc_x_arr[kThisAgentArrayIdx] += speed_x_arr[kThisAgentArrayIdx] * cos(direction_arr[kThisAgentArrayIdx]);
        loc_y_arr[kThisAgentArrayIdx] += speed_x_arr[kThisAgentArrayIdx] * sin(direction_arr[kThisAgentArrayIdx]);
      }
      else
      {
        loc_x_arr[kThisAgentArrayIdx] += speed_x_arr[kThisAgentArrayIdx] * cos(direction_arr[kThisAgentArrayIdx]) - speed_y_arr[kThisAgentArrayIdx] * sin(direction_arr[kThisAgentArrayIdx]);
        loc_y_arr[kThisAgentArrayIdx] += speed_x_arr[kThisAgentArrayIdx] * sin(direction_arr[kThisAgentArrayIdx]) + speed_y_arr[kThisAgentArrayIdx] * cos(direction_arr[kThisAgentArrayIdx]);
      }

      // Crossing the edge
      bool has_crossed_edge = (
        (loc_x_arr[kThisAgentArrayIdx] < 0) |
        (loc_x_arr[kThisAgentArrayIdx] > kGridLength) |
        (loc_y_arr[kThisAgentArrayIdx] < 0) |
        (loc_y_arr[kThisAgentArrayIdx] > kGridLength));

      // Clip x and y if agent has crossed edge
      if (has_crossed_edge) {
        if (loc_x_arr[kThisAgentArrayIdx] < 0) {
          loc_x_arr[kThisAgentArrayIdx] = 0.0;
        } else if (loc_x_arr[kThisAgentArrayIdx] > kGridLength) {
          loc_x_arr[kThisAgentArrayIdx] = kGridLength;
        }
        if (loc_y_arr[kThisAgentArrayIdx] < 0) {
          loc_y_arr[kThisAgentArrayIdx] = 0.0;
        } else if (loc_y_arr[kThisAgentArrayIdx] > kGridLength) {
          loc_y_arr[kThisAgentArrayIdx] = kGridLength;
        }
      } 
    }


    // Make sure all agents have updated their states

    CudaTagContinuousGenerateObservation(
      loc_x_arr,
      loc_y_arr,
      speed_x_arr,
      speed_y_arr,
      direction_arr,
      acceleration_x_arr,
      acceleration_y_arr,
      obs_arr,
      neighbor_distances_arr,
      agent_size_arr,
      grid_map_z,
      different_point_angle_distance,
      Training,
      kGridLength,
      kGridDensity,
      kNumOtherAgentsObserved,
      spot_lidar_angle,
      kMaxSpeed_x,
      kMaxSpeed_y,
      MaxAcceleration_x,
      MaxAcceleration_y,
      grid_length_max_y,
      grid_length_height,
      still_in_the_game_arr,
      neighbor_ids_sorted_by_distance_arr,
      nearest_neighbor_ids,
      agent_types_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx
    );

    CudaTagContinuousComputeReward(
      grid_map_z,
      different_point_angle_distance,
      loc_x_arr,
      loc_y_arr,
      speed_x_arr,
      speed_y_arr,
      direction_arr,
      acceleration_x_arr,
      acceleration_y_arr,
      rewards_arr,
      agent_size_arr,
      skill_levels_arr,
      display_reward,
      delta_turn,
      Training,
      TrainingPhase,
      kGridLength,
      kGridDensity,
      spot_lidar_angle,
      MaxTurn,
      kMaxSpeed_x,
      kMaxSpeed_y,
      MaxAcceleration_x,
      MinAcceleration_x,
      MaxAcceleration_y,
      MinAcceleration_y,
      kRewardSpotGetGoal,
      kRewardSpotFloorDirection,
      kRewardSpotFloorDirectionBase,
      kPenaltyAgentHit,
      kCostOperate,
      kCostSpeedYRateWithAccX,
      kCostTurnRateWithAccX,
      kRewardGoalStep,
      kRewardGoalSpeed,
      kRewardOnFloor,
      grid_length_max_y,
      grid_length_height,
      still_in_the_game_arr,
      agent_types_arr,
      num_hit_obstacle,
      num_get_goal,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx
    );
  }
}
