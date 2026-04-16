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
    int * agent_types_arr,
    float agent1_dx,
    float agent1_dy
  ) {
    const int index1 = kEnvId * kNumAgents + kThisAgentId1;
    const int index2 = kEnvId * kNumAgents + kThisAgentId2;
    float min_dis = sqrt(
      pow(loc_x_arr[index1] + agent1_dx - loc_x_arr[index2], 2) +
      pow(loc_y_arr[index1] + agent1_dy - loc_y_arr[index2], 2)
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
    else if(map_x >= grid_length_max_y - 2) point_dx = grid_length_height;
    else if(map_y >= 0 && map_y < grid_length_max_y) 
    {
      int big_dx_id = grid_length_max_y * map_y + map_x + 1;
      int small_dx_id = grid_length_max_y * map_y + map_x - 1;
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
    else if(map_y >= grid_length_max_y - 2) point_dy = grid_length_height;
    else if(map_x >= 0 && map_x < grid_length_max_y) 
    {
      int big_dy_id = grid_length_max_y * (map_y + 1) + map_x;
      int small_dy_id = grid_length_max_y * (map_y - 1) + map_x;
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
    const float Testing_type,
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
    const float four_sign_distance,
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
    int num_features = 6;
    __sync_env_threads();
    //printf("obs died ");
    if (kThisAgentId < kNumAgents) {
        
        int num_angle_lidar = int(360 / spot_lidar_angle);
        const int kNumOtherAgentsObserved = kNumOtherAgentsObserved_;
        const int kThisAgentIdxOffset = kEnvId * kNumAgents *
          (num_features * kNumOtherAgentsObserved + 9 + 2 * num_angle_lidar) +
          kThisAgentId * (num_features * kNumOtherAgentsObserved + 9 + 2 * num_angle_lidar);
        
        //printf("4");
        obs_arr[kThisAgentIdxOffset] = 0.0;      // speed x
        obs_arr[kThisAgentIdxOffset + 1] = 0.0;  // speed y
        obs_arr[kThisAgentIdxOffset + 2] = 20.0;  // goal x
        obs_arr[kThisAgentIdxOffset + 3] = 20.0;  // goal y
        obs_arr[kThisAgentIdxOffset + 4] = 2.0;  // goal z
        obs_arr[kThisAgentIdxOffset + 5] = 0.0;  // goal dir
        obs_arr[kThisAgentIdxOffset + 6] = 1.0;  // goal speed
        obs_arr[kThisAgentIdxOffset + 7] = 0.0;  // time
        obs_arr[kThisAgentIdxOffset + 8] = 0.0;  // in game

        //printf("5");
        for (int idx = 0; idx < kNumOtherAgentsObserved; idx++) {
          obs_arr[kThisAgentIdxOffset + 9 + 0 * kNumOtherAgentsObserved + idx] = 20.0;  // other LOC_X
          obs_arr[kThisAgentIdxOffset + 9 + 1 * kNumOtherAgentsObserved + idx] = 20.0;  // other LOC_Y
          obs_arr[kThisAgentIdxOffset + 9 + 2 * kNumOtherAgentsObserved + idx] = 0.0;  // other SP_X
          obs_arr[kThisAgentIdxOffset + 9 + 3 * kNumOtherAgentsObserved + idx] = 0.0;  // other SP_Y
          obs_arr[kThisAgentIdxOffset + 9 + 4 * kNumOtherAgentsObserved + idx] = 0.0;  // other DIR
          obs_arr[kThisAgentIdxOffset + 9 + 5 * kNumOtherAgentsObserved + idx] = 0.0;  // other agent size
        }

        for (int angle = 0; angle < num_angle_lidar; angle++){
          obs_arr[kThisAgentIdxOffset + 9 + num_features * kNumOtherAgentsObserved + angle] = 0.0;  // local map infomation
        }
        for (int angle = 0; angle < num_angle_lidar; angle++){
          obs_arr[kThisAgentIdxOffset + 9 + num_features * kNumOtherAgentsObserved + num_angle_lidar + angle] = 2.0;  // local map infomation
        }
        
        int distance_arr_idx;
        int i_index;
        int j_index;
        int neighbor_ids_sorted_by_distance_arr_idx;

        // Find the nearest agents
        const int kThisAgentArrayIdxOffset = kEnvId * kNumAgents * (kNumAgents - 1) + kThisAgentId * (kNumAgents - 1);

        // Initialize neighbor_ids_sorted_by_distance_arr
        // other agents that are still in the same
        int num_valid_other_agents = 0;
        int num_goal_other_agents = 0;
        for (int other_agent_id = 0; other_agent_id < kNumAgents; other_agent_id++) {
          if ((other_agent_id != kThisAgentId) && (still_in_the_game_arr[kEnvId * kNumAgents + other_agent_id]) && (agent_types_arr[kThisAgentId] != 0) && (agent_types_arr[other_agent_id] == 2)) {
            neighbor_ids_sorted_by_distance_arr_idx = kThisAgentArrayIdxOffset + num_valid_other_agents;
            neighbor_ids_sorted_by_distance_arr[neighbor_ids_sorted_by_distance_arr_idx] = other_agent_id;
            num_valid_other_agents++;
          }
          else if ((other_agent_id != kThisAgentId) && (still_in_the_game_arr[kEnvId * kNumAgents + other_agent_id]) && (agent_types_arr[kThisAgentId] == 0) && (agent_types_arr[other_agent_id] == 1)) {
            neighbor_ids_sorted_by_distance_arr_idx = kThisAgentArrayIdxOffset + num_valid_other_agents;
            neighbor_ids_sorted_by_distance_arr[neighbor_ids_sorted_by_distance_arr_idx] = other_agent_id;
            num_valid_other_agents++;
          }
          
        }
        // save goal distance
        for (int other_agent_id = 0; other_agent_id < kNumAgents; other_agent_id++) {
         if ((still_in_the_game_arr[kEnvId * kNumAgents + other_agent_id]) && (agent_types_arr[other_agent_id] == 0)) {
            neighbor_ids_sorted_by_distance_arr_idx = kThisAgentArrayIdxOffset + num_valid_other_agents + num_goal_other_agents;
            neighbor_ids_sorted_by_distance_arr[neighbor_ids_sorted_by_distance_arr_idx] = other_agent_id;
            num_goal_other_agents++;
          }
        }

        //neighbor_ids_sorted_by_distance_arr have this env no sort but delete no in the game agent
        // First, find distances to all the valid agents
        for (int idx = 0; idx < num_valid_other_agents + num_goal_other_agents; idx++) {
          distance_arr_idx = kThisAgentArrayIdxOffset + idx;
          neighbor_distances_arr[distance_arr_idx] = ComputeAgentDistance(
            agent_size_arr,
            loc_x_arr,
            loc_y_arr,
            kThisAgentId,
            neighbor_ids_sorted_by_distance_arr[distance_arr_idx],
            kEnvId,
            kNumAgents,
            agent_types_arr,
            0,
            0);
        }
        

        // sort neighbor agent indices
        // add some "if" to let car agent at least see 1 goal 
        float goal_x_ = 0;
        float goal_y_ = 0;
        float goal_z_ = 0;
        float goal_dir_ = 0;
        float goal_spot_x_ = 0;
        float goal_spot_y_ = 0;
        float goal_speed_ = 0;

        //bool check_car_can_look_goal = false;

        for (int i = 0; i < min(num_valid_other_agents, kNumOtherAgentsObserved); i++) {
          i_index = kThisAgentArrayIdxOffset + i;
          for (int j = i + 1; j < num_valid_other_agents; j++) {
            j_index = kThisAgentArrayIdxOffset + j;
            if (neighbor_distances_arr[j_index] < neighbor_distances_arr[i_index]) {
              
              float tmp1 = neighbor_distances_arr[i_index];
              neighbor_distances_arr[i_index] = neighbor_distances_arr[j_index];
              neighbor_distances_arr[j_index] = tmp1;

              int tmp2 = neighbor_ids_sorted_by_distance_arr[i_index];
              neighbor_ids_sorted_by_distance_arr[i_index] = neighbor_ids_sorted_by_distance_arr[j_index];
              neighbor_ids_sorted_by_distance_arr[j_index] = tmp2;
            }
          }
        }

        i_index = kThisAgentArrayIdxOffset + num_valid_other_agents;
        int close_goal_idx = neighbor_ids_sorted_by_distance_arr[i_index];
        for (int i = 0; i < num_goal_other_agents; i++) {
          j_index = kThisAgentArrayIdxOffset + num_valid_other_agents + i;
          if (neighbor_distances_arr[j_index] < neighbor_distances_arr[i_index]) 
          {
            neighbor_distances_arr[i_index] = neighbor_distances_arr[j_index];
            close_goal_idx = neighbor_ids_sorted_by_distance_arr[j_index];
          }
        }
        
        // Save nearest neighbor ids.
        for (int idx = 0; idx < min(num_valid_other_agents, kNumOtherAgentsObserved); idx++) {
          const int kNearestNeighborsIdx =
            kEnvId * kNumAgents * kNumOtherAgentsObserved +
            kThisAgentId * kNumOtherAgentsObserved +
            idx;
          nearest_neighbor_ids[kNearestNeighborsIdx] = neighbor_ids_sorted_by_distance_arr[kThisAgentArrayIdxOffset + idx];
        }
        //nearest_neighbor_ids have sort agent ids

        // Update observation
        float cos_dir = cos(direction_arr[kThisAgentArrayIdx]);
        float sin_dir = sin(direction_arr[kThisAgentArrayIdx]);
        for (int idx = 0; idx < min(num_valid_other_agents, kNumOtherAgentsObserved); idx++) {
          const int kNearestNeighborsIdx =
            kEnvId * kNumAgents * kNumOtherAgentsObserved +
            kThisAgentId * kNumOtherAgentsObserved +
            idx;
          const int kOtherAgentId = nearest_neighbor_ids[kNearestNeighborsIdx];
          const int kOtherAgentArrayIdx = kEnvId * kNumAgents + kOtherAgentId;
          float agent_sizes = agent_size_arr[kOtherAgentId];
          float agent_speed_y = 0;
          float dif_x =  static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] - loc_x_arr[kThisAgentArrayIdx]);
          float dif_y =  static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] - loc_y_arr[kThisAgentArrayIdx]);
          float dif_turn = direction_arr[kOtherAgentArrayIdx] - direction_arr[kThisAgentArrayIdx];
          if (dif_turn < 0) dif_turn += kTwoPi;
          float obs_spot_x_ = dif_x * cos_dir + dif_y * sin_dir;
          float obs_spot_y_ = - 1 * dif_x * sin_dir + dif_y * cos_dir;
          obs_arr[kThisAgentIdxOffset + 9 + 0 * kNumOtherAgentsObserved + idx] = static_cast<float>(obs_spot_x_);
          obs_arr[kThisAgentIdxOffset + 9 + 1 * kNumOtherAgentsObserved + idx] = static_cast<float>(obs_spot_y_);
          obs_arr[kThisAgentIdxOffset + 9 + 2 * kNumOtherAgentsObserved + idx] = static_cast<float>(speed_x_arr[kOtherAgentArrayIdx]) / (kMaxSpeed_x + kEpsilon);
          obs_arr[kThisAgentIdxOffset + 9 + 3 * kNumOtherAgentsObserved + idx] = static_cast<float>(agent_speed_y);
          obs_arr[kThisAgentIdxOffset + 9 + 4 * kNumOtherAgentsObserved + idx] = static_cast<float>(dif_turn) / (kTwoPi);
          obs_arr[kThisAgentIdxOffset + 9 + 5 * kNumOtherAgentsObserved + idx] = static_cast<float>(agent_sizes);
        }
        if (num_goal_other_agents > 0)
        {
          const int kOtherAgentArrayIdx = kEnvId * kNumAgents + close_goal_idx;
          goal_x_ = static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] - loc_x_arr[kThisAgentArrayIdx]);
          goal_y_ = static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] - loc_y_arr[kThisAgentArrayIdx]);
          goal_dir_ = static_cast<float>(direction_arr[kOtherAgentArrayIdx] - direction_arr[kThisAgentArrayIdx]);
          if (goal_dir_ < 0) goal_dir_ += kTwoPi;
          int goal_map_x = min(int(grid_length_max_y) - 1, int(loc_x_arr[kOtherAgentArrayIdx] / kGridDensity + 0.5));
          int goal_map_y = min(int(grid_length_max_y) - 1, int(loc_y_arr[kOtherAgentArrayIdx] / kGridDensity + 0.5));
          float goal_map_z = grid_map_z[int(grid_length_max_y * goal_map_y + goal_map_x)];
          int agent_map_x = min(int(grid_length_max_y) - 1, int(loc_x_arr[kThisAgentArrayIdx] / kGridDensity + 0.5));
          int agent_map_y = min(int(grid_length_max_y) - 1, int(loc_y_arr[kThisAgentArrayIdx] / kGridDensity + 0.5));
          float agent_map_z = grid_map_z[int(grid_length_max_y * agent_map_y + agent_map_x)];
          goal_z_ = static_cast<float>(goal_map_z - agent_map_z);
          goal_spot_x_ =  static_cast<float>(goal_x_ * cos_dir + goal_y_ *  sin_dir);
          goal_spot_y_ =  static_cast<float>(- 1 * goal_x_ * sin_dir + goal_y_ * cos_dir);
          goal_speed_ = static_cast<float>(speed_x_arr[kOtherAgentArrayIdx] / (kMaxSpeed_x + kEpsilon));
        }
        else
        {
          goal_spot_x_ = static_cast<float>(20.0);
          goal_spot_y_ = static_cast<float>(20.0);
          goal_z_ = static_cast<float>(2.0);
          goal_dir_ =  static_cast<float>(0.0);
          goal_speed_ = static_cast<float>(0.0);
        }
        
        
        int agent_grid_x = min(int(loc_x_arr[kThisAgentArrayIdx] / kGridDensity + 0.5), int(grid_length_max_y - 1));
        int agent_grid_y = min(int(loc_y_arr[kThisAgentArrayIdx] / kGridDensity + 0.5), int(grid_length_max_y - 1));
        float agent_z = grid_map_z[int(grid_length_max_y * agent_grid_y + agent_grid_x)];
        float noise = fmodf(loc_y_arr[kThisAgentArrayIdx + 13], 0.1f) - 0.05f;
        float noise2 = fmodf(loc_x_arr[kThisAgentArrayIdx + 7], 0.1f) - 0.05f;
        if (agent_types_arr[kThisAgentId] == 0) //goal obs
        {
          int front_agent_grid_x = min(max(int((loc_x_arr[kThisAgentArrayIdx] + cos_dir * four_sign_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          int front_agent_grid_y = min(max(int((loc_y_arr[kThisAgentArrayIdx] + sin_dir * four_sign_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          int back_agent_grid_x = min(max(int((loc_x_arr[kThisAgentArrayIdx] - cos_dir * four_sign_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          int back_agent_grid_y = min(max(int((loc_y_arr[kThisAgentArrayIdx] - sin_dir * four_sign_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          
          int left_agent_grid_x = min(max(int((loc_x_arr[kThisAgentArrayIdx] - cos_dir * four_sign_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          int left_agent_grid_y = min(max(int((loc_y_arr[kThisAgentArrayIdx] + sin_dir * four_sign_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          int right_agent_grid_x = min(max(int((loc_x_arr[kThisAgentArrayIdx] + cos_dir * four_sign_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          int right_agent_grid_y = min(max(int((loc_y_arr[kThisAgentArrayIdx] - sin_dir * four_sign_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          
          float front_z = grid_map_z[int(grid_length_max_y * front_agent_grid_y + front_agent_grid_x)] - agent_z;
          float back_z = grid_map_z[int(grid_length_max_y * back_agent_grid_y + back_agent_grid_x)] - agent_z;
          float left_z = grid_map_z[int(grid_length_max_y * left_agent_grid_y + left_agent_grid_x)] - agent_z;
          float right_z = grid_map_z[int(grid_length_max_y * right_agent_grid_y + right_agent_grid_x)] - agent_z;
          
          float self_row = MapDx(grid_map_z, grid_length_max_y, grid_length_height, agent_grid_x, agent_grid_y) * cos_dir +
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, agent_grid_x, agent_grid_y) * sin_dir;
          float self_pitch = -1 * MapDx(grid_map_z, grid_length_max_y, grid_length_height, agent_grid_x, agent_grid_y) * sin_dir +
            MapDy(grid_map_z, grid_length_max_y, grid_length_height, agent_grid_x, agent_grid_y) * cos_dir;
          obs_arr[kThisAgentIdxOffset] = static_cast<float>(speed_x_arr[kThisAgentArrayIdx] + noise / (kMaxSpeed_x + kEpsilon));  
          obs_arr[kThisAgentIdxOffset + 1] = self_row;
          obs_arr[kThisAgentIdxOffset + 2] = self_pitch;
          obs_arr[kThisAgentIdxOffset + 3] = front_z;
          obs_arr[kThisAgentIdxOffset + 4] = back_z;
          obs_arr[kThisAgentIdxOffset + 5] = left_z;
          obs_arr[kThisAgentIdxOffset + 6] = right_z;
        }
        else
        {
          obs_arr[kThisAgentIdxOffset] = static_cast<float>(speed_x_arr[kThisAgentArrayIdx] + noise / (kMaxSpeed_x + kEpsilon));
          obs_arr[kThisAgentIdxOffset + 1] = static_cast<float>(speed_y_arr[kThisAgentArrayIdx] + noise2 / (kMaxSpeed_y + kEpsilon));
          obs_arr[kThisAgentIdxOffset + 2] = goal_spot_x_;
          obs_arr[kThisAgentIdxOffset + 3] = goal_spot_y_;
          obs_arr[kThisAgentIdxOffset + 4] = goal_z_;
          obs_arr[kThisAgentIdxOffset + 5] = goal_dir_ / (kTwoPi);
          obs_arr[kThisAgentIdxOffset + 6] = goal_speed_;
        }
        
        obs_arr[kThisAgentIdxOffset + 7] = static_cast<float>(env_timestep_arr[kEnvId]) / kEpisodeLength;
        obs_arr[kThisAgentIdxOffset + 8] = static_cast<float>(1);
        
        //if (agent_types_arr[kThisAgentId] == 1)  printf("goal x y = %f,  %f\n", goal_spot_x_, goal_spot_y_);

        int left_x = min(int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity)), int(grid_length_max_y) - 1);
        int down_y = min(int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity)), int(grid_length_max_y) - 1);
        int right_x = min(left_x + 1, int(grid_length_max_y) - 1);
        int up_y = min(down_y + 1, int(grid_length_max_y) - 1);
        int nomal_agent_grid_x = min(int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1);
        int nomal_agent_grid_y = min(int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1);

        float dx = loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) - left_x;
        float dy = loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) - down_y;
        float angle_ = direction_arr[kThisAgentArrayIdx] * 360 / (kTwoPi) / spot_lidar_angle;
        for (int i = 0; i < num_angle_lidar; i++)
        {
          int real_angle = int(i + angle_ + 0.5);
          if (real_angle >= num_angle_lidar) real_angle = real_angle - num_angle_lidar;
          if (real_angle < 0) real_angle = real_angle + num_angle_lidar;
          float ld_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * down_y + num_angle_lidar * left_x + real_angle)];
          float lu_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * up_y + num_angle_lidar * left_x + real_angle)];
          float rd_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * down_y + num_angle_lidar * right_x + real_angle)];
          float ru_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * up_y + num_angle_lidar * right_x + real_angle)];
          float nomal_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * nomal_agent_grid_y + num_angle_lidar * nomal_agent_grid_x + real_angle)];
          float precise_angle_distance = dx * dy * ru_distance + (1 - dx) * dy * lu_distance + dx * (1 - dy) * rd_distance + (1 - dx) * (1 - dy) * ld_distance;
          float max_distance = max(max(ld_distance, lu_distance), max(rd_distance, ru_distance));
          float min_distance = min(min(ld_distance, lu_distance), min(rd_distance, ru_distance));
          float smallest_distance = float(kGridDensity) *  min(min(dx, 1 - dx), min(dy, 1 - dy));
          if (max_distance - min_distance > square_root_2 * kGridDensity)
          {
            obs_arr[kThisAgentIdxOffset + 9 + num_features * kNumOtherAgentsObserved + i] = nomal_angle_distance - smallest_distance;
          }
          else
          {
            obs_arr[kThisAgentIdxOffset + 9 + num_features * kNumOtherAgentsObserved + i] = precise_angle_distance;
          }
        }
        for (int i = 0; i < num_angle_lidar; i++)
        {
          float grident_distance = 1.0;
          float lidar_angle = direction_arr[kThisAgentArrayIdx] + i * spot_lidar_angle * kTwoPi / 360.0;
          float lidar_x_dir = cos(lidar_angle);
          float lidar_y_dir = sin(lidar_angle);
          int lidar_grid_x = min(max(int((loc_x_arr[kThisAgentArrayIdx] + lidar_x_dir * grident_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          int lidar_grid_y = min(max(int((loc_y_arr[kThisAgentArrayIdx] + lidar_y_dir * grident_distance) / kGridDensity + 0.5), 0), int(grid_length_max_y - 1));
          float lidar_z = grid_map_z[int(grid_length_max_y * lidar_grid_y + lidar_grid_x)] - agent_z;
          obs_arr[kThisAgentIdxOffset + 9 + num_features * kNumOtherAgentsObserved + num_angle_lidar +  i] = lidar_z;
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
    const float origin_position_x,
    const float origin_position_y,
    const float Testing_type,
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
    const float spot_reward_on_floor,
    const float spot_penalty_hit,
    const float goal_penalty_hit,
    const float kCostOperate,
    const float kCostSpeedYRateWithAccX,
    const float kCostTurnRateWithAccX,
    const float kRewardGoalStep,
    const float kRewardGoalSpeed,
    const float kRewardOnFloor,
    const float grid_length_max_y,
    const float grid_length_height,
    const float spot_head_distance,
    const float spot_ass_distance,
    const float random_agent_seed,
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
    //printf("re died ");
    if (kThisAgentId < kNumAgents) {

      // initialize rewards
      rewards_arr[kThisAgentArrayIdx] = 0.0;
      __sync_env_threads();
      bool is_goal = agent_types_arr[kThisAgentId] == 0;
      bool is_spot = agent_types_arr[kThisAgentId] == 1;
      bool is_wall = agent_types_arr[kThisAgentId] == 2;
      
      //bool is_obstacle = agent_types_arr[kThisAgentId] == 2;
      float dist_obstacle = kGridLength * sqrt(2.0);
      float dist_ass = dist_obstacle;
      float dist_goal = dist_obstacle;
      float dist_spot = dist_obstacle;
      int nearest_obstacle_idx = 0;
      int nearest_goal_idx = 0;
      //int nearest_spot_idx = 0;
      const int kThisAgentIdxOffset = kEnvId * kNumAgents * 14 + kThisAgentId * 14;
      const int num_angle_lidar = int(360 / spot_lidar_angle);

      if (still_in_the_game_arr[kThisAgentArrayIdx] || is_goal) {

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
            agent_types_arr,
            0,
            0
          );
          const float head_dist = ComputeAgentDistance(
            agent_size_arr,
            loc_x_arr,
            loc_y_arr,
            kThisAgentId,
            other_agent_id,
            kEnvId,
            kNumAgents,
            agent_types_arr,
            cos(direction_arr[kThisAgentArrayIdx]) * spot_head_distance,
            sin(direction_arr[kThisAgentArrayIdx]) * spot_head_distance
          );
          const float ass_dist = ComputeAgentDistance(
            agent_size_arr,
            loc_x_arr,
            loc_y_arr,
            kThisAgentId,
            other_agent_id,
            kEnvId,
            kNumAgents,
            agent_types_arr,
            -1 * cos(direction_arr[kThisAgentArrayIdx]) * spot_ass_distance,
            -1 * sin(direction_arr[kThisAgentArrayIdx]) * spot_ass_distance
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
            //nearest_spot_idx = OtherAgentArrayIdx;
          }
          if(is_spot && other_is_obstacle && head_dist < dist_obstacle && other_agent_id != kThisAgentId && still_in_the_game_arr[OtherAgentArrayIdx]){
            dist_obstacle = head_dist;
            nearest_obstacle_idx = OtherAgentArrayIdx;
          }
          if(is_spot && other_is_obstacle && ass_dist < dist_ass && other_agent_id != kThisAgentId && still_in_the_game_arr[OtherAgentArrayIdx]){
            dist_ass = ass_dist;
            nearest_obstacle_idx = OtherAgentArrayIdx;
          }
        }
        if(is_goal)
        {
          // 動態障礙物
          bool goal_hit = false;
          if(dist_obstacle <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += 0; //goal_penalty_hit;
            display_reward[kThisAgentIdxOffset] += 0; //goal_penalty_hit;
            //goal_hit = true;
          }

          // 靜態障礙物 is delete
          bool goal_hit_wall = false;
          int left_x = min(int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity)), int(grid_length_max_y) - 1);
          int down_y = min(int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity)), int(grid_length_max_y) - 1);
          float dx = loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) - left_x;
          float dy = loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) - down_y;
          int nomal_agent_grid_x = min(int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1);
          int nomal_agent_grid_y = min(int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1);

          for (int i = 0; i < num_angle_lidar; i++)
          {
            float nomal_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * nomal_agent_grid_y + num_angle_lidar * nomal_agent_grid_x + i)];
            float smallest_distance = float(kGridDensity) *  min(min(dx, 1 - dx), min(dy, 1 - dy));
            if (nomal_angle_distance - smallest_distance < agent_size_arr[kThisAgentId] )
            {
              goal_hit_wall = true;
              break;
            }
          }
          // SPOT reward
          if(dist_spot <= 0)
          {
            //rewards_arr[kThisAgentArrayIdx] += goal_penalty_hit;
            display_reward[kThisAgentIdxOffset + 2] += goal_penalty_hit;
            //goal_hit = true;
          }

          // Step reward
          if (goal_hit_wall == false)
          {
            rewards_arr[kThisAgentArrayIdx] += kRewardGoalStep;
            display_reward[kThisAgentIdxOffset + 3] += kRewardGoalStep;
          }

          // Move reward

          float nomal_speed_x = abs(speed_x_arr[kThisAgentArrayIdx] + 0.07) / kMaxSpeed_x;
          float nomal_acc_x = abs(acceleration_x_arr[kThisAgentArrayIdx]) / MaxAcceleration_x;
          //float obs_move_reward = 2 * kRewardGoalSpeed * (nomal_speed_x + 1 - nomal_acc_x);
          float goal_move_reward = kRewardGoalSpeed * (nomal_speed_x + 0.3 * nomal_acc_x);;
          if (goal_hit_wall == false)
          {
            rewards_arr[kThisAgentArrayIdx] += goal_move_reward;
            display_reward[kThisAgentIdxOffset + 4] += goal_move_reward;
          }

          // Floor reward
          int new_agent_grid_x = min(int(grid_length_max_y) - 1, int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5));
          int new_agent_grid_y = min(int(grid_length_max_y) - 1, int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5));
          int origin_agent_grid_x = min(int(grid_length_max_y) - 1, int(origin_position_x / float(kGridDensity) + 0.5));
          int origin_agent_grid_y = min(int(grid_length_max_y) - 1, int(origin_position_y / float(kGridDensity) + 0.5));
          if (TrainingPhase > 0)
          {
            int self_agent_id = int(grid_length_max_y * new_agent_grid_y + new_agent_grid_x);
            int origin_agent_id = int(grid_length_max_y * origin_agent_grid_y + origin_agent_grid_x);
            if(goal_hit_wall == false && goal_hit == false)
            {
              float map_diff = abs(grid_map_z[self_agent_id] - grid_map_z[origin_agent_id]);
              rewards_arr[kThisAgentArrayIdx] += kRewardOnFloor * map_diff;
              display_reward[kThisAgentIdxOffset + 5] += kRewardOnFloor * map_diff;
            }
          }
        }
        else if(is_spot)
        {
          // 動態障礙物
          int hit_obstacle_ = 0;
          if(dist_ass <= 0)
          {

            display_reward[kThisAgentIdxOffset + 6] += spot_penalty_hit;
            hit_obstacle_ = 1;
          }
          if(dist_obstacle <= 0 && hit_obstacle_ == 0)
          {
            display_reward[kThisAgentIdxOffset + 6] += spot_penalty_hit;
            hit_obstacle_ = 1;
          }

          // 靜態障礙物 
          if(hit_obstacle_ == 0)
          {
            int left_x = min(int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity)), int(grid_length_max_y) - 1);
            int down_y = min(int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity)), int(grid_length_max_y) - 1);
            int right_x = min(left_x + 1, int(grid_length_max_y) - 1);
            int up_y = min(down_y + 1, int(grid_length_max_y) - 1);
            int nomal_agent_grid_x = min(int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1);
            int nomal_agent_grid_y = min(int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1);

            float dx = loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) - left_x;
            float dy = loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) - down_y;
            for (int i = 0; i < num_angle_lidar; i++)
            {
              float ld_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * down_y + num_angle_lidar * left_x + i)];
              float lu_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * up_y + num_angle_lidar * left_x + i)];
              float rd_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * down_y + num_angle_lidar * right_x + i)];
              float ru_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * up_y + num_angle_lidar * right_x + i)];
              float nomal_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * nomal_agent_grid_y + num_angle_lidar * nomal_agent_grid_x + i)];

              float precise_angle_distance =  dx * dy * ru_distance + (1 - dx) * dy * lu_distance + dx * (1 - dy) * rd_distance + (1 - dx) * (1 - dy) * ld_distance;
              float max_distance = max(max(ld_distance, lu_distance), max(rd_distance, ru_distance));
              float min_distance = min(min(ld_distance, lu_distance), min(rd_distance, ru_distance));
              float smallest_distance = float(kGridDensity) *  min(min(dx, 1 - dx), min(dy, 1 - dy));
              if (max_distance - min_distance > square_root_2 * kGridDensity)
              {
                if (nomal_angle_distance - smallest_distance < agent_size_arr[kThisAgentId] )
                {
                  display_reward[kThisAgentIdxOffset + 7] += spot_penalty_hit;
                  hit_obstacle_ = 1;
                  break;
                }
              }
              else
              {
                if (precise_angle_distance < agent_size_arr[kThisAgentId])
                {
                  display_reward[kThisAgentIdxOffset + 7] += spot_penalty_hit;
                  hit_obstacle_ = 1;
                  break;
                }
              }
            }
          }
          if(hit_obstacle_ == 0)
          {
            float head_x = loc_x_arr[kThisAgentArrayIdx] + cos(direction_arr[kThisAgentArrayIdx]) * spot_head_distance;
            float head_y = loc_y_arr[kThisAgentArrayIdx] + sin(direction_arr[kThisAgentArrayIdx]) * spot_head_distance;
            int left_x = max(min(int(head_x / float(kGridDensity)), int(grid_length_max_y) - 1), 0);
            int down_y = max(min(int(head_y / float(kGridDensity)), int(grid_length_max_y) - 1), 0);
            int right_x = max(min(left_x + 1, int(grid_length_max_y) - 1), 0);
            int up_y = max(min(down_y + 1, int(grid_length_max_y) - 1), 0);
            int nomal_agent_grid_x = max(min(int(head_x / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1), 0);
            int nomal_agent_grid_y = max(min(int(head_y / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1), 0);

            float dx = max(min(int(head_x / float(kGridDensity)), int(grid_length_max_y) - 1), 0) - left_x;
            float dy = max(min(int(head_y / float(kGridDensity)), int(grid_length_max_y) - 1), 0) - down_y;
            for (int i = 0; i < num_angle_lidar; i++)
            {
              float ld_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * down_y + num_angle_lidar * left_x + i)];
              float lu_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * up_y + num_angle_lidar * left_x + i)];
              float rd_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * down_y + num_angle_lidar * right_x + i)];
              float ru_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * up_y + num_angle_lidar * right_x + i)];
              float nomal_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * nomal_agent_grid_y + num_angle_lidar * nomal_agent_grid_x + i)];

              float precise_angle_distance =  dx * dy * ru_distance + (1 - dx) * dy * lu_distance + dx * (1 - dy) * rd_distance + (1 - dx) * (1 - dy) * ld_distance;
              float max_distance = max(max(ld_distance, lu_distance), max(rd_distance, ru_distance));
              float min_distance = min(min(ld_distance, lu_distance), min(rd_distance, ru_distance));
              float smallest_distance = float(kGridDensity) *  min(min(dx, 1 - dx), min(dy, 1 - dy));
              if (max_distance - min_distance > square_root_2 * kGridDensity)
              {
                if (nomal_angle_distance - smallest_distance < agent_size_arr[kThisAgentId] )
                {
                  display_reward[kThisAgentIdxOffset + 7] += spot_penalty_hit;
                  hit_obstacle_ = 1;
                  break;
                }
              }
              else
              {
                if (precise_angle_distance < agent_size_arr[kThisAgentId])
                {
                  display_reward[kThisAgentIdxOffset + 7] += spot_penalty_hit;
                  hit_obstacle_ = 1;
                  break;
                }
              }
            }
          }
          if(hit_obstacle_ == 0)
          {
            float ass_x = loc_x_arr[kThisAgentArrayIdx] - cos(direction_arr[kThisAgentArrayIdx]) * spot_ass_distance;
            float ass_y = loc_y_arr[kThisAgentArrayIdx] - sin(direction_arr[kThisAgentArrayIdx]) * spot_ass_distance;
            int left_x = max(min(int(ass_x / float(kGridDensity)), int(grid_length_max_y) - 1), 0);
            int down_y = max(min(int(ass_y / float(kGridDensity)), int(grid_length_max_y) - 1), 0);
            int right_x = max(min(left_x + 1, int(grid_length_max_y) - 1), 0);
            int up_y = max(min(down_y + 1, int(grid_length_max_y) - 1), 0);
            int nomal_agent_grid_x = max(min(int(ass_x / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1), 0);
            int nomal_agent_grid_y = max(min(int(ass_y / float(kGridDensity) + 0.5), int(grid_length_max_y) - 1), 0);

            float dx = max(min(int(ass_x / float(kGridDensity)), int(grid_length_max_y) - 1), 0) - left_x;
            float dy = max(min(int(ass_y / float(kGridDensity)), int(grid_length_max_y) - 1), 0) - down_y;
            for (int i = 0; i < num_angle_lidar; i++)
            {
              float ld_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * down_y + num_angle_lidar * left_x + i)];
              float lu_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * up_y + num_angle_lidar * left_x + i)];
              float rd_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * down_y + num_angle_lidar * right_x + i)];
              float ru_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * up_y + num_angle_lidar * right_x + i)];
              float nomal_angle_distance = different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * nomal_agent_grid_y + num_angle_lidar * nomal_agent_grid_x + i)];

              float precise_angle_distance =  dx * dy * ru_distance + (1 - dx) * dy * lu_distance + dx * (1 - dy) * rd_distance + (1 - dx) * (1 - dy) * ld_distance;
              float max_distance = max(max(ld_distance, lu_distance), max(rd_distance, ru_distance));
              float min_distance = min(min(ld_distance, lu_distance), min(rd_distance, ru_distance));
              float smallest_distance = float(kGridDensity) *  min(min(dx, 1 - dx), min(dy, 1 - dy));
              if (max_distance - min_distance > square_root_2 * kGridDensity)
              {
                if (nomal_angle_distance - smallest_distance < agent_size_arr[kThisAgentId] )
                {

                  display_reward[kThisAgentIdxOffset + 7] += spot_penalty_hit;
                  hit_obstacle_ = 1;
                  break;
                }
              }
              else
              {
                if (precise_angle_distance < agent_size_arr[kThisAgentId])
                {
                  display_reward[kThisAgentIdxOffset + 7] += spot_penalty_hit;
                  hit_obstacle_ = 1;
                  break;
                }
              }
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
              display_reward_ = kCostOperate;
            }
            rewards_arr[kThisAgentArrayIdx] += display_reward_;
            display_reward[kThisAgentIdxOffset + 9] += display_reward_;
          }

          // Floor reward
          float nomal_speed_x = abs(speed_x_arr[kThisAgentArrayIdx]) / kMaxSpeed_x;
          int new_agent_grid_x = min(int(grid_length_max_y) - 1, int(loc_x_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5));
          int new_agent_grid_y = min(int(grid_length_max_y) - 1, int(loc_y_arr[kThisAgentArrayIdx] / float(kGridDensity) + 0.5));
          int self_agent_id = int(grid_length_max_y * new_agent_grid_y + new_agent_grid_x);
          int origin_agent_grid_x = min(int(grid_length_max_y) - 1, int(origin_position_x / float(kGridDensity) + 0.5));
          int origin_agent_grid_y = min(int(grid_length_max_y) - 1, int(origin_position_y / float(kGridDensity) + 0.5));
          int origin_agent_id = int(grid_length_max_y * origin_agent_grid_y + origin_agent_grid_x);

          if(hit_obstacle_ == 0)
          {
            float cos_dir = cos(direction_arr[kThisAgentArrayIdx]);
            float sin_dir = sin(direction_arr[kThisAgentArrayIdx]);
            float agent_pitch = MapDx(grid_map_z, grid_length_max_y, grid_length_height, new_agent_grid_x, new_agent_grid_y) * cos_dir +
              MapDy(grid_map_z, grid_length_max_y, grid_length_height, new_agent_grid_x, new_agent_grid_y) * sin_dir;
            if (agent_pitch >= 0)  
            {
              float map_diff = abs(grid_map_z[self_agent_id] - grid_map_z[origin_agent_id]);
              rewards_arr[kThisAgentArrayIdx] += spot_reward_on_floor * map_diff;
              display_reward[kThisAgentIdxOffset + 10] += spot_reward_on_floor * map_diff;
              if(dist_goal <= 0 && agent_pitch > 0)  display_reward[kThisAgentIdxOffset + 10] += kRewardSpotGetGoal;
            }
            else 
            {
              hit_obstacle_ = 1;
              display_reward[kThisAgentIdxOffset + 10] += spot_penalty_hit;
              
            }
          }
          if(hit_obstacle_ == 1)
          {
            rewards_arr[kThisAgentArrayIdx] += spot_penalty_hit;
            still_in_the_game_arr[kThisAgentArrayIdx] = 0;
            num_hit_obstacle[kEnvId] += 1;
          }
          
          // Goal reward
          if(dist_goal <= 0 && hit_obstacle_ == 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kRewardSpotGetGoal;
            display_reward[kThisAgentIdxOffset + 8] += kRewardSpotGetGoal;
            num_get_goal[kEnvId] += 1;
            if (Testing_type > 1 && Training == 0) 
            {
              loc_x_arr[nearest_goal_idx] = static_cast<float>((static_cast<int>(random_agent_seed * 135.2 + static_cast<float>(num_get_goal[kEnvId]) * 17)) % static_cast<int>(kGridLength));
              loc_y_arr[nearest_goal_idx] = static_cast<float>((static_cast<int>(random_agent_seed * 179.9 + static_cast<float>(num_get_goal[kEnvId]) * 23)) % static_cast<int>(kGridLength));
            }
            else
            {
              still_in_the_game_arr[nearest_goal_idx] = 0;
            }
          }
          
        }
        else if(is_wall)
        {

          // Move reward
          float nomal_speed_x = abs(speed_x_arr[kThisAgentArrayIdx] + 0.07) / kMaxSpeed_x;
          float nomal_acc_x = abs(acceleration_x_arr[kThisAgentArrayIdx]) / MaxAcceleration_x;
          //float obs_move_reward = 2 * kRewardGoalSpeed * (nomal_speed_x + 1 - nomal_acc_x);
          float obs_move_reward = 0.3 * (nomal_speed_x + 0.3 * nomal_acc_x);
          rewards_arr[kThisAgentArrayIdx] += obs_move_reward;

        }
      }
      else if (env_timestep_arr[kEnvId] == 2 && is_spot) display_reward[kThisAgentIdxOffset + 13] = 1.0;  // 計算 spot 出生死亡率

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

  __global__ void CudaSpot_3dInLayerStep(
    int * done_arr,
    int * env_timestep_arr,
    int * num_spot,
    int * num_hit_obstacle,
    int * num_get_goal,
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
    float * grid_map_z,
    float * different_point_angle_distance,
    const float Testing_type,
    const float Training,
    const float TrainingPhase,
    const float kGridLength,
    const float kGridDensity,
    const float kNumOtherAgentsObserved,
    const float rl_fps,
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
    const float spot_reward_on_floor,
    const float spot_penalty_hit,
    const float goal_penalty_hit,
    const float kCostOperate,
    const float kCostSpeedYRateWithAccX,
    const float kCostTurnRateWithAccX,
    const float kRewardGoalStep,
    const float kRewardGoalSpeed,
    const float kRewardOnFloor,
    const float grid_length_max_y,
    const float grid_length_height,
    const float four_sign_distance,
    const float spot_head_distance,
    const float spot_ass_distance,
    const float random_agent_seed,
    int kNumAgents,
    int kEpisodeLength,
    int * action_indices_arr,
    int * still_in_the_game_arr,
    int * neighbor_ids_sorted_by_distance_arr,
    int * nearest_neighbor_ids,
    int * agent_types_arr
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

    // Wait here until timestep has been updated
    __sync_env_threads();

    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <= kEpisodeLength);
    const float origin_position_x = loc_x_arr[kThisAgentArrayIdx];
    const float origin_position_y = loc_y_arr[kThisAgentArrayIdx];
    
    if (kThisAgentId < kNumAgents) {
      int kThisAgentActionIdxOffset = kEnvId * kNumAgents * kNumActions + kThisAgentId * kNumActions;
      float delta_acceleration_x = acceleration_actions_x_arr[action_indices_arr[ kThisAgentActionIdxOffset + 0]];
      float delta_acceleration_y = acceleration_actions_y_arr[action_indices_arr[ kThisAgentActionIdxOffset + 1]];
      delta_turn = turn_actions_arr[action_indices_arr[ kThisAgentActionIdxOffset + 2]] * skill_levels_arr[kThisAgentId];

      acceleration_x_arr[kThisAgentArrayIdx] = delta_acceleration_x;
      acceleration_y_arr[kThisAgentArrayIdx] = delta_acceleration_y;
      direction_arr[kThisAgentArrayIdx] = fmodf(
        direction_arr[kThisAgentArrayIdx] + delta_turn, kTwoPi);
      if (direction_arr[kThisAgentArrayIdx] < 0) {
        direction_arr[kThisAgentArrayIdx] = kTwoPi + direction_arr[
          kThisAgentArrayIdx];
      }
      
      // Speed clipping
      float skill_action = skill_levels_arr[kThisAgentId];
      if (agent_types_arr[kThisAgentId] == 2) skill_action = 1.0;
      float old_speed_x = speed_x_arr[kThisAgentArrayIdx];
      float old_speed_y = speed_y_arr[kThisAgentArrayIdx];
      
      speed_x_arr[kThisAgentArrayIdx] = min(kMaxSpeed_x * skill_levels_arr[kThisAgentId], max( kMinSpeed_x * skill_levels_arr[kThisAgentId], speed_x_arr[kThisAgentArrayIdx] + acceleration_x_arr[kThisAgentArrayIdx] * skill_action));
      speed_y_arr[kThisAgentArrayIdx] = min(kMaxSpeed_y * skill_levels_arr[kThisAgentId], max( kMinSpeed_y * skill_levels_arr[kThisAgentId], speed_y_arr[kThisAgentArrayIdx] + acceleration_y_arr[kThisAgentArrayIdx] * skill_action));
      if (agent_types_arr[kThisAgentId] != 1 && speed_x_arr[kThisAgentArrayIdx] < 0) speed_x_arr[kThisAgentArrayIdx] = 0.0;
      // acceleration clipping
      if (speed_x_arr[kThisAgentArrayIdx] <= kMinSpeed_x * skill_levels_arr[kThisAgentId])  {
          acceleration_x_arr[kThisAgentArrayIdx] = speed_x_arr[kThisAgentArrayIdx] - old_speed_x;
      }
      else if (speed_x_arr[kThisAgentArrayIdx] >= kMaxSpeed_x * skill_levels_arr[kThisAgentId]) {
          acceleration_x_arr[kThisAgentArrayIdx] = speed_x_arr[kThisAgentArrayIdx] - old_speed_x;
      }
      if (speed_y_arr[kThisAgentArrayIdx] <= kMinSpeed_y * skill_levels_arr[kThisAgentId]){
          acceleration_y_arr[kThisAgentArrayIdx] = speed_y_arr[kThisAgentArrayIdx] - old_speed_y;
      }
      else if (speed_y_arr[kThisAgentArrayIdx] >= kMaxSpeed_y * skill_levels_arr[kThisAgentId]) {
          acceleration_y_arr[kThisAgentArrayIdx] = speed_y_arr[kThisAgentArrayIdx] - old_speed_y;
      }

      float frame_speed_x;
      float frame_speed_y;
      if (agent_types_arr[kThisAgentId] != 1)  // not spot
      {
        
        speed_y_arr[kThisAgentArrayIdx] = 0.0;
        acceleration_y_arr[kThisAgentArrayIdx] = 0.0;
        //if (agent_types_arr[kThisAgentId] == 2)  // obstacles
        //{
          bool need_cheak_obstacles_hit_wall = true;
          int random_seed = 5;
          while (need_cheak_obstacles_hit_wall)
          {
            random_seed += 1;
            frame_speed_x = speed_x_arr[kThisAgentArrayIdx] / rl_fps;
            float _new_loc_x_arr = loc_x_arr[kThisAgentArrayIdx] + frame_speed_x * cos(direction_arr[kThisAgentArrayIdx]) ;
            float _new_loc_y_arr = loc_y_arr[kThisAgentArrayIdx] + frame_speed_x * sin(direction_arr[kThisAgentArrayIdx]) ;
            int new_agent_grid_x = max(0, min(int(_new_loc_x_arr / kGridDensity + 0.5), int(grid_length_max_y) - 1));
            int new_agent_grid_y = max(0, min(int(_new_loc_y_arr / kGridDensity + 0.5), int(grid_length_max_y) - 1));
            int new_hit_wall_ = 0;
            int or_agent_grid_x = max(0, min(int(loc_x_arr[kThisAgentArrayIdx] / kGridDensity + 0.5), int(grid_length_max_y) - 1));
            int or_agent_grid_y = max(0, min(int(loc_y_arr[kThisAgentArrayIdx] / kGridDensity + 0.5), int(grid_length_max_y) - 1));
            int or_hit_wall_ = 0;

            for (int i = 0; i < num_angle_lidar; i++)
            {
              if(different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * or_agent_grid_y + num_angle_lidar *  or_agent_grid_x + i)] < agent_size_arr[kThisAgentId])
              {
                or_hit_wall_ = 1;
              }
              if(different_point_angle_distance[int(num_angle_lidar * grid_length_max_y * new_agent_grid_y + num_angle_lidar * new_agent_grid_x + i)] < agent_size_arr[kThisAgentId])
              {
                new_hit_wall_ = 1;
                break;
              }
            }

            if (new_hit_wall_ == 1 && or_hit_wall_ == 0)
            {
              direction_arr[kThisAgentArrayIdx] = fmodf(direction_arr[kThisAgentArrayIdx] + kPi + int(loc_y_arr[kThisAgentArrayIdx + random_seed]), kTwoPi);  // 隨機角度
              if (agent_types_arr[kThisAgentId] == 2) speed_x_arr[kThisAgentArrayIdx] = speed_x_arr[kThisAgentArrayIdx] * 0.1;
            }
            if (or_hit_wall_ == 1 || new_hit_wall_ == 0)
            {
              break;
            }
          }
          
        //}

        loc_x_arr[kThisAgentArrayIdx] += frame_speed_x * cos(direction_arr[kThisAgentArrayIdx]);
        loc_y_arr[kThisAgentArrayIdx] += frame_speed_x * sin(direction_arr[kThisAgentArrayIdx]);

      }
      else
      {
        float speed_x_ = speed_x_arr[kThisAgentArrayIdx] ;
        if (speed_x_ < 0) speed_x_ = speed_x_ * kMaxSpeed_x / kMinSpeed_x;
        float total_speed = sqrt(pow(speed_x_, 2) + pow(speed_y_arr[kThisAgentArrayIdx] * kMaxSpeed_x / kMaxSpeed_y, 2));
        if (total_speed > kMaxSpeed_x)
        {
          speed_x_arr[kThisAgentArrayIdx] = speed_x_arr[kThisAgentArrayIdx] * kMaxSpeed_x / total_speed;
          speed_y_arr[kThisAgentArrayIdx] = speed_y_arr[kThisAgentArrayIdx] * kMaxSpeed_x / total_speed;
        }
        frame_speed_x = speed_x_arr[kThisAgentArrayIdx] / rl_fps;
        frame_speed_y = speed_y_arr[kThisAgentArrayIdx] / rl_fps;

        loc_x_arr[kThisAgentArrayIdx] += frame_speed_x * cos(direction_arr[kThisAgentArrayIdx]) - frame_speed_y * sin(direction_arr[kThisAgentArrayIdx]);
        loc_y_arr[kThisAgentArrayIdx] += frame_speed_x * sin(direction_arr[kThisAgentArrayIdx]) + frame_speed_y * cos(direction_arr[kThisAgentArrayIdx]);
      }
      // Crossing the edge

      if (loc_x_arr[kThisAgentArrayIdx] < 0) loc_x_arr[kThisAgentArrayIdx] = 0.0;
      if (loc_x_arr[kThisAgentArrayIdx] > kGridLength) loc_x_arr[kThisAgentArrayIdx] = kGridLength;
      if (loc_y_arr[kThisAgentArrayIdx] < 0)  loc_y_arr[kThisAgentArrayIdx] = 0.0;
      if (loc_y_arr[kThisAgentArrayIdx] > kGridLength) loc_y_arr[kThisAgentArrayIdx] = kGridLength;
      //printf("state ");

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
      Testing_type,
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
      four_sign_distance,
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
      origin_position_x,
      origin_position_y,
      Testing_type,
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
      spot_reward_on_floor,
      spot_penalty_hit,
      goal_penalty_hit,
      kCostOperate,
      kCostSpeedYRateWithAccX,
      kCostTurnRateWithAccX,
      kRewardGoalStep,
      kRewardGoalSpeed,
      kRewardOnFloor,
      grid_length_max_y,
      grid_length_height,
      spot_head_distance,
      spot_ass_distance,
      random_agent_seed,
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
