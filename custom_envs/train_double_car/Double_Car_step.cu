// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

__constant__ float kPi = 3.141592654;
__constant__ float kTwoPi = 6.283185308;
__constant__ float kEpsilon = 1.0e-10;  // to prevent indeterminate cases

extern "C" {
  // Device helper function to compute distances between two agents
  __device__ float ComputeCanViewDistance(
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

    bool is_car1 = agent_types_arr[index1] == 0;
    bool is_car2 = agent_types_arr[index1] == 1;
    bool is_wall = agent_types_arr[index1] == 2;
    bool is_goal = agent_types_arr[index1] == 3;
    bool other_is_car1 = agent_types_arr[index2] == 0;
    bool other_is_car2 = agent_types_arr[index2] == 1;
    bool other_is_wall = agent_types_arr[index2] == 2;
    float min_dis;
    if (is_car1 && (other_is_car1 || other_is_wall) ||
      is_car2 && (other_is_car2 || other_is_wall) ||
      is_wall ||
      is_goal
    ){
      min_dis = sqrt(
        pow(loc_x_arr[index1] - loc_x_arr[index2], 2) +
        pow(loc_y_arr[index1] - loc_y_arr[index2], 2)
      ) - agent_size_arr[kThisAgentId1] - agent_size_arr[kThisAgentId2];
    }
    else
    {
      min_dis = sqrt(
        pow(loc_x_arr[index1] - loc_x_arr[index2], 2) +
        pow(loc_y_arr[index1] - loc_y_arr[index2], 2)
      );
    }
    return min_dis;
  }

  // Device helper function to generate observation
  __device__ void CudaTagContinuousGenerateObservation(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * direction_arr,
    float * acceleration_arr,
    int * agent_types_arr,
    float * agent_size_arr,
    const float kGridLength,
    const float kMaxSpeed,
    const int kNumOtherAgentsObserved,
    int * still_in_the_game_arr,
    float * obs_arr,
    float * neighbor_distances_arr,
    int * neighbor_ids_sorted_by_distance_arr,
    int * nearest_neighbor_ids,
    bool Training,
    int * env_timestep_arr,
    const int kNumAgents,
    const int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {
    int num_features = 8;
    if (kThisAgentId < kNumAgents) {
        // Initialize obs to all zeros
        // obs shape is (num_envs, kNumAgents,
        //   num_features * kNumOtherAgentsObserved + 1)
        const int kThisAgentIdxOffset = kEnvId * kNumAgents *
          ((num_features + 1) * kNumOtherAgentsObserved) +
          kThisAgentId * ((num_features + 1) * kNumOtherAgentsObserved);

        obs_arr[kThisAgentIdxOffset] = 0.0;
        obs_arr[kThisAgentIdxOffset + 1] = 0.0;
        obs_arr[kThisAgentIdxOffset + 2] = 0.0;
        obs_arr[kThisAgentIdxOffset + 3] = 0.0;
        obs_arr[kThisAgentIdxOffset + 4] = 0.0;
        obs_arr[kThisAgentIdxOffset + 5] = 0.0;
        obs_arr[kThisAgentIdxOffset + 6] = 0.0;
        obs_arr[kThisAgentIdxOffset + 7] = 0.0;
        for (int idx = 0; idx < kNumOtherAgentsObserved; idx++) {
          obs_arr[kThisAgentIdxOffset + 8 + 0 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 8 + 1 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 8 + 2 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 8 + 3 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 8 + 4 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 8 + 5 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 8 + 6 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 8 + 7 * kNumOtherAgentsObserved + idx] = 0.0;
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
            neighbor_distances_arr[distance_arr_idx] = ComputeCanViewDistance(
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
            if ((agent_types_arr[neighbor_ids_sorted_by_distance_arr[i_index]] == 3) ||
                (agent_types_arr[neighbor_ids_sorted_by_distance_arr[i_index]] == 0 && agent_types_arr[kThisAgentId] == 1) ||
                (agent_types_arr[neighbor_ids_sorted_by_distance_arr[i_index]] == 1 && agent_types_arr[kThisAgentId] == 0)
              ){
              check_car_can_look_goal = true;
            }
            if (i == kNumOtherAgentsObserved -1 &&
              check_car_can_look_goal == false &&
              (agent_types_arr[kThisAgentId] == 0 || agent_types_arr[kThisAgentId] == 1)
            ){
              int k = num_valid_other_agents;
              for(int j = i + 1; j < k; j++) {
                j_index = kThisAgentArrayIdxOffset + j;
                if (agent_types_arr[neighbor_ids_sorted_by_distance_arr[j_index]] == 3 ||
                    (agent_types_arr[neighbor_ids_sorted_by_distance_arr[j_index]] == 0 && agent_types_arr[kThisAgentId] == 1) ||
                    (agent_types_arr[neighbor_ids_sorted_by_distance_arr[j_index]] == 1 && agent_types_arr[kThisAgentId] == 0)
                  ){
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
            int kOtherAgentId = nearest_neighbor_ids[
              kNearestNeighborsIdx];

            const int kOtherAgentArrayIdx = kEnvId * kNumAgents + kOtherAgentId;
            int agent_types = 0;
            float agent_sizes = 0.0;
            if (agent_types_arr[kThisAgentId] == 2)
            {
              agent_types = 1;
              agent_sizes = 0.0;
            }
            else if (agent_types_arr[kThisAgentId] == agent_types_arr[kOtherAgentId] || agent_types_arr[kOtherAgentId] == 2)
            {
              agent_types = 1;
              agent_sizes = agent_size_arr[kOtherAgentId] + agent_size_arr[kThisAgentId];
            }
            else
            {
              agent_types = 0;
              agent_sizes = 0.0;

            }
            // goal look type 0 0 1 0   goal look size 0 0 wall_size 0

            obs_arr[kThisAgentIdxOffset + 8 + 0 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] -
              loc_x_arr[kThisAgentArrayIdx]) / (sqrt(2.0) * kGridLength);
            obs_arr[kThisAgentIdxOffset + 8 + 1 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] -
              loc_y_arr[kThisAgentArrayIdx]) / (sqrt(2.0) * kGridLength);
            obs_arr[kThisAgentIdxOffset + 8 + 2 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(speed_arr[kOtherAgentArrayIdx] -
              speed_arr[kThisAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 8 + 3 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(acceleration_arr[kOtherAgentArrayIdx] -
              acceleration_arr[kThisAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 8 + 4 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(direction_arr[kOtherAgentArrayIdx] -
              direction_arr[kThisAgentArrayIdx]) / (kTwoPi);
            obs_arr[kThisAgentIdxOffset + 8 + 5 * kNumOtherAgentsObserved + idx] =
              agent_types;
            obs_arr[kThisAgentIdxOffset + 8 + 6 * kNumOtherAgentsObserved + idx] =
              still_in_the_game_arr[kOtherAgentArrayIdx];
            obs_arr[kThisAgentIdxOffset + 8 + 7 * kNumOtherAgentsObserved + idx] =
              agent_sizes;
          }
          obs_arr[kThisAgentIdxOffset] = static_cast<float>(env_timestep_arr[kEnvId]) / kEpisodeLength;
          obs_arr[kThisAgentIdxOffset + 1] = agent_size_arr[kThisAgentId];  
          obs_arr[kThisAgentIdxOffset + 2] = 0;  
          obs_arr[kThisAgentIdxOffset + 3] = 0;  
          obs_arr[kThisAgentIdxOffset + 4] = 0;  
          obs_arr[kThisAgentIdxOffset + 5] = 0;  
          obs_arr[kThisAgentIdxOffset + 6] = 0;  
          obs_arr[kThisAgentIdxOffset + 7] = 0;
        }
    }
  }

  // Device helper function to compute rewards
  __device__ void CudaTagContinuousComputeReward(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * acceleration_arr,
    int * agent_types_arr,
    float * agent_size_arr,
    const float kGridLength,
    const float kMaxSpeed,
    int * still_in_the_game_arr,
    float * rewards_arr,
    float * step_rewards_arr,
    const float kDistanceMargin,
    const float kTagRewardForTagger,
    const float kTaggerHitPenalty,
    const float kLimitAccelerationReward,
    const float kLimitTurnReward,
    const float MaxAcceleration,
    const float MaxTurn,
    bool Training,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx,
    float delta_turn
  ) {
    if (kThisAgentId < kNumAgents) {
      // initialize rewards
      rewards_arr[kThisAgentArrayIdx] = 0.0;
      __sync_env_threads();
      
      bool is_car1 = agent_types_arr[kThisAgentId] == 0;
      bool is_car2 = agent_types_arr[kThisAgentId] == 1;
      bool is_wall = agent_types_arr[kThisAgentId] == 2;
      bool is_goal = agent_types_arr[kThisAgentId] == 3;
      float min_dist = kGridLength * sqrt(2.0);
      float dist_car1 = min_dist;
      float dist_car2 = min_dist;
      float dist_wall = min_dist;
      float dist_goal = min_dist;
      int nearest_car1_id = 0;
      int nearest_car2_id = 0;
      int nearest_wall_id = 0;
      int nearest_goal_id = 0;

      if (still_in_the_game_arr[kThisAgentArrayIdx]) {
        for (int other_agent_id = 0; other_agent_id < kNumAgents; other_agent_id++) {
          bool other_is_car1 = agent_types_arr[other_agent_id] == 0;
          bool other_is_car2 = agent_types_arr[other_agent_id] == 1;
          bool other_is_wall = agent_types_arr[other_agent_id] == 2;
          bool other_is_goal = agent_types_arr[other_agent_id] == 3;
          const float dist = ComputeCanViewDistance(
            agent_size_arr,
            loc_x_arr,
            loc_y_arr,
            kThisAgentId,
            other_agent_id,
            kEnvId,
            kNumAgents,
            agent_types_arr);
          if (other_is_car1 && dist < dist_car1) {
            dist_car1 = dist;
            nearest_car1_id = kEnvId * kNumAgents + other_agent_id;
          }
          if(other_is_car2 && dist < dist_car2){
            dist_car2 = dist;
            nearest_car2_id = kEnvId * kNumAgents + other_agent_id;
          }
          if(other_is_wall && dist < dist_wall){
            dist_wall = dist;
            nearest_wall_id = kEnvId * kNumAgents + other_agent_id;
          }
          if(other_is_goal && dist < dist_goal){
            dist_goal = dist;
            nearest_goal_id = kEnvId * kNumAgents + other_agent_id;
          }
        }
        if (is_car1)
        {
          if (dist_car1 < kDistanceMargin || dist_wall < kDistanceMargin)
          {
            rewards_arr[kThisAgentArrayIdx] += kTaggerHitPenalty;
          }
          if (dist_car2 < kDistanceMargin)
          {
            rewards_arr[kThisAgentArrayIdx] += kTagRewardForTagger;
            still_in_the_game_arr[kThisAgentArrayIdx] = 0;
            still_in_the_game_arr[nearest_car2_id] = 0;
          }
          if (dist_goal < kDistanceMargin)
          {
            rewards_arr[kThisAgentArrayIdx] += kTagRewardForTagger;
            still_in_the_game_arr[nearest_goal_id] = 0;
          }
        }
        else if(is_car2)
        {
          if (dist_car2 < kDistanceMargin || dist_wall < kDistanceMargin)
          {
            rewards_arr[kThisAgentArrayIdx] += kTaggerHitPenalty;
          }
          if (dist_car1 < kDistanceMargin)
          {
            rewards_arr[kThisAgentArrayIdx] += kTagRewardForTagger;
          }
          if (dist_goal < kDistanceMargin)
          {
            rewards_arr[kThisAgentArrayIdx] += kTagRewardForTagger;
            still_in_the_game_arr[nearest_goal_id] = 0;
          }
        }
        else if(is_goal)
        {
          if (dist_wall < kDistanceMargin)
          {
            rewards_arr[kThisAgentArrayIdx] += kTaggerHitPenalty * (0.3 + 1.2 * speed_arr[kThisAgentArrayIdx] / kMaxSpeed);
          }
        }
        else if(is_wall)
        {
          // Move reward
          float nomal_speed_x = abs(speed_arr[kThisAgentArrayIdx]  + 0.07);
          float nomal_acc_x = abs(acceleration_arr[kThisAgentArrayIdx]);
          //float obs_move_reward = 2 * kRewardGoalSpeed * (nomal_speed_x + 1 - nomal_acc_x);
          float obs_move_reward = 0.3 * (nomal_speed_x + 0.3 * nomal_acc_x);
          rewards_arr[kThisAgentArrayIdx] += obs_move_reward;

        }
      }
            
      // Wait here to update the number of runners before determining done_arr
      __sync_env_threads();
      // Use only agent 0's thread to set done_arr
      if (kThisAgentId == 0) {
        if (env_timestep_arr[kEnvId] == kEpisodeLength) {
          done_arr[kEnvId] = 1;
        }
      }
    }
  }

  __global__ void CudaDoubleCarStep(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * direction_arr,
    float * acceleration_arr,
    int * agent_types_arr,
    float * agent_size_arr,
    const float kGridLength,
    float * acceleration_actions_arr,
    float * turn_actions_arr,
    const float kMaxSpeed,
    const int kNumOtherAgentsObserved,
    float * skill_levels_arr,
    float * display_reward,
    int * still_in_the_game_arr,
    float * obs_arr,
    int * action_indices_arr,
    float * neighbor_distances_arr,
    int * neighbor_ids_sorted_by_distance_arr,
    int * nearest_neighbor_ids,
    float * rewards_arr,
    float * step_rewards_arr,
    const float kDistanceMargin,
    const float kTagRewardForTagger,
    const float kTaggerHitPenalty,
    const float kLimitAccelerationReward,
    const float kLimitTurnReward,
    const float MaxAcceleration,
    const float MaxTurn,
    bool Training,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength
  ) {
    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;
    const int kNumActions = 3;
    float delta_turn = kTwoPi;

    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == 0) {
      env_timestep_arr[kEnvId] += 1;
    }

    // Wait here until timestep has been updated
    __sync_env_threads();

    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <=
      kEpisodeLength);

    if (kThisAgentId < kNumAgents) {
      int kThisAgentActionIdxOffset = kEnvId * kNumAgents * kNumActions +
        kThisAgentId * kNumActions;
      float delta_acceleration = acceleration_actions_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 0]];
      delta_turn = turn_actions_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 2]];

      acceleration_arr[kThisAgentArrayIdx] += delta_acceleration;

      direction_arr[kThisAgentArrayIdx] = fmod(
        direction_arr[kThisAgentArrayIdx] + delta_turn, kTwoPi) *
        still_in_the_game_arr[kThisAgentArrayIdx];
      if (direction_arr[kThisAgentArrayIdx] < 0) {
        direction_arr[kThisAgentArrayIdx] = kTwoPi + direction_arr[
          kThisAgentArrayIdx];
      }

      // Speed clipping
      speed_arr[kThisAgentArrayIdx] = min(
          kMaxSpeed * skill_levels_arr[kThisAgentId],
          max(
            0.0,
            speed_arr[kThisAgentArrayIdx] + acceleration_arr[
              kThisAgentArrayIdx])) * still_in_the_game_arr[kThisAgentArrayIdx];

      // Reset acceleration to 0 when speed becomes 0 or
      // kMaxSpeed (multiplied by skill levels)
      if ((speed_arr[kThisAgentArrayIdx] <= 0.0) ||
        (speed_arr[kThisAgentArrayIdx] >=
        (kMaxSpeed * skill_levels_arr[kThisAgentId]))) {
          acceleration_arr[kThisAgentArrayIdx] = 0.0;
      }
      loc_x_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] * cos(direction_arr[kThisAgentArrayIdx]);
      loc_y_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] * sin(direction_arr[kThisAgentArrayIdx]);

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
    __sync_env_threads();

    CudaTagContinuousGenerateObservation(
      loc_x_arr,
      loc_y_arr,
      speed_arr,
      direction_arr,
      acceleration_arr,
      agent_types_arr,
      agent_size_arr,
      kGridLength,
      kMaxSpeed,
      kNumOtherAgentsObserved,
      still_in_the_game_arr,
      obs_arr,
      neighbor_distances_arr,
      neighbor_ids_sorted_by_distance_arr,
      nearest_neighbor_ids,
      Training,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx);

    CudaTagContinuousComputeReward(
      loc_x_arr,
      loc_y_arr,
      speed_arr,
      acceleration_arr,
      agent_types_arr,
      agent_size_arr,
      kGridLength,
      kMaxSpeed,
      still_in_the_game_arr,
      rewards_arr,
      step_rewards_arr,
      kDistanceMargin,
      kTagRewardForTagger,
      kTaggerHitPenalty,
      kLimitAccelerationReward,
      kLimitTurnReward,
      MaxAcceleration,
      MaxTurn,
      Training,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx,
      delta_turn);
  }
}
