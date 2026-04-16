// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause
#include <math.h>
__constant__ float kPi = 3.141592654;
__constant__ float kTwoPi = 6.283185308;
__constant__ float kEpsilon = 1.0e-10;  // to prevent indeterminate cases

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

  __device__ void CudaTagContinuousGenerateObservation(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * direction_arr,
    float * acceleration_arr,
    bool Training,
    float give_car_one_fix_goal,
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
    const float testing_observation_eff,
    const float MaxAcceleration,
    int * env_timestep_arr,
    const int kNumAgents,
    const int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {
    int num_features = 7;
    if (kThisAgentId < kNumAgents) {
        const int kThisAgentIdxOffset = kEnvId * kNumAgents *
          (num_features * (kNumOtherAgentsObserved ) + 9) +
          kThisAgentId * (num_features * (kNumOtherAgentsObserved ) + 9);

        obs_arr[kThisAgentIdxOffset] = 0.0;
        obs_arr[kThisAgentIdxOffset + 1] = 0.0;
        obs_arr[kThisAgentIdxOffset + 2] = 0.0;
        obs_arr[kThisAgentIdxOffset + 3] = 0.0;
        obs_arr[kThisAgentIdxOffset + 4] = 0.0;
        obs_arr[kThisAgentIdxOffset + 5] = 0.0;
        obs_arr[kThisAgentIdxOffset + 6] = 0.0;
        obs_arr[kThisAgentIdxOffset + 7] = 0.0;
        obs_arr[kThisAgentIdxOffset + 8] = 0.0;

        for (int idx = 0; idx < kNumOtherAgentsObserved; idx++) {
          obs_arr[kThisAgentIdxOffset + 9 + 0 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 9 + 1 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 9 + 2 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 9 + 3 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 9 + 4 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 9 + 5 * kNumOtherAgentsObserved + idx] = 0.0;
          obs_arr[kThisAgentIdxOffset + 9 + 6 * kNumOtherAgentsObserved + idx] = 0.0;
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
          float give_car_one_fix_goal_reset = give_car_one_fix_goal;
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
            if (agent_types_arr[neighbor_ids_sorted_by_distance_arr[i_index]] == 1) {
              check_car_can_look_goal = true;
            }
            if (i == kNumOtherAgentsObserved -1 &&
              check_car_can_look_goal == false &&
              agent_types_arr[kThisAgentId] == 2
            ){
              int k = num_valid_other_agents;
              for(int j = i + 1; j < k; j++) {
                j_index = kThisAgentArrayIdxOffset + j;
                if (agent_types_arr[neighbor_ids_sorted_by_distance_arr[j_index]] == 1){
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
            // float agent_sizes = agent_size_arr[kOtherAgentId] + agent_size_arr[kThisAgentId] ;
            float agent_sizes = agent_size_arr[kOtherAgentId] ;

            if (agent_types_arr[kOtherAgentId] == 1)
            {
              agent_types = 0;
            }
            else
            {
              agent_types = 1;
            }
       
            if (give_car_one_fix_goal == 1 && agent_types_arr[kOtherAgentId] == 1 && goal_x_ == 0 && goal_y_ == 0){
              give_car_one_fix_goal = 0;
              goal_x_ = static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] - loc_x_arr[kThisAgentArrayIdx]);
              goal_y_ = static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] - loc_y_arr[kThisAgentArrayIdx]);
            }
            else if (idx == min(num_valid_other_agents,kNumOtherAgentsObserved) - 1 && give_car_one_fix_goal == 1 ){
              goal_x_ = 0.0;
              goal_y_ = 0.0;
            }

            obs_arr[kThisAgentIdxOffset + 9 + 0 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] -
              loc_x_arr[kThisAgentArrayIdx]);
            obs_arr[kThisAgentIdxOffset + 9 + 1 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] -
              loc_y_arr[kThisAgentArrayIdx]);
            obs_arr[kThisAgentIdxOffset + 9 + 2 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(speed_arr[kOtherAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 9 + 3 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(direction_arr[kOtherAgentArrayIdx]) / (kTwoPi);
            obs_arr[kThisAgentIdxOffset + 9 + 4 * kNumOtherAgentsObserved + idx] =
              agent_types;
            obs_arr[kThisAgentIdxOffset + 9 + 5 * kNumOtherAgentsObserved + idx] =
              still_in_the_game_arr[kOtherAgentArrayIdx];
            obs_arr[kThisAgentIdxOffset + 9 + 6 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(agent_sizes);
          }
          obs_arr[kThisAgentIdxOffset] = static_cast<float>(acceleration_arr[kThisAgentArrayIdx]) / (MaxAcceleration + kEpsilon); 
          obs_arr[kThisAgentIdxOffset + 1] = static_cast<float>(speed_arr[kThisAgentArrayIdx] / (kMaxSpeed + kEpsilon));
          obs_arr[kThisAgentIdxOffset + 2] = static_cast<float>(direction_arr[kThisAgentArrayIdx]) / (kTwoPi);
          obs_arr[kThisAgentIdxOffset + 3] = static_cast<float>(agent_size_arr[kThisAgentId]);
          obs_arr[kThisAgentIdxOffset + 4] = goal_x_;
          obs_arr[kThisAgentIdxOffset + 5] = goal_y_;
          obs_arr[kThisAgentIdxOffset + 6] = static_cast<float>(loc_x_arr[kThisAgentArrayIdx]) / (kGridLength);
          obs_arr[kThisAgentIdxOffset + 7] = static_cast<float>(loc_y_arr[kThisAgentArrayIdx]) / (kGridLength);
          obs_arr[kThisAgentIdxOffset + 8] = static_cast<float>(env_timestep_arr[kEnvId]) / kEpisodeLength;
          if (Training == false)
          {
            obs_arr[kThisAgentIdxOffset + 8] = static_cast<float>(0);
          }
          give_car_one_fix_goal = give_car_one_fix_goal_reset;
        }
    }
  }

  // Device helper function to compute rewards
  __device__ void CudaTagContinuousComputeReward(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * acceleration_arr,
    bool Training,
    int * agent_types_arr,
    float * agent_size_arr,
    const float kGridLength,
    float delta_turn,
    const float kMaxSpeed,
    int * still_in_the_game_arr,
    float * rewards_arr,
    float * step_rewards_arr,
    const float distance_eff_resize,
    const float testing_acceleration_rw_eff,
    const float kRewardTaggerGetGoal,
    const float kPenaltyAgentHit,
    const float kPenaltyGoalBeCatch,
    const float kRewardCarEndTheGame,
    const float kRewardLimitAcceleration,
    const float kRewardLimitTurn,
    const float MaxAcceleration,
    const float MinAcceleration,
    const float MaxTurn,
    int * num_hit_wall,
    int * num_get_goal,
    int * num_taggers,
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

      bool is_wall = agent_types_arr[kThisAgentId] == 0;
      bool is_goal = agent_types_arr[kThisAgentId] == 1 ;
      bool is_tagger = agent_types_arr[kThisAgentId] == 2;
      float dist_wall = kGridLength * sqrt(2.0);
      float dist_goal = dist_wall;
      float dist_tagger = dist_wall;
      int nearest_wall_idx = 0;
      int nearest_goal_idx = 0;
      int nearest_tagger_idx = 0;

      if (still_in_the_game_arr[kThisAgentArrayIdx]) {
        rewards_arr[kThisAgentArrayIdx] += step_rewards_arr[kThisAgentId];
        float nomal_turn_ = delta_turn / MaxTurn;
        float nomal_acc_ = acceleration_arr[kThisAgentArrayIdx] / max(abs(MinAcceleration), MaxAcceleration);

        rewards_arr[kThisAgentArrayIdx] += kRewardLimitAcceleration * (1 - abs(nomal_acc_)) * (1 - abs(nomal_acc_));
        rewards_arr[kThisAgentArrayIdx] += kRewardLimitTurn * (1 - abs(nomal_turn_)) * (1 - abs(nomal_turn_)) ;

        for (int other_agent_id = 0; other_agent_id < kNumAgents; other_agent_id++) {
          bool other_is_wall = agent_types_arr[other_agent_id] == 0;
          bool other_is_goal = agent_types_arr[other_agent_id] == 1;
          bool other_is_tagger = agent_types_arr[other_agent_id] == 2;
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
          if (other_is_wall && dist < dist_wall && other_agent_id != kThisAgentId && still_in_the_game_arr[OtherAgentArrayIdx]){
            dist_wall = dist;
            nearest_wall_idx = OtherAgentArrayIdx;
          }
          if(other_is_goal && dist < dist_goal && other_agent_id != kThisAgentId && still_in_the_game_arr[OtherAgentArrayIdx]){
            dist_goal = dist;
            nearest_goal_idx = OtherAgentArrayIdx;
          }
          if(other_is_tagger && dist < dist_tagger && other_agent_id != kThisAgentId && still_in_the_game_arr[OtherAgentArrayIdx]){
            dist_tagger = dist;
            nearest_tagger_idx = OtherAgentArrayIdx;
          }
        }
        if(is_goal)
        {
          if(dist_wall <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit * (speed_arr[nearest_wall_idx] + speed_arr[kThisAgentArrayIdx]) / kMaxSpeed;
          }
          if(dist_tagger <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kPenaltyGoalBeCatch;
            
          }
        }
        else if(is_tagger)
        {
          if(dist_wall <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
            num_hit_wall[kEnvId] += 1;
            if (Training == true) 
            {
              still_in_the_game_arr[kThisAgentArrayIdx] = 0;
              num_taggers[kEnvId] -= 1;
            }
          }
          if(dist_goal <= 0)
          {
            rewards_arr[kThisAgentArrayIdx] += kRewardTaggerGetGoal;
            num_get_goal[kEnvId] += 1;
            if (Training == true) 
            {
              still_in_the_game_arr[nearest_goal_idx] = 0;
            }
            else
            {
              
              loc_x_arr[nearest_goal_idx] = float(int(loc_y_arr[nearest_goal_idx] + 21.02) % int(kGridLength));
              loc_y_arr[nearest_goal_idx] = float(int(loc_x_arr[nearest_goal_idx] + 2.35 + speed_arr[nearest_goal_idx]) % int(kGridLength));
            }
          }
          if(dist_tagger <= 0)
          {
            if (Training == true && still_in_the_game_arr[kThisAgentArrayIdx] == 1) 
            {
              rewards_arr[kThisAgentArrayIdx] += kPenaltyAgentHit;
              rewards_arr[nearest_tagger_idx] += kPenaltyAgentHit;
              still_in_the_game_arr[nearest_tagger_idx] = 0;
              still_in_the_game_arr[kThisAgentArrayIdx] = 0;
              num_taggers[kEnvId] -= 2;
            }
            //rewards_arr[kThisAgentArrayIdx] += 0.3 * kPenaltyAgentHit * (1 - 10 * dist_tagger + (speed_arr[nearest_tagger_idx] + speed_arr[kThisAgentArrayIdx]) / kMaxSpeed);
            num_hit_wall[kEnvId] += 2;
          }
          if (env_timestep_arr[kEnvId] == kEpisodeLength) {
            rewards_arr[kThisAgentArrayIdx] += kRewardCarEndTheGame;
          }
        }
      }
      __sync_env_threads();
      // Use only agent 0's thread to set done_arr
      if (kThisAgentId == 0) {
        if (env_timestep_arr[kEnvId] == kEpisodeLength || num_taggers[kEnvId] <= 0) {
          done_arr[kEnvId] = 1;
        }
      }
    }
  }

  __global__ void CudaFinalCarStep(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * direction_arr,
    float * acceleration_arr,
    bool Training,
    const float testing_acceleration_rw_eff,
    const float testing_observation_eff,
    float give_car_one_fix_goal,
    const int Testing_type,
    int * agent_types_arr,
    float * agent_size_arr,
    const float kGridLength,
    float * acceleration_actions_arr,
    float * turn_actions_arr,
    const float kMaxSpeed,
    const float kMinSpeed,
    const int kNumOtherAgentsObserved,
    float * skill_levels_arr,
    int * still_in_the_game_arr,
    float * obs_arr,
    int * action_indices_arr,
    float * neighbor_distances_arr,
    int * neighbor_ids_sorted_by_distance_arr,
    int * nearest_neighbor_ids,
    float * rewards_arr,
    float * step_rewards_arr,
    const float distance_eff_resize,
    const float kRewardTaggerGetGoal,
    const float kPenaltyAgentHit,
    const float kPenaltyGoalBeCatch,
    const float kRewardCarEndTheGame,
    const float kRewardLimitAcceleration,
    const float kRewardLimitTurn,
    const float MaxAcceleration,
    const float MinAcceleration,
    const float MaxTurn,
    int * num_hit_wall,
    int * num_get_goal,
    int * num_taggers,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength
  ) {
    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;
    const int kNumActions = 2;
    float delta_turn = kTwoPi;

    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == 0) {
      env_timestep_arr[kEnvId] += 1;
    }
    printf("kNumOtherAgentsObserved %d , \n", Testing_type);

    // Wait here until timestep has been updated
    __sync_env_threads();

    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <= kEpisodeLength);
    
    if (kThisAgentId < kNumAgents) {
      int kThisAgentActionIdxOffset = kEnvId * kNumAgents * kNumActions +
        kThisAgentId * kNumActions;
      float delta_acceleration = acceleration_actions_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 0]];
      delta_turn = turn_actions_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 1]];

      //printf("%f, \n", delta_turn);

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
            kMinSpeed,
            speed_arr[kThisAgentArrayIdx] + acceleration_arr[
              kThisAgentArrayIdx])) * still_in_the_game_arr[kThisAgentArrayIdx];

      // Reset acceleration to 0 when speed becomes 0 or
      // kMaxSpeed (multiplied by skill levels)
      if ((speed_arr[kThisAgentArrayIdx] <= kMinSpeed) ||
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
      Training,
      give_car_one_fix_goal,
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
      testing_observation_eff,
      MaxAcceleration,
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
      Training,
      agent_types_arr,
      agent_size_arr,
      kGridLength, 
      delta_turn,
      kMaxSpeed,
      still_in_the_game_arr,
      rewards_arr,
      step_rewards_arr,
      distance_eff_resize,
      testing_acceleration_rw_eff,
      kRewardTaggerGetGoal,
      kPenaltyAgentHit,
      kPenaltyGoalBeCatch,
      kRewardCarEndTheGame,
      kRewardLimitAcceleration,
      kRewardLimitTurn,
      MaxAcceleration,
      MinAcceleration,
      MaxTurn,
      num_hit_wall,
      num_get_goal,
      num_taggers,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx);
  }
}
