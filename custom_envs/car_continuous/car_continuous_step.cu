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
  __device__ float ComputeDistance(
    float * loc_x_arr,
    float * loc_y_arr,
    const int kThisAgentId1,
    const int kThisAgentId2,
    const int kEnvId,
    const int kNumAgents
  ) {
    const int index1 = kEnvId * kNumAgents + kThisAgentId1;
    const int index2 = kEnvId * kNumAgents + kThisAgentId2;
    return sqrt(
      pow(loc_x_arr[index1] - loc_x_arr[index2], 2) +
      pow(loc_y_arr[index1] - loc_y_arr[index2], 2));
  }

  // Device helper function to generate observation
  __device__ void CudaCarContinuousGenerateObservation(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * direction_arr,
    float * acceleration_arr,
    int * agent_types_arr,
    const float kGridLength,
    const float kMaxSpeed,
    int * still_in_the_game_arr,
    float * obs_arr,
    int * env_timestep_arr,
    const int kNumAgents,
    const int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx,
    int * right_goal_arr
  ) {
    int num_features = 7;
    

    if (kThisAgentId < kNumAgents) {
      const int kThisAgentIdxOffset = kEnvId * kNumAgents *
       (num_features * (kNumAgents - 1) + 4) +
        kThisAgentId * (num_features * (kNumAgents - 1) + 4);
      int index = 0;
      for (int other_agent_id = 0; other_agent_id < kNumAgents;
        other_agent_id++) {
         if (other_agent_id != kThisAgentId) {
          obs_arr[kThisAgentIdxOffset + 0 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 1 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 2 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 3 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 4 * (kNumAgents - 1) + index]
            = 0.0;
          obs_arr[kThisAgentIdxOffset + 5 * (kNumAgents - 1) + index]
            = agent_types_arr[other_agent_id];
          obs_arr[kThisAgentIdxOffset + 6 * (kNumAgents - 1) + index]
            = still_in_the_game_arr[kEnvId * kNumAgents + other_agent_id];
          index += 1;
        }
      }
      obs_arr[kThisAgentIdxOffset + num_features * (kNumAgents - 1)] = 0.0;

        // Update obs for agents still in the game
      if (still_in_the_game_arr[kThisAgentArrayIdx]) {
        int index = 0;
        for (int other_agent_id = 0; other_agent_id < kNumAgents;
        other_agent_id++) {
          if (other_agent_id != kThisAgentId) {
            const int kOtherAgentArrayIdx = kEnvId * kNumAgents +
              other_agent_id;
            obs_arr[kThisAgentIdxOffset + 0 * (kNumAgents - 1) + index] =
              static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] -
              loc_x_arr[kThisAgentArrayIdx]) / (sqrt(2.0) * kGridLength);
            obs_arr[kThisAgentIdxOffset + 1 * (kNumAgents - 1) + index] =
              static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] -
              loc_y_arr[kThisAgentArrayIdx]) / (sqrt(2.0) * kGridLength);
            obs_arr[kThisAgentIdxOffset + 2 * (kNumAgents - 1) + index] =
              static_cast<float>(speed_arr[kOtherAgentArrayIdx] -
              speed_arr[kThisAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 3 * (kNumAgents - 1) + index] =
              static_cast<float>(acceleration_arr[kOtherAgentArrayIdx] -
              acceleration_arr[kThisAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 4 * (kNumAgents - 1) + index] =
              static_cast<float>(direction_arr[kOtherAgentArrayIdx] -
              direction_arr[kThisAgentArrayIdx]) / (kTwoPi);
            index += 1;
          }
        }
        obs_arr[kThisAgentIdxOffset + num_features * (kNumAgents - 1)] =
          static_cast<float>(loc_x_arr[right_goal_arr[kThisAgentArrayIdx]]) / (sqrt(2.0)* kGridLength);
        obs_arr[kThisAgentIdxOffset + num_features * (kNumAgents - 1) + 1] =
          static_cast<float>(loc_y_arr[right_goal_arr[kThisAgentArrayIdx]]) / (sqrt(2.0)* kGridLength);
        obs_arr[kThisAgentIdxOffset + num_features * (kNumAgents - 1) + 2] =
          static_cast<float>(agent_types_arr[kThisAgentArrayIdx]);
        obs_arr[kThisAgentIdxOffset + num_features * (kNumAgents - 1) + 3] =
          static_cast<float>(env_timestep_arr[kEnvId]) / kEpisodeLength;
      }
    } 
  }
  

  // Device helper function to compute rewards
  __device__ void CudaCarContinuousComputeReward(
    float * rewards_arr,
    float * loc_x_arr,
    float * loc_y_arr,
    const float kGridLength,
    float * edge_hit_reward_penalty,
    float * step_rewards_arr,
    int * agent_types_arr,
    int * still_in_the_game_arr,
    int * done_arr,
    int * env_timestep_arr,
    const int kNumAgents,
    int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx,
    const float distance_margin_for_reward,
    const float reward_car_collision,
    const float  reward_car_goal,
    const float  reward_car_right_goal,
    float * reward_people_walk_arr,
    const float kMaxSpeed,
    int * right_goal_arr
  ) {
    if (kThisAgentId < kNumAgents) {
      // initialize rewards
      // Ensure that all the agents rewards are initialized before we proceed.
      // The rewards are only set by the runners, so this pause is necessary.
      rewards_arr[kThisAgentArrayIdx] = 0.0;
      __sync_env_threads();
      if (still_in_the_game_arr[kThisAgentArrayIdx]) {
        // Add the edge hit penalty and the  step rewards / penalties
        rewards_arr[kThisAgentArrayIdx] = edge_hit_reward_penalty[kThisAgentArrayIdx];
        rewards_arr[kThisAgentArrayIdx] = step_rewards_arr[kThisAgentArrayIdx];
      }

      float min_dist = kGridLength * sqrt(2.0);
      bool is_car = (agent_types_arr[kThisAgentId] == 0);
      bool is_people = (agent_types_arr[kThisAgentId] == 3);
      if (is_car) {
        for (int other_agent_id = 0; other_agent_id < kNumAgents;
          other_agent_id++) {
          bool is_other_car = (agent_types_arr[other_agent_id] == 0);
          bool is_goal = (agent_types_arr[other_agent_id] == 1);
          bool is_wall = (agent_types_arr[other_agent_id] == 2);
          bool is_people = (agent_types_arr[other_agent_id] == 3);
          const float dist = ComputeDistance(
            loc_x_arr,
            loc_y_arr,
            kThisAgentId,
            other_agent_id,
            kEnvId,
            kNumAgents
          );
          
          if (dist < distance_margin_for_reward)
          {
            if (is_other_car && other_agent_id != kThisAgentId )
            {
              rewards_arr[kThisAgentArrayIdx] += reward_car_collision;
            }
            else if (is_goal)
            {
              if (right_goal_arr[kThisAgentId] == other_agent_id)
              {
               rewards_arr[kThisAgentArrayIdx] += reward_car_right_goal;
              }
              else
              {
               rewards_arr[kThisAgentArrayIdx] += reward_car_goal;
              }
              still_in_the_game_arr[kEnvId * kNumAgents + other_agent_id] = 0;
          
            }
            else if(is_wall || is_people)
            {
              rewards_arr[kThisAgentArrayIdx] += reward_car_collision;
            }
          }
        }
      }
      if (is_people){
        rewards_arr[kThisAgentArrayIdx] += reward_people_walk_arr[kThisAgentArrayIdx] ;
      }

      // Wait here to update the number of runners before determining done_arr
      __sync_env_threads();
      // Use only agent 0's thread to set done_arr
      if (kThisAgentId == 0 && env_timestep_arr[kEnvId] == kEpisodeLength) {
        done_arr[kEnvId] = 1;
      }
    }
  }

  __global__ void CudaCarContinuousStep(
    int * done_arr,
    int * env_timestep_arr,
    int * agent_types_arr,
    int * right_goal_arr,
    int * still_in_the_game_arr,
    int * action_indices_arr,
    float * obs_arr,
    float * rewards_arr,
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * direction_arr,
    float * acceleration_arr,
    float * step_rewards_arr,
    float * acceleration_actions_arr,
    float * turn_actions_arr,
    float * reward_people_walk_arr,
    float * skill_levels_arr,
    float * edge_hit_reward_penalty,
    const float kEdgeHitPenalty,
    const float kGridLength,
    const float kMaxSpeed,
    const float distance_margin_for_reward,
    const float reward_car_step_penalty,
    const float reward_car_collision,
    const float reward_car_goal,
    const float reward_car_right_goal,
    int kNumAgents,
    int kEpisodeLength
  ) {
    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;
    const int kNumActions = 2;

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
      float delta_turn = turn_actions_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 1]];

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

      if (agent_types_arr[kThisAgentId] == 0)
      {
        step_rewards_arr[kThisAgentArrayIdx] = reward_car_step_penalty;
        loc_x_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] * cos(direction_arr[kThisAgentArrayIdx]);
        loc_y_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] * sin(direction_arr[kThisAgentArrayIdx]);
      }
      else if (agent_types_arr[kThisAgentId] == 3)
      {
        step_rewards_arr[kThisAgentArrayIdx] = 0;
        loc_x_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] * cos(direction_arr[kThisAgentArrayIdx]);
        loc_y_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] * sin(direction_arr[kThisAgentArrayIdx]);
        reward_people_walk_arr[kThisAgentArrayIdx] = speed_arr[kThisAgentArrayIdx] / kMaxSpeed;
      }
      else
      {
        step_rewards_arr[kThisAgentArrayIdx] = 0;
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

        edge_hit_reward_penalty[kThisAgentArrayIdx] = kEdgeHitPenalty;
      } else {
        edge_hit_reward_penalty[kThisAgentArrayIdx] = 0.0;
      }
    }

    // Make sure all agents have updated their states
    __sync_env_threads();
    // -------------------------------
    // Generate observation
    // -------------------------------
    CudaCarContinuousGenerateObservation(
      loc_x_arr,
      loc_y_arr,
      speed_arr,
      direction_arr,
      acceleration_arr,
      agent_types_arr,
      kGridLength,
      kMaxSpeed,
      still_in_the_game_arr,
      obs_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx,
      right_goal_arr
      );

    // -------------------------------
    // Compute reward
    // -------------------------------
    CudaCarContinuousComputeReward(
      rewards_arr,
      loc_x_arr,
      loc_y_arr,
      kGridLength,
      edge_hit_reward_penalty,
      step_rewards_arr,
      agent_types_arr,
      still_in_the_game_arr,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx,
      distance_margin_for_reward,
      reward_car_collision,
      reward_car_goal,
      reward_car_right_goal,
      reward_people_walk_arr,
      kMaxSpeed,
      right_goal_arr
    );
  }
}
