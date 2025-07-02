import time
import numpy as np
import cv2

from robot.airbot import AIRBOTPlay

from client.client import WebsocketPolicyClient

all_time_actions = None
all_actions = None

def get_act_action(obs: dict, policy: WebsocketPolicyClient, stats: dict, state_dim: int, num_queries: int, 
                   temporal_agg: bool, t: int, max_timesteps: int, camera_names: list) -> np.ndarray:
    global all_time_actions, all_actions
    
    pre_process = lambda s_qpos: (s_qpos - stats['state_mean']) / stats['state_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = num_queries
    if temporal_agg:
        query_frequency = 1
        num_queries = num_queries

    ### evaluation loop
    if temporal_agg and all_time_actions is None:
        all_time_actions = np.zeros([max_timesteps, max_timesteps+num_queries, state_dim])

    ### process previous timestep to get qpos and image_list
    qpos_numpy = np.array(obs["observation.state"])
    qpos = pre_process(qpos_numpy)
    qpos = np.expand_dims(qpos, axis=0)

    curr_images = []
    for cam_name in camera_names:
        # curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_image = np.transpose(obs[f"observation.images.{cam_name}"], (2, 0, 1))
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0, dtype=np.float32) / 255.0
    curr_image = np.expand_dims(curr_image, axis=0)

    ### query policy
    if t % query_frequency == 0:
        all_actions = policy.infer({
            "qpos": qpos,
            "image": curr_image,
        })
    if temporal_agg:
        all_time_actions[[t], t:t+num_queries] = all_actions
        actions_for_curr_step = all_time_actions[:, t]
        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = np.expand_dims(exp_weights, axis=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
    else:
        raw_action = all_actions[:, t % query_frequency]

    ### post-process actions
    raw_action = np.squeeze(raw_action, axis=0)
    action = post_process(raw_action)
    target_qpos = action
    
    return target_qpos

if __name__ == "__main__":
    temporal_agg: bool = True
    max_timesteps: int = 1000

    robot = AIRBOTPlay()
    print("obs1:", robot.capture_observation())
    # robot.back_home()
    # robot.send_action([0.0] * 6 + [0.2] + [0.0] * 6 + [0.2])
    # time.sleep(5)
    # robot.send_action([0.0] * 6 + [0.4] + [0.0] * 6 + [0.4])
    # time.sleep(5)
    # print("obs2:", robot.capture_observation())

    policy = WebsocketPolicyClient(
        host="192.168.3.101",
        port="8060",
    )
    metadata = policy.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    stats = metadata["stats"]
    state_dim = metadata["state_dim"]
    num_queries = metadata["policy_config"]["num_queries"]
    camera_names = metadata["camera_names"]

    # Send 1 observation to make sure the model is loaded.
    # action = policy.infer(obs_fn())
    # print(action)
    # print(action.shape)
    # print(action.dtype)

    # start = time.time()
    # for _ in range(args.num_steps):
    #     policy.infer(obs_fn())
    # end = time.time()

    start = time.time()
    for i in range(max_timesteps):
        obs = robot.capture_observation()

        for cam_name in camera_names:
            view = cv2.putText(obs[f"observation.images.{cam_name}"], f"frame {i}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.imshow(cam_name, view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        action = get_act_action(obs, policy, stats, state_dim, num_queries, temporal_agg, 
                                i, max_timesteps, camera_names)
        robot.send_action(action)

        print(f"Average FPS: {(i + 1) / (time.time() - start)} Hz")

    end = time.time()

    print(f"Total time taken: {end - start:.2f} s")
    print(f"Average inference time: {1000 * (end - start) / max_timesteps:.2f} ms")

    robot.disconnect()

    cv2.destroyAllWindows()
