open Torch

module EnergyBasedImitationLearning = struct
  type demonstration = {
    state: Tensor.t;
    action: Tensor.t;
  }

  type t = {
    energy_model: Tensor.t -> Tensor.t -> Tensor.t;
    policy: Tensor.t -> Tensor.t;
    phi_optimizer: Optimizer.t;
    theta_optimizer: Optimizer.t;
  }

  let create energy_model policy phi_lr theta_lr =
    let phi_params = Tensor.grad_params energy_model in
    let theta_params = Tensor.grad_params policy in
    {
      energy_model;
      policy;
      phi_optimizer = Optimizer.adam phi_params ~lr:phi_lr;
      theta_optimizer = Optimizer.adam theta_params ~lr:theta_lr;
    }

  let optimize_phi t demonstrations =
    let objective = 
      List.fold_left (fun acc demo ->
        let energy = t.energy_model demo.state demo.action in
        Tensor.(add acc (neg (log (sigmoid energy))))
      ) (Tensor.zeros []) demonstrations
    in
    Optimizer.zero_grad t.phi_optimizer;
    Tensor.backward objective;
    Optimizer.step t.phi_optimizer

  let compute_surrogate_reward t state action =
    Tensor.neg (t.energy_model state action)

  let update_theta t env max_steps =
    let rec loop steps total_reward =
      if steps >= max_steps then total_reward
      else
        let state = env.get_state () in
        let action = t.policy state in
        let reward = compute_surrogate_reward t state action in
        let next_state = env.step action in
        
        let loss = Tensor.neg reward in
        Optimizer.zero_grad t.theta_optimizer;
        Tensor.backward loss;
        Optimizer.step t.theta_optimizer;

        loop (steps + 1) (Tensor.add total_reward reward)
    in
    loop 0 (Tensor.zeros [])

  let train t demonstrations env max_phi_steps max_theta_steps num_iterations =
    for _ = 1 to num_iterations do
      (* Optimize phi *)
      for _ = 1 to max_phi_steps do
        optimize_phi t demonstrations
      done;

      (* Update theta *)
      for _ = 1 to max_theta_steps do
        ignore (update_theta t env max_theta_steps)
      done
    done;
    t.policy
end