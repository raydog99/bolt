open Torch

module SingleStepFEBM = struct
  type t = {
    mutable energy_function: Tensor.t -> Tensor.t;
    mutable variational_function: Tensor.t -> Tensor.t;
    mutable theta: Tensor.t;
    mutable omega: Tensor.t;
  }

  let create energy_init variational_init =
    {
      energy_function = energy_init;
      variational_function = variational_init;
      theta = Tensor.randn [1];
      omega = Tensor.randn [1];
    }

  let draw_minibatch data batch_size =
    let n = Tensor.shape data |> List.hd in
    let indices = Tensor.randint ~high:n [batch_size] ~dtype:(T Int64) in
    Tensor.index_select data ~dim:0 ~index:indices

  let langevin_dynamics q_theta num_steps step_size =
    let rec loop x steps =
      if steps = 0 then x
      else
        let noise = Tensor.randn_like x in
        let grad = Tensor.grad x in
        let x' = Tensor.(x - (grad * float step_size) + (noise * sqrt (float (2. *. step_size)))) in
        loop x' (steps - 1)
    in
    loop q_theta num_steps

  let estimate_f_ebm_loss t d_p d_q =
    let energy_p = t.energy_function d_p in
    let energy_q = t.energy_function d_q in
    let log_q = t.variational_function d_q in
    Tensor.(mean (energy_p - energy_q - log_q))

  let estimate_equation_18 t d_p d_q =
    let energy_p = t.energy_function d_p in
    let energy_q = t.energy_function d_q in
    Tensor.(mean (energy_p - energy_q))

  let train t p_data num_iterations batch_size learning_rate =
    let optimizer = Optimizer.adam [t.theta; t.omega] ~lr:learning_rate in
    for _ = 1 to num_iterations do
      let d_p = draw_minibatch p_data batch_size in
      let q_0 = Tensor.randn_like d_p in
      let d_q = langevin_dynamics q_0 10 0.1 in
      
      let loss = estimate_f_ebm_loss t d_p d_q in
      Optimizer.zero_grad optimizer;
      Tensor.backward loss;
      Optimizer.step optimizer;

      let eq_18 = estimate_equation_18 t d_p d_q in
      Optimizer.zero_grad optimizer;
      Tensor.backward eq_18;
      Optimizer.step optimizer;
    done

  let learned_energy_based_model t x =
    Tensor.exp (Tensor.neg (t.energy_function x))

  let density_ratio_estimator t x =
    Tensor.exp (Tensor.add (t.variational_function x) (t.energy_function x))
end