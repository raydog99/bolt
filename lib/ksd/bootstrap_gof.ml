open Torch

module BootstrapGof = struct
  type t = {
    sample: Tensor.t;
    score_fn: Tensor.t -> Tensor.t;
    bootstrap_size: int;
  }

  let create sample score_fn bootstrap_size =
    { sample; score_fn; bootstrap_size }

  let compute_su t x x' =
    let grad_log_q_x = t.score_fn x in
    let grad_log_q_x' = t.score_fn x' in
    Tensor.(grad_log_q_x * grad_log_q_x' + (grad_log_q_x + grad_log_q_x') * (x - x'))

  let generate_bootstrap_sample t su =
    let n = Tensor.shape t.sample |> List.hd in
    let indices = Tensor.randint ~high:n [t.bootstrap_size; n] ~dtype:Int64 in
    Tensor.index_select su ~dim:0 ~index:indices

  let compute_su_hat t =
    let n = Tensor.shape t.sample |> List.hd in
    let su_matrix = Tensor.zeros [n; n] in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let xi = Tensor.select t.sample ~dim:0 ~index:i in
        let xj = Tensor.select t.sample ~dim:0 ~index:j in
        let su_ij = compute_su t xi xj in
        Tensor.set su_matrix [|i; j|] su_ij
      done
    done;
    Tensor.mean su_matrix

  let test t alpha =
    let su_hat = compute_su_hat t in
    let bootstrap_samples = generate_bootstrap_sample t su_hat in
    let count = Tensor.(sum (bootstrap_samples > su_hat)) in
    let percentage = Tensor.to_float0_exn count /. float_of_int t.bootstrap_size in
    percentage < alpha
end