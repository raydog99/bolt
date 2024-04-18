open Base

(* MLE via Compositional Optimization *)
let meco data noise_dist iterations =
  let n = Array.length data in
  let u, v = ref (noise_dist ()), ref [] in

  for t = 1 to iterations do
    let z = data.((Random.int n)) in
    let z_tilde = noise_dist () in

    (* Update function value estimator *)
    u := (1.0 -. gamma t) *. !u +. gamma t *. (unnorm_model z_tilde / noise_dist_pdf z_tilde);

    (* Update gradient estimator *)
    let grad_z = -(unnorm_model_grad z / unnorm_model z) in
    let grad_z_tilde = (unnorm_model_grad z_tilde / noise_dist_pdf z_tilde) /. !u in
    v := (1.0 -. beta t) *. !v +. beta t *. (grad_z +. grad_z_tilde);

    (* Update model parameters *)
    update_params !v
done