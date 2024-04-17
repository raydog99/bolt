open Core

let denoising_score_matching ?(noise_levels=[0.1; 0.5; 1.0]) ~score_model ~data_dist =
  let rec loop theta =
    let x = Distributions.draw data_dist in
    let noisy_xs = List.map noise_levels ~f:(fun sigma -> x +. sigma *. Distributions.draw Gaussian.unit_d) in
    let loss =
      List.fold noisy_xs ~init:0. ~f:(fun acc noisy_x ->
          let score = Score_model.eval ~theta noisy_x in
          let grad_score = Score_model.gradient ~theta noisy_x in
          let term1 = (noisy_x -. x) /. sigma in
          let term2 = 0.5 *. Vec.l2_norm_sqr (score -. term1) in
          acc +. term2)
    in
    let theta' = Optimization.update ~theta ~loss in
    loop theta'
  in
  loop Score_model.init_params