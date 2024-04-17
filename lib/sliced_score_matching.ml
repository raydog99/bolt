open Core

let sliced_score_matching ?(num_projections=1) ~score_model ~data_dist =
  let rec loop theta =
    let x = Distributions.draw data_dist in
    let projections = Array.init num_projections ~f:(fun _ -> Distributions.draw Gaussian.unit_d) in
    let loss =
      Array.fold projections ~init:0. ~f:(fun acc proj ->
          let score = Score_model.eval ~theta x in
          let grad_score = Score_model.gradient ~theta x in
          let term1 = Vec.dot proj grad_score in
          let term2 = 0.5 *. (Vec.dot proj score) ** 2. in
          acc +. term1 +. term2)
    in
    let theta' = Optimization.update ~theta ~loss in
    loop theta'
  in
  loop Score_model.init_params