open Core

let gaussian_perturbation ~x ~t =
  let noise = Tor_stats.norml ~mu:0. ~sigma:1. (Array.length x) in
  Array.map2_exn ~f:(+.) x (Array.map ~f:(fun x -> x *. sqrt t) noise)

let energy_discrepancy ~data ~energy ~t ~m ~w =
  let n = Array.length data in
  let log_sum_exp xs =
    let max_x = Array.max_elt xs ~f:Fn.id |> Option.value_exn in
    max_x +. log (Array.sum (module Float) xs ~f:(fun x -> exp (x -. max_x)))
  in
  let loss =
    Sequence.sum (module Float) (Array.to_sequence data) ~f:(fun x ->
        let x_t = gaussian_perturbation ~x ~t in
        let perturbed_energies =
          Array.init m ~f:(fun _ ->
              let perturbed_x = gaussian_perturbation ~x:x_t ~t in
              exp (energy perturbed_x -. energy x_t))
        in
        log ((w /. Float.of_int m) +. (1. /. Float.of_int m) *. Float.of_int (Array.length perturbed_energies) *. exp (-energy x) +. Float.sum (module Float) perturbed_energies ~f:Fn.id))
  in
  loss /. Float.of_int n