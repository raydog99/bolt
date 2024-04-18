open Base

let ed_bern data epsilon =
  let n = Array.length data in
  let bernoulli_perturb x =
    let xi = ref x in
    for i = 0 to Array.length x - 1 do
      if Random.float 1.0 < epsilon then xi := Stdlib.xor xi.(i) 1
    done;
    !xi
  in
  let log_sum_exp xs =
    let max_x = Array.fold_left max (-infinity) xs in
    let sum = ref 0.0 in
    Array.iter (fun x -> sum := !sum +. exp (x -. max_x)) xs;
    max_x +. log !sum
  in
  let u_bern x =
    let samples = Array.init n (fun _ -> bernoulli_perturb x) in
    -. log_sum_exp (Array.map (fun x' -> exp (energy_function x')) samples) +. log (float_of_int n)
  in
  let loss = ref 0.0 in
  Array.iter (fun x -> loss := !loss +. (energy_function x -. u_bern x)) data;
  !loss /. (float_of_int n)